#!/usr/bin/env python3
"""Training loop for the RIS-FLoc MixedModel."""

from __future__ import annotations
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from .config import Config
from .data import ClassBalancedIndexSampler, RSSDataset, rss_collate_fn, _picture_output_path
from .model import MixedModel
from .utils import _amp_autocast_ctx, _make_grad_scaler, unwrap_model

def train_model():
    print('\n[train] loading datasets...')
    train_ds = RSSDataset(Config.TRAIN_PATH, is_train=True)
    val_ds = RSSDataset(Config.VAL_PATH, mean=train_ds.mean, std=train_ds.std, is_train=False)
    nc_s = len(Config.RP_LIST)
    y_samp = np.asarray(train_ds.labels, dtype=np.int64)
    nw = int(getattr(Config, 'DATALOADER_NUM_WORKERS', 4))
    if getattr(Config, 'USE_WEIGHTED_SAMPLER', False):
        cnt = np.bincount(np.clip(y_samp, 0, nc_s - 1), minlength=nc_s).astype(np.float64)
        w_cls = 1.0 / np.maximum(cnt, 1.0)
        samp_w = w_cls[y_samp]
        n_train = len(samp_w)
        if n_train >= 2 ** 24:
            sampler = ClassBalancedIndexSampler(y_samp, nc_s, num_samples=n_train)
            print(f'[sampler] class-balanced index (N={n_train} ≥ 2^24) | {nc_s} RP classes')
        else:
            sampler = WeightedRandomSampler(weights=torch.as_tensor(samp_w, dtype=torch.double), num_samples=n_train, replacement=True)
            print(f'[sampler] weighted | {nc_s} RP classes')
        train_loader = DataLoader(train_ds, Config.BATCH_SIZE, sampler=sampler, shuffle=False, collate_fn=rss_collate_fn, num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    else:
        train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, collate_fn=rss_collate_fn, num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, collate_fn=rss_collate_fn, num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    model = MixedModel().to(Config.DEVICE)
    use_dp = getattr(Config, 'USE_DATA_PARALLEL', False) and torch.cuda.is_available() and (torch.cuda.device_count() > 1)
    if not use_dp and getattr(Config, 'USE_TORCH_COMPILE', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='default')
            print('[compile] torch.compile (single GPU)')
        except Exception as e:
            print(f'[compile] skipped: {e}')
    if use_dp:
        model = nn.DataParallel(model)
        print(f'[gpu] DataParallel on {torch.cuda.device_count()} device(s)')
    ls = float(getattr(Config, 'LABEL_SMOOTHING', 0.0))
    weight_tensor = None
    if getattr(Config, 'USE_CLASS_WEIGHTS', False):
        cnt = np.bincount(np.clip(y_samp, 0, len(Config.RP_LIST) - 1), minlength=len(Config.RP_LIST)).astype(np.float64)
        cw = 1.0 / np.maximum(cnt, 1.0)
        cw = cw * (len(cw) / np.sum(cw))
        weight_tensor = torch.tensor(cw, dtype=torch.float32, device=Config.DEVICE)
    crit_cls = nn.CrossEntropyLoss(label_smoothing=ls, weight=weight_tensor)
    crit_xy = nn.SmoothL1Loss(beta=1.0)
    opt = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.001, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(opt, T_max=Config.EPOCHS, eta_min=1e-06)
    try:
        plateau_scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
    except TypeError:
        plateau_scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    use_amp = bool(getattr(Config, 'USE_AMP', True)) and torch.cuda.is_available()
    scaler = _make_grad_scaler(use_amp)
    if use_amp:
        print('[train] AMP (autocast + GradScaler) enabled')
    train_losses, train_accs, val_losses, val_accs = ([], [], [], [])
    best_metric_val = 0.0
    patience, patience_counter = (8, 0)
    cls_disp = f'{nc_s}-RP'
    tqdm.write(f'\n[train] epochs={Config.EPOCHS} | val: {cls_disp} Top-1')
    epoch_pbar = tqdm(range(Config.EPOCHS), desc='Epoch', unit='ep', dynamic_ncols=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = train_correct = train_total = 0.0
        train_pbar = tqdm(train_loader, desc=f'Train {epoch + 1}/{Config.EPOCHS}', leave=False, dynamic_ncols=True)
        for batch in train_pbar:
            x, lens, y, xy_gt = batch
            x, y = (x.to(Config.DEVICE), y.to(Config.DEVICE))
            xy_gt = xy_gt.to(Config.DEVICE).float()
            opt.zero_grad(set_to_none=True)
            with _amp_autocast_ctx(use_amp):
                outputs, pred_xy, _ = model(x, lens)
                loss = crit_cls(outputs, y) + Config.LOSS_XY_WEIGHT * crit_xy(pred_xy, xy_gt)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            if train_total > 0:
                train_pbar.set_postfix(loss=f'{train_loss / train_total:.4f}', acc=f'{100.0 * train_correct / train_total:.1f}%')
        scheduler.step()
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total * 100
        train_losses.append(train_avg_loss)
        train_accs.append(train_acc)
        model.eval()
        val_loss = val_correct = val_total = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Val', leave=False, dynamic_ncols=True)
            for batch in val_pbar:
                x, lens, y, xy_gt = batch
                x, y = (x.to(Config.DEVICE), y.to(Config.DEVICE))
                xy_gt = xy_gt.to(Config.DEVICE).float()
                with _amp_autocast_ctx(use_amp):
                    outputs, pred_xy, _ = model(x, lens)
                    loss = crit_cls(outputs, y) + Config.LOSS_XY_WEIGHT * crit_xy(pred_xy, xy_gt)
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                if val_total > 0:
                    val_pbar.set_postfix(loss=f'{val_loss / val_total:.4f}', acc=f'{100.0 * val_correct / val_total:.1f}%')
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_avg_loss)
        val_accs.append(val_acc)
        plateau_scheduler.step(val_acc)
        if val_acc > best_metric_val:
            best_metric_val = val_acc
            patience_counter = 0
            torch.save(unwrap_model(model).state_dict(), Config.OUTPUT_MODEL_PTH)
            tqdm.write(f'[best] val {cls_disp} = {best_metric_val:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f'\n[early-stop] best val {cls_disp}: {best_metric_val:.2f}%')
                break
        epoch_pbar.set_postfix(tr_loss=f'{train_avg_loss:.4f}', tr_acc=f'{train_acc:.1f}%', va_loss=f'{val_avg_loss:.4f}', va_acc=f'{val_acc:.1f}%', best=f'{best_metric_val:.1f}%')
    epoch_pbar.close()
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=300)
    er = range(1, len(train_losses) + 1)
    ax1.plot(er, train_losses, 'b-', label='train loss')
    ax1.plot(er, val_losses, 'r-', label='val loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(er, train_accs, 'b-', label=f'train {cls_disp}')
    ax2.plot(er, val_accs, 'r-', label=f'val {cls_disp}')
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    p_train_curves = _picture_output_path(Config.OUTPUT_TRAINING_CURVES_PNG)
    plt.savefig(p_train_curves, bbox_inches='tight')
    plt.close()
    print(f'\n[saved] {p_train_curves}')
    try:
        from .evaluate import save_validation_diagnostic_plots
        save_validation_diagnostic_plots(val_loader, Config.DEVICE, use_amp, len(Config.RP_LIST), Config.RP_LIST)
    except Exception as ex:
        print(f'\n[plots] validation diagnostics skipped: {ex}')
    np.save(Config.OUTPUT_MEAN_NPY, train_ds.mean)
    np.save(Config.OUTPUT_STD_NPY, train_ds.std)
    vc_path = Config.VALID_RP_CLASSES_JSON
    uniq = np.unique(np.asarray(train_ds.labels, dtype=np.int64)).astype(int).tolist()
    with open(vc_path, 'w', encoding='utf-8') as f:
        json.dump(uniq, f)
    print(f'\n[saved] {vc_path} (class indices seen in training; used as test mask)')
    return (model, train_ds.mean, train_ds.std)
