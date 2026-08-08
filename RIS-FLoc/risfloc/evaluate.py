#!/usr/bin/env python3
"""Test-set localization evaluation and diagnostic plots."""

from __future__ import annotations
import json, math, os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .config import Config
from .data import (_cache_prefix, _depth_from_seq_length, _path_depth_label, _picture_output_path, _resolve_test_depths, _test_drl_cache_filenames, build_test_drl_cache_from_scan)
from .normalize import z_score_apply_phase_grid
from .model import MixedModel
from .utils import _amp_autocast_ctx

def _errors_within_cap(errors, cap_m):
    if not errors:
        return np.array([], dtype=np.float64)
    a = np.asarray(errors, dtype=np.float64)
    if cap_m is None or cap_m <= 0:
        return a
    return a[a <= cap_m]

def _eval_topk_weight_mode() -> str:
    mode = str(getattr(Config, 'EVAL_TOPK_WEIGHT_MODE', 'topk_softmax') or 'topk_softmax').strip().lower()
    if mode in ('full_softmax_no_renorm', 'full_softmax', 'approach_b', 'b'):
        return 'full_softmax_no_renorm'
    return 'topk_softmax'

def _eval_topk_prob_mass_div() -> float:
    d = float(getattr(Config, 'EVAL_TOPK_PROB_MASS_DIV', 1.0) or 1.0)
    return max(d, 1e-06)

def _apply_topk_prob_mass_scale(probs: torch.Tensor) -> torch.Tensor:
    div = _eval_topk_prob_mass_div()
    if div == 1.0:
        return probs
    return probs / div

def apply_eval_localization_config(*, topk_weight_mode: Optional[str]=None, prob_mass_div: Optional[float]=None) -> None:
    if topk_weight_mode is not None:
        Config.EVAL_TOPK_WEIGHT_MODE = str(topk_weight_mode)
    if prob_mass_div is not None:
        Config.EVAL_TOPK_PROB_MASS_DIV = max(float(prob_mass_div), 1e-06)

def _weighted_rp_coords_from_logits(outputs, coord_df, batch_idx: int=0):
    c = outputs.size(1)
    top_k_use = min(int(Config.K_VALUE), int(c))
    if _eval_topk_weight_mode() == 'full_softmax_no_renorm':
        scaled = outputs[int(batch_idx)].detach().cpu().numpy().astype(np.float64, copy=False)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / (np.sum(probs) + 1e-12)
        mass_div = _eval_topk_prob_mass_div()
        if mass_div != 1.0:
            probs = probs / mass_div
        top_idx = np.argsort(-probs)[:top_k_use]
        weights = probs[top_idx]
    else:
        top_vals, top_idx_t = torch.topk(outputs, top_k_use, dim=1)
        top_vals = top_vals[int(batch_idx)].detach().cpu().numpy()
        top_idx = top_idx_t[int(batch_idx)].detach().cpu().numpy()
        top_vals = top_vals - np.max(top_vals)
        w = np.exp(top_vals)
        weights = w / (np.sum(w) + 1e-12)
        mass_div = _eval_topk_prob_mass_div()
        if mass_div != 1.0:
            weights = weights / mass_div
    px_w, py_w = (0.0, 0.0)
    for i in range(top_k_use):
        rp_idx = int(top_idx[i])
        rp = Config.RP_LIST[rp_idx]
        px_w += float(weights[i]) * float(coord_df.loc[0, rp])
        py_w += float(weights[i]) * float(coord_df.loc[1, rp])
    return (px_w, py_w)

def _build_rp_xy_lut_tensor(coord_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    n = len(Config.RP_LIST)
    lut = torch.zeros(n, 2, dtype=torch.float32)
    for j, rp in enumerate(Config.RP_LIST):
        lut[j, 0] = float(coord_df.loc[0, rp])
        lut[j, 1] = float(coord_df.loc[1, rp])
    return lut.to(device)

def _load_valid_class_indices(num_classes: int) -> List[int]:
    path = getattr(Config, 'VALID_RP_CLASSES_JSON', 'valid_rp_classes.json')
    if not os.path.isfile(path):
        return list(range(int(num_classes)))
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    out: List[int] = []
    for x in raw:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < int(num_classes):
            out.append(xi)
    return out if out else list(range(int(num_classes)))

def _valid_class_mask_tensor(num_classes: int, device: torch.device) -> torch.Tensor:
    idxs = _load_valid_class_indices(num_classes)
    m = torch.zeros(int(num_classes), dtype=torch.bool, device=device)
    for i in idxs:
        m[int(i)] = True
    if not bool(m.any()):
        m[:] = True
    return m

def _soft_weighted_xy_from_logits_batch_topk_softmax(outputs: torch.Tensor, xy_lut: torch.Tensor) -> torch.Tensor:
    c = int(outputs.size(1))
    k = min(int(Config.K_VALUE), c)
    lut = xy_lut.to(dtype=outputs.dtype)
    top_vals, top_idx = torch.topk(outputs, k, dim=1)
    tv = top_vals - top_vals.max(dim=1, keepdim=True).values
    w = torch.softmax(tv, dim=1)
    w = _apply_topk_prob_mass_scale(w)
    gathered = lut[top_idx.long()]
    return (w.unsqueeze(-1) * gathered).sum(dim=1)

def _soft_weighted_xy_from_logits_batch_full_softmax_no_renorm(outputs: torch.Tensor, xy_lut: torch.Tensor) -> torch.Tensor:
    c = int(outputs.size(1))
    k = min(int(Config.K_VALUE), c)
    lut = xy_lut.to(dtype=outputs.dtype)
    probs = torch.softmax(outputs, dim=1)
    probs = _apply_topk_prob_mass_scale(probs)
    top_probs, top_idx = torch.topk(probs, k, dim=1)
    gathered = lut[top_idx.long()]
    return (top_probs.unsqueeze(-1) * gathered).sum(dim=1)

def _soft_weighted_xy_from_logits_batch(outputs: torch.Tensor, xy_lut: torch.Tensor) -> torch.Tensor:
    if _eval_topk_weight_mode() == 'full_softmax_no_renorm':
        return _soft_weighted_xy_from_logits_batch_full_softmax_no_renorm(outputs, xy_lut)
    return _soft_weighted_xy_from_logits_batch_topk_softmax(outputs, xy_lut)

class _TestEvalCachedDataset(Dataset):

    def __init__(self, lens: np.ndarray, sequences_padded: np.ndarray, true_rp_names: np.ndarray, true_xy: np.ndarray, mean: np.ndarray, std: np.ndarray, depths: Optional[np.ndarray]=None):
        self.lens = np.asarray(lens, dtype=np.int32)
        self.depths = np.asarray(depths, dtype=np.int8).reshape(-1) if depths is not None else np.asarray([_depth_from_seq_length(int(L)) for L in self.lens], dtype=np.int8)
        self.sequences = sequences_padded
        self.n_phases = int(Config.NUM_PHASES)
        if self.sequences.ndim == 2:
            raise ValueError('Legacy 1D test cache; delete *_DRL_* caches and rescan Test_Data.')
        self.true_rp_names = true_rp_names
        self.true_xy = np.asarray(true_xy, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.lens.shape[0])

    def __getitem__(self, idx: int):
        L = int(self.lens[idx])
        rss = np.asarray(self.sequences[idx, :, :L], dtype=np.float32).copy()
        rss_n = z_score_apply_phase_grid(rss, self.mean, self.std)
        xy = self.true_xy[idx]
        rp = str(self.true_rp_names[idx])
        depth = int(self.depths[idx])
        return (torch.as_tensor(rss_n, dtype=torch.float32), L, torch.as_tensor(xy, dtype=torch.float32), rp, depth)

def test_eval_collate_fn(batch):
    xs, lens_list, xys, rps, depths = zip(*batch)
    lens_t = torch.tensor(lens_list, dtype=torch.long)
    max_l = int(lens_t.max().item())
    B = len(xs)
    n_ph = int(Config.NUM_PHASES)
    padded = torch.zeros(B, n_ph, max_l, dtype=torch.float32)
    for i, (xv, L) in enumerate(zip(xs, lens_list)):
        L = int(L)
        padded[i, :, :L] = xv[:, :L].float()
    return (padded, lens_t, torch.stack(list(xys)), list(rps), list(depths))

def _evaluate_localization_batched(model: nn.Module, coord: pd.DataFrame, lens: np.ndarray, sequences_padded: np.ndarray, true_rp_names: np.ndarray, true_xy: np.ndarray, mean: np.ndarray, std: np.ndarray, use_amp: bool, official_rp_set: Set[str], valid_mask: torch.Tensor, depths: Optional[np.ndarray]=None) -> Tuple[List[float], List[float], List[float], int, int, List[str], List[int]]:
    device = Config.DEVICE
    nb = torch.cuda.is_available()
    te_bs = getattr(Config, 'TEST_EVAL_BATCH_SIZE', None)
    bs = int(te_bs) if te_bs is not None else int(Config.BATCH_SIZE)
    bs = max(bs, 1)
    tw = getattr(Config, 'TEST_DATALOADER_NUM_WORKERS', None)
    if tw is not None:
        nw = max(0, int(tw))
    else:
        nw = min(int(getattr(Config, 'DATALOADER_NUM_WORKERS', 4)), 32)
    ds = _TestEvalCachedDataset(lens, sequences_padded, true_rp_names, true_xy, mean, std, depths=depths)
    dl_kw: dict = dict(num_workers=nw, pin_memory=nb, persistent_workers=nw > 0)
    if nw > 0:
        dl_kw['prefetch_factor'] = max(2, int(getattr(Config, 'TEST_DATALOADER_PREFETCH_FACTOR', 4)))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=test_eval_collate_fn, **dl_kw)
    xy_lut = _build_rp_xy_lut_tensor(coord, device)
    lut32 = xy_lut.to(dtype=torch.float32)
    vm = valid_mask.to(device=device, dtype=torch.bool)
    lam = float(getattr(Config, 'EVAL_FUSE_XY_WEIGHT', 0.72))
    wt_mode = _eval_topk_weight_mode()
    mass_div = _eval_topk_prob_mass_div()
    print(f'[eval] test batch_size={bs} dataloader_workers={nw} | Top-{int(Config.K_VALUE)} weight_mode={wt_mode} prob_mass_div={mass_div:.2f} + fuse_xy lam={lam:.2f} (valid masked classes={int(vm.sum().item())})')
    all_errors: List[float] = []
    known_errors: List[float] = []
    unknown_errors: List[float] = []
    rp_names_ordered: List[str] = []
    depths_ordered: List[int] = []
    cls_correct_known = 0
    cls_total_known = 0
    with torch.no_grad():
        for x, lens_t, xy_gt, rp_batch, depth_batch in tqdm(loader, desc='Test (batched GPU)', unit='batch', dynamic_ncols=True):
            x = x.to(device, non_blocking=nb)
            xy_gt = xy_gt.to(device, non_blocking=nb).float()
            with _amp_autocast_ctx(use_amp):
                outputs, pred_xy, _ = model(x, lens_t)
            masked = outputs.float().clone()
            masked[:, ~vm] = float('-inf')
            pred_idx = torch.argmax(masked, dim=1)
            pxy_soft = _soft_weighted_xy_from_logits_batch(masked, xy_lut)
            pxy_fused = lam * pred_xy.float() + (1.0 - lam) * pxy_soft.float()
            err_vec = torch.linalg.vector_norm(pxy_fused - xy_gt, dim=1)
            errs = err_vec.detach().cpu().numpy()
            all_errors.extend(errs.tolist())
            rp_names_ordered.extend([str(r) for r in rp_batch])
            depths_ordered.extend([int(d) for d in depth_batch])
            is_known = np.array([str(rp) in official_rp_set for rp in rp_batch], dtype=bool)
            if bool(np.any(is_known)):
                known_errors.extend(errs[is_known].tolist())
            if bool(np.any(~is_known)):
                unknown_errors.extend(errs[~is_known].tolist())
            B = int(outputs.size(0))
            for b in range(B):
                if not is_known[b]:
                    continue
                cls_total_known += 1
                if Config.RP_LIST[int(pred_idx[b].item())] == str(rp_batch[b]):
                    cls_correct_known += 1
    return (all_errors, known_errors, unknown_errors, cls_correct_known, cls_total_known, rp_names_ordered, depths_ordered)

def _load_model_state(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def _confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        ti, pi = (int(t), int(p))
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1
    return cm

def save_validation_diagnostic_plots(val_loader: DataLoader, device: torch.device, use_amp: bool, num_classes: int, rp_list: Sequence[str]) -> None:
    coord = pd.read_csv(Config.ORDINATE_CSV)
    lam = float(getattr(Config, 'EVAL_FUSE_XY_WEIGHT', 0.72))
    diag_model = MixedModel().to(device)
    diag_model.load_state_dict(_load_model_state(Config.OUTPUT_MODEL_PTH, device))
    diag_model.eval()
    ys_list: List[np.ndarray] = []
    pr_list: List[np.ndarray] = []
    true_xy_list: List[Tuple[float, float]] = []
    err_list: List[float] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val diagnostics', leave=False, dynamic_ncols=True):
            x, lens, y, xy_gt = batch
            x = x.to(device)
            y = y.to(device)
            xy_gt = xy_gt.to(device).float()
            with _amp_autocast_ctx(use_amp):
                outputs, pred_xy, _ = diag_model(x, lens)
            pred = torch.argmax(outputs, dim=1)
            ys_list.append(y.detach().cpu().numpy())
            pr_list.append(pred.detach().cpu().numpy())
            B = int(y.size(0))
            for b in range(B):
                px_w, py_w = _weighted_rp_coords_from_logits(outputs, coord, batch_idx=b)
                px = lam * float(pred_xy[b, 0].detach().cpu()) + (1.0 - lam) * px_w
                py = lam * float(pred_xy[b, 1].detach().cpu()) + (1.0 - lam) * py_w
                txx = float(xy_gt[b, 0].detach().cpu())
                tyy = float(xy_gt[b, 1].detach().cpu())
                err_list.append(math.hypot(px - txx, py - tyy))
                true_xy_list.append((txx, tyy))
    y_true = np.concatenate(ys_list, axis=0)
    y_pred = np.concatenate(pr_list, axis=0)
    cm = _confusion_matrix_np(y_true, y_pred, num_classes)
    support = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(support > 0, np.diag(cm).astype(np.float64) / support, np.nan)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    fig_cm, ax_cm = plt.subplots(figsize=(16, 14), dpi=200)
    vmax = max(int(np.max(cm)), 1)
    im = ax_cm.imshow(cm, cmap='Blues', norm=Normalize(vmin=0, vmax=vmax), aspect='equal', interpolation='nearest')
    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Count', rotation=270, labelpad=18)
    tick_idx = np.arange(num_classes)
    ax_cm.set_xticks(tick_idx)
    ax_cm.set_yticks(tick_idx)
    ax_cm.set_xticklabels(list(rp_list), rotation=90, fontsize=6)
    ax_cm.set_yticklabels(list(rp_list), fontsize=6)
    ax_cm.set_xlabel('Predicted RP', fontsize=11)
    ax_cm.set_ylabel('True RP', fontsize=11)
    ax_cm.set_title('Validation confusion matrix (best checkpoint)', fontsize=13, pad=12)
    fig_cm.tight_layout()
    p_val_cm = _picture_output_path(Config.OUTPUT_VAL_CONFUSION_PNG)
    fig_cm.savefig(p_val_cm, bbox_inches='tight', facecolor='white')
    plt.close(fig_cm)
    fig_b, (ax_s, ax_a) = plt.subplots(2, 1, figsize=(18, 10), dpi=200, sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12})
    x_idx = np.arange(num_classes)
    colors_s = plt.cm.viridis(np.linspace(0.25, 0.85, num_classes))
    ax_s.bar(x_idx, support, color=colors_s, edgecolor='white', linewidth=0.4)
    ax_s.set_ylabel('Support (#val samples)', fontsize=11)
    ax_s.set_title('Per-class support (validation)', fontsize=12)
    ax_s.set_xlim(-0.6, num_classes - 0.4)
    acc_pct = per_class_acc * 100.0
    acc_pct_draw = np.nan_to_num(acc_pct, nan=0.0)
    bar_colors_a: List[Tuple[float, float, float, float]] = []
    for i in range(num_classes):
        if support[i] <= 0:
            bar_colors_a.append((0.82, 0.82, 0.82, 1.0))
        else:
            bar_colors_a.append(plt.cm.RdYlGn(float(np.clip(acc_pct[i], 0.0, 100.0)) / 100.0))
    ax_a.bar(x_idx, acc_pct_draw, color=bar_colors_a, edgecolor='white', linewidth=0.4)
    ax_a.axhline(100.0 / num_classes, color='crimson', linestyle='--', linewidth=1.0, alpha=0.7, label=f'random {100.0 / num_classes:.1f}%')
    ax_a.set_ylabel('Top-1 accuracy (%)', fontsize=11)
    ax_a.set_xlabel('RP index (same order as ordinate.csv columns)', fontsize=11)
    ax_a.set_title('Per-class Top-1 accuracy (validation)', fontsize=12)
    ax_a.set_ylim(0, min(105, max(10, float(np.nanmax(acc_pct)) * 1.15) if np.any(np.isfinite(acc_pct)) else 100))
    ax_a.legend(loc='lower right', fontsize=9)
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(list(rp_list), rotation=75, ha='right', fontsize=6)
    plt.setp(ax_s.get_xticklabels(), visible=False)
    fig_b.align_labels()
    p_val_pc = _picture_output_path(Config.OUTPUT_VAL_PER_CLASS_PNG)
    fig_b.savefig(p_val_pc, bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    tx = np.array([p[0] for p in true_xy_list], dtype=np.float64)
    ty = np.array([p[1] for p in true_xy_list], dtype=np.float64)
    err = np.array(err_list, dtype=np.float64)
    fig_sc, ax_sc = plt.subplots(figsize=(10, 9), dpi=200)
    vmax_e = float(np.percentile(err, 98)) if err.size else 1.0
    vmax_e = max(vmax_e, 0.001)
    sc = ax_sc.scatter(tx, ty, c=err, cmap='plasma', s=36, alpha=0.82, edgecolors='white', linewidths=0.35, norm=Normalize(vmin=0, vmax=vmax_e))
    cb = fig_sc.colorbar(sc, ax=ax_sc, fraction=0.046, pad=0.03)
    cb.set_label('Localization error (m)', rotation=270, labelpad=16)
    ax_sc.set_aspect('equal', adjustable='datalim')
    ax_sc.set_xlabel('x (m)', fontsize=11)
    ax_sc.set_ylabel('y (m)', fontsize=11)
    ax_sc.set_title('Validation: true position colored by fused position error', fontsize=12, pad=10)
    ax_sc.grid(True, alpha=0.35)
    fig_sc.tight_layout()
    p_val_sc = _picture_output_path(Config.OUTPUT_VAL_ERROR_SCATTER_PNG)
    fig_sc.savefig(p_val_sc, bbox_inches='tight', facecolor='white')
    plt.close(fig_sc)
    plt.style.use('default')
    print(f'[saved] {p_val_cm}')
    print(f'[saved] {p_val_pc}')
    print(f'[saved] {p_val_sc}')

def _build_folder_results_train_test(rp_names_ordered: Sequence[str], errors: Sequence[float]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for rp, err in zip(rp_names_ordered, errors):
        rp_s = str(rp)
        if rp_s not in results:
            results[rp_s] = {'errors': []}
        results[rp_s]['errors'].append(float(err))
    for _rp_s, fd in results.items():
        errs = fd['errors']
        fd['average_error'] = float(np.mean(errs)) if errs else 0.0
        fd['std_deviation'] = float(np.std(errs)) if len(errs) > 1 else 0.0
        fd['num_samples'] = len(errs)
    return results

def _plot_error_cdf_by_path_depth(all_errors: Sequence[float], depths_ordered: Sequence[int], out_path: str, cap_m: float=0) -> None:
    if not all_errors or len(all_errors) != len(depths_ordered):
        return
    errs = np.asarray(all_errors, dtype=np.float64)
    depths = np.asarray(depths_ordered, dtype=np.int32)
    if cap_m and cap_m > 0:
        m = errs <= cap_m
        errs = errs[m]
        depths = depths[m]
    if errs.size == 0:
        return
    depth_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    xmax = 0.0
    for depth in Config.PATH_DEPTHS:
        d = int(depth)
        mask = depths == d
        sub = errs[mask]
        if sub.size == 0:
            continue
        s = np.sort(sub)
        cdf_y = np.arange(1, len(s) + 1, dtype=np.float64) / len(s)
        ax.plot(s, cdf_y, label=f'{_path_depth_label(d)} (n={len(s)})', color=depth_colors.get(d, None), linewidth=1.8)
        xmax = max(xmax, float(np.percentile(s, 99)))
    ax.set_xlabel('Localization error (m)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF by max path id (DRL sparse phase×path grid)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    if xmax > 0:
        ax.set_xlim(0, min(15.0, xmax * 1.05))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def _plot_test_cdf_train_test_style(all_errors: Sequence[float], out_path: str) -> None:
    if not all_errors:
        return
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('ggplot')
        except OSError:
            pass
    fig = plt.figure(figsize=(10, 6), dpi=300)
    arr = np.sort(np.asarray(list(all_errors), dtype=np.float64))
    cdf = np.arange(1, len(arr) + 1) / len(arr)
    plt.plot(arr, cdf, color='#1f77b4', linewidth=2)
    plt.xlabel('Localization Error (meters)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Localization Error (train_test-style Top-K logit weights)')
    plt.grid(True)
    plt.xlim(0, 12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.style.use('default')

def _plot_test_bar_train_test_style(results: Dict[str, Dict[str, Any]], out_path: str) -> None:
    if not results:
        return
    try:
        plt.style.use('ggplot')
    except OSError:
        pass
    categories = sorted(results.keys())
    averages = [float(results[k]['average_error']) for k in categories]
    stds = [float(results[k]['std_deviation']) for k in categories]
    fig = plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(categories, averages, yerr=stds, capsize=5, color='#3498db', edgecolor='black', alpha=0.8)
    for bar, avg, std in zip(bars, averages, stds):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f'{avg:.2f}m ± {std:.2f}m', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Test points (true RP folder)', fontsize=12)
    plt.ylabel('Mean Error ± STD (meters)', fontsize=12)
    plt.title('Localization Error per Test Point (train_test style)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.style.use('default')

def evaluate_localization(mean: np.ndarray, std: np.ndarray):
    apply_eval_localization_config()
    print('\n[eval] test localization (DRL sparse grid)...')
    use_amp = bool(getattr(Config, 'USE_AMP', True)) and torch.cuda.is_available()
    model = MixedModel().to(Config.DEVICE)
    model.load_state_dict(_load_model_state(Config.OUTPUT_MODEL_PTH, Config.DEVICE))
    use_dp_eval = getattr(Config, 'USE_DATA_PARALLEL', False) and (not bool(getattr(Config, 'TEST_EVAL_SINGLE_GPU', False))) and torch.cuda.is_available() and (torch.cuda.device_count() > 1)
    if use_dp_eval:
        model = nn.DataParallel(model)
    elif getattr(Config, 'USE_DATA_PARALLEL', False) and torch.cuda.is_available() and (torch.cuda.device_count() > 1) and bool(getattr(Config, 'TEST_EVAL_SINGLE_GPU', False)):
        print('[eval] TEST_EVAL_SINGLE_GPU=True: DataParallel off for eval')
    model.eval()
    coord = pd.read_csv(Config.ORDINATE_CSV)
    official_rp_set = set(Config.RP_LIST)
    valid_mask = _valid_class_mask_tensor(len(Config.RP_LIST), Config.DEVICE)
    vc_json = Config.VALID_RP_CLASSES_JSON
    if os.path.isfile(vc_json):
        print(f'[eval] valid-class mask from {vc_json}')
    else:
        print(f'[eval] not found {vc_json}; using all {len(Config.RP_LIST)}  classes for mask / Top-K weighting')
    test_prefix = _cache_prefix(Config.TEST_PATH)
    lens_f, depth_f, rss_pad_f, rp_f, xy_f = _test_drl_cache_filenames(test_prefix)
    all_errors: List[float] = []
    known_errors: List[float] = []
    unknown_errors: List[float] = []
    rp_names_ordered: List[str] = []
    depths_ordered: List[int] = []
    cls_correct_known = 0
    cls_total_known = 0
    has_pad_cache = os.path.exists(lens_f) and os.path.exists(rss_pad_f) and os.path.exists(rp_f) and os.path.exists(xy_f)
    if has_pad_cache:
        lens = np.load(lens_f)
        true_rp_names = np.load(rp_f, allow_pickle=True)
        true_xy = np.load(xy_f)
        depths_arr = _resolve_test_depths(lens, depth_f if os.path.isfile(depth_f) else None)
        print(f'\n[cache] mmap DRL {rss_pad_f}')
        sequences_padded = np.load(rss_pad_f, mmap_mode='r')
        if sequences_padded.ndim != 3:
            raise RuntimeError(f'Test cache {rss_pad_f} shape should be (N,6,L), got {sequences_padded.shape}; delete *_rp52_{Config.CACHE_DRL_TAG}_*  then rescan Test_Data.')
        all_errors, known_errors, unknown_errors, cls_correct_known, cls_total_known, rp_names_ordered, depths_ordered = _evaluate_localization_batched(model, coord, lens, sequences_padded, true_rp_names, true_xy, mean, std, use_amp, official_rp_set, valid_mask, depths=depths_arr)
    else:
        nw_scan = int(getattr(Config, 'TEST_SCAN_READ_WORKERS', 0))
        print(f'\n[scan] sparse-labeled Test_Data under {os.path.abspath(Config.TEST_PATH)} (pathN_phaseM -> 6xL grid; no cache; TEST_SCAN_READ_WORKERS={nw_scan}）')
        test_root = Path(Config.TEST_PATH)
        n_ok = build_test_drl_cache_from_scan(test_root, coord, lens_f, depth_f, rss_pad_f, rp_f, xy_f)
        if n_ok == 0:
            print('[warn] No valid sparse-labeled samples under Test_Data')
        else:
            print(f'[cache] saved test DRL cache ({n_ok} samples, (N,6,L) mmap); starting batched GPU eval...')
            lens = np.load(lens_f)
            depths_arr = np.load(depth_f)
            sequences_padded = np.load(rss_pad_f, mmap_mode='r')
            true_rp_names = np.load(rp_f, allow_pickle=True)
            true_xy = np.load(xy_f)
            all_errors, known_errors, unknown_errors, cls_correct_known, cls_total_known, rp_names_ordered, depths_ordered = _evaluate_localization_batched(model, coord, lens, sequences_padded, true_rp_names, true_xy, mean, std, use_amp, official_rp_set, valid_mask, depths=depths_arr)
    folder_results: Dict[str, Dict[str, Any]] = {}
    if all_errors and rp_names_ordered and (len(rp_names_ordered) == len(all_errors)):
        folder_results = _build_folder_results_train_test(rp_names_ordered, all_errors)
        for folder in sorted(folder_results.keys()):
            fd = folder_results[folder]
            print(f'\n=== test point [{folder}] ===')
            print(f'test sample number: {int(fd['num_samples'])}')
            print(f'average error: {float(fd['average_error']):.2f}m ± {float(fd['std_deviation']):.2f}m')
        all_glob: List[float] = []
        for fd in folder_results.values():
            all_glob.extend([float(fd['average_error'])] * int(fd['num_samples']))
        print('\n=== global stats ===')
        if all_glob:
            n_tot = sum((int(v['num_samples']) for v in folder_results.values()))
            print(f'total test samples: {n_tot}')
            print(f'mean error: {np.mean(all_glob):.2f}m ± {np.std(all_glob):.2f}m')
    elif all_errors:
        print('[warn] RP name list length != errors; skip per-point summary')
    print('\n======== results ========')
    if cls_total_known > 0:
        cls_acc = 100.0 * cls_correct_known / cls_total_known
        print(f'Test RP Top-1 (true RP in train {len(Config.RP_LIST)} classes): {cls_acc:.2f}% ({cls_correct_known}/{cls_total_known})')
    else:
        print('Test RP Top-1: n/a')
    print(f'total={len(all_errors)} known={len(known_errors)} unknown={len(unknown_errors)}')
    cap = getattr(Config, 'EVAL_ERROR_OUTLIER_MAX_M', 0) or 0
    filt_all = filt_known = filt_unknown = np.array([])
    if all_errors:
        raw = np.asarray(all_errors, dtype=np.float64)
        print(f'mean={np.mean(raw):.3f}m p90={np.percentile(raw, 90):.3f}m')
        filt_all = _errors_within_cap(all_errors, cap)
        filt_known = _errors_within_cap(known_errors, cap)
        filt_unknown = _errors_within_cap(unknown_errors, cap)
        if cap > 0:
            print(f'outlier cap={cap:.0f}m dropped={len(raw) - len(filt_all)}')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(8, 5), dpi=300)
    suffix = f' (<={cap:.0f}m)' if cap > 0 else ''
    for err_list, label, c, ls in ((filt_all, f'all{suffix}', '#1f77b4', '-'), (filt_known, f'known{suffix}', '#ff7f0e', '--'), (filt_unknown, f'unknown{suffix}', '#2ca02c', '-.')):
        if len(err_list) > 0:
            s = np.sort(np.asarray(err_list, dtype=np.float64))
            plt.plot(s, np.arange(1, len(s) + 1) / len(s), label=label, color=c, linestyle=ls, linewidth=1.5)
    plt.xlabel('error (m)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(alpha=0.3)
    if len(filt_all) > 0:
        hi = float(np.percentile(filt_all, 99))
        plt.xlim(0, min(15.0, hi if hi > 0 else 15.0))
    plt.tight_layout()
    p_cdf_global = _picture_output_path(Config.OUTPUT_ERROR_CDF_PNG)
    plt.savefig(p_cdf_global, bbox_inches='tight')
    plt.close()
    print(f'\n[saved] {p_cdf_global}')
    if depths_ordered and len(depths_ordered) == len(all_errors):
        print('\n=== by max path id ===')
        cap = getattr(Config, 'EVAL_ERROR_OUTLIER_MAX_M', 0) or 0
        errs_arr = np.asarray(all_errors, dtype=np.float64)
        dep_arr = np.asarray(depths_ordered, dtype=np.int32)
        for d in Config.PATH_DEPTHS:
            m = dep_arr == int(d)
            sub = errs_arr[m]
            if sub.size == 0:
                continue
            print(f'  {_path_depth_label(d)}: n={sub.size} mean={np.mean(sub):.3f}m p90={np.percentile(sub, 90):.3f}m')
        p_cdf_depth = _picture_output_path(Config.OUTPUT_ERROR_CDF_BY_PATH_DEPTH_PNG)
        _plot_error_cdf_by_path_depth(all_errors, depths_ordered, p_cdf_depth, cap_m=cap)
        print(f'[saved] {p_cdf_depth}')
    k_tag = int(getattr(Config, 'K_VALUE', 3))
    cdf_tt = _picture_output_path(Config.OUTPUT_ERROR_CDF_TRAIN_TEST_PNG.format(k=k_tag))
    bar_tt = _picture_output_path(Config.OUTPUT_TEST_BAR_PNG)
    if all_errors:
        _plot_test_cdf_train_test_style(all_errors, cdf_tt)
        print(f'[saved] {cdf_tt}')
    if folder_results:
        _plot_test_bar_train_test_style(folder_results, bar_tt)
        print(f'[saved] {bar_tt}')
    if all_errors:
        try:
            from cdf_export import save_cdf_bundle
            _cdf_meta = {'script': os.path.basename(__file__), 'eval_scope': 'Test_Data full'}
            if hasattr(Config, 'K_VALUE'):
                _cdf_meta['k_value'] = int(Config.K_VALUE)
            if hasattr(Config, 'EVAL_FUSE_XY_WEIGHT'):
                _cdf_meta['eval_fuse_xy_weight'] = float(Config.EVAL_FUSE_XY_WEIGHT)
            save_cdf_bundle('cnn_v2', all_errors, depths=depths_ordered if depths_ordered else None, meta=_cdf_meta)
        except Exception as ex:
            print(f'[warn] CDF export failed: {ex}')
