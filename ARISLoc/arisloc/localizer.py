#!/usr/bin/env python3
"""Frozen RISFLoc MixedModel inference used as the localization oracle."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
from pathlib import Path as _Path
_RISFLOC_ROOT = _Path(__file__).resolve().parents[2] / "RIS-FLoc"
if str(_RISFLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISFLOC_ROOT))
import RISFLoc as loc  # noqa: E402
from .caches import LocInferResultCache
from .config import (EVAL_TOPK_PROB_MASS_DIV, EVAL_TOPK_WEIGHT_MODE, LOC_INFER_BATCH_SIZE, LOC_INFER_CACHE_ENABLED, LOC_INFER_CACHE_MAX_ENTRIES, NUM_PHASES, OBS_GRID_ZSCORE, PPOConfig, default_ppo_config)
from .traj import parse_traj_folder_name

def apply_drl_localizer_eval_config(*, topk_weight_mode: Optional[str]=None, prob_mass_div: Optional[float]=None) -> None:
    loc.apply_eval_localization_config(
        topk_weight_mode=str(topk_weight_mode) if topk_weight_mode is not None else str(EVAL_TOPK_WEIGHT_MODE),
        prob_mass_div=float(prob_mass_div) if prob_mass_div is not None else float(EVAL_TOPK_PROB_MASS_DIV),
    )

def load_localizer(cfg: PPOConfig) -> Tuple[nn.Module, np.ndarray, np.ndarray, pd.DataFrame, torch.Tensor, torch.Tensor]:
    apply_drl_localizer_eval_config()
    device = torch.device(cfg.device)
    m = loc.MixedModel().to(device)
    try:
        state = torch.load(cfg.weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(cfg.weights, map_location=device)
    if isinstance(state, dict) and any((k.startswith('module.') for k in state)):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    m.load_state_dict(state)
    m.eval()
    mean = np.load(cfg.mean_npy)
    std = np.load(cfg.std_npy)
    coord = pd.read_csv(cfg.ordinate_csv)
    xy_lut = loc._build_rp_xy_lut_tensor(coord, device)
    valid_mask = loc._valid_class_mask_tensor(len(loc.Config.RP_LIST), device)
    lam = float(loc.Config.EVAL_FUSE_XY_WEIGHT)
    wt_mode = loc._eval_topk_weight_mode()
    mass_div = float(loc._eval_topk_prob_mass_div())
    vc_json = getattr(loc.Config, 'VALID_RP_CLASSES_JSON', 'valid_rp_classes.json')
    print(f'[DRL localizer] aligned with Test_Eval | Top-{int(loc.Config.K_VALUE)} weight_mode={wt_mode} prob_mass_div={mass_div:.2f} fuse_xy lam={lam:.2f} valid_classes={int(valid_mask.sum().item())}/{len(loc.Config.RP_LIST)} ({('mask from ' + vc_json if Path(vc_json).is_file() else 'no mask file; all classes valid')})')
    return (m, mean, std, coord, xy_lut, valid_mask)

def _masked_logits(outputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = outputs.float()
    vm = valid_mask
    if vm.device != masked.device:
        vm = vm.to(device=masked.device)
    return masked.masked_fill(~vm.view(1, -1), float('-inf'))

def _build_rp_xy_lookup(coord: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for rp in coord.columns:
        try:
            out[str(rp)] = (float(coord.loc[0, rp]), float(coord.loc[1, rp]))
        except Exception:
            continue
    return out

class LocalizerInferEngine:

    def __init__(self, bundle: Tuple[Any, ...], cfg: PPOConfig):
        model, mean, std, coord, xy_lut, valid_mask = bundle
        self.cfg = cfg
        self.model = model
        self.mean = mean
        self.std = std
        self.device = torch.device(cfg.device)
        self.xy_lut = xy_lut
        self.valid_mask = valid_mask
        self.rp_xy = _build_rp_xy_lookup(coord)
        self.n_ph = int(loc.Config.NUM_PHASES)
        self.lam = float(loc.Config.EVAL_FUSE_XY_WEIGHT)
        self.use_amp = bool(getattr(loc.Config, 'USE_AMP', True)) and self.device.type == 'cuda'
        self.batch_size = max(1, int(cfg.loc_infer_batch_size))
        self.invalid_err_m = float(cfg.invalid_rp_err_m)
        self._result_cache: Optional[LocInferResultCache] = None
        if LOC_INFER_CACHE_ENABLED and LOC_INFER_CACHE_MAX_ENTRIES > 0:
            self._result_cache = LocInferResultCache(int(LOC_INFER_CACHE_MAX_ENTRIES))
        self._pad_buf: Optional[torch.Tensor] = None
        self._pad_cap_b = 0
        self._pad_cap_l = 0
        self._lens_cap = 0
        self._lens_buf: Optional[torch.Tensor] = None

    def _lens_tensor(self, lens: List[int]) -> torch.Tensor:
        b = len(lens)
        if self._lens_buf is None or self._lens_cap < b:
            cap = max(b, self._lens_cap * 2 if self._lens_cap else b, 64)
            self._lens_buf = torch.empty(cap, dtype=torch.long, device=self.device)
            self._lens_cap = cap
        out = self._lens_buf[:b]
        out.copy_(torch.as_tensor(lens, dtype=torch.long, device=self.device))
        return out

    def _ensure_pad(self, b: int, max_l: int) -> torch.Tensor:
        b = int(b)
        max_l = int(max_l)
        if self._pad_buf is None or self._pad_cap_b < b or self._pad_cap_l < max_l:
            cap_b = max(b, self._pad_cap_b * 2 if self._pad_cap_b else b, 64)
            cap_l = max(max_l, self._pad_cap_l, int(loc.Config.MAX_SEQ_LEN))
            self._pad_buf = torch.zeros(cap_b, self.n_ph, cap_l, dtype=torch.float32, device=self.device)
            self._pad_cap_b = cap_b
            self._pad_cap_l = cap_l
        return self._pad_buf[:b, :, :max_l]

    def infer_one(self, rss: np.ndarray, rp: str, traj_key: str) -> float:
        return float(self.infer_errors([rss], [rp], [traj_key])[0])

    @torch.inference_mode()
    def infer_errors(self, rss_list: List[np.ndarray], rp_names: List[str], traj_keys: List[str], *, invalid_err_m: Optional[float]=None) -> List[float]:
        inv = float(self.invalid_err_m if invalid_err_m is None else invalid_err_m)
        n = len(rss_list)
        errs: List[float] = [inv] * n
        if n == 0:
            return errs
        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            self._infer_chunk(rss_list[start:end], rp_names[start:end], traj_keys[start:end], errs, start, inv)
        return errs

    def _infer_chunk(self, rss_list: List[np.ndarray], rp_names: List[str], traj_keys: List[str], errs: List[float], offset: int, invalid_err_m: float) -> None:
        batch_idx: List[int] = []
        grids: List[np.ndarray] = []
        lens: List[int] = []
        for i, rss in enumerate(rss_list):
            if self._result_cache is not None:
                cached = self._result_cache.get(rp_names[i], traj_keys[i], rss)
                if cached is not None:
                    errs[offset + i] = cached
                    continue
            built = _sparse_grid_from_traj_csv(rss, traj_keys[i], rp_names[i])
            if built is None:
                errs[offset + i] = 1000.0
                continue
            grid, L = built
            z = loc.z_score_apply_phase_grid(grid[:, :L], self.mean, self.std)
            batch_idx.append(i)
            grids.append(z)
            lens.append(int(z.shape[1]))
        if not batch_idx:
            return
        max_l = max(lens)
        B = len(grids)
        padded = self._ensure_pad(B, max_l)
        padded.zero_()
        for bi, z in enumerate(grids):
            L = lens[bi]
            padded[bi, :, :L] = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        lens_t = self._lens_tensor(lens)
        with loc._amp_autocast_ctx(self.use_amp):
            outputs, pred_xy, _ = self.model(padded, lens_t)
        masked = _masked_logits(outputs, self.valid_mask)
        pxy_soft = loc._soft_weighted_xy_from_logits_batch(masked, self.xy_lut)
        pxy_fused = (self.lam * pred_xy.float() + (1.0 - self.lam) * pxy_soft.float()).float()
        for bi, i in enumerate(batch_idx):
            rp = rp_names[i]
            xy = self.rp_xy.get(rp)
            if xy is None:
                errs[offset + i] = invalid_err_m
                continue
            px = float(pxy_fused[bi, 0].item())
            py = float(pxy_fused[bi, 1].item())
            tx, ty = xy
            err_val = float(math.hypot(px - tx, py - ty))
            errs[offset + i] = err_val
            if self._result_cache is not None:
                self._result_cache.put(rp_names[i], traj_keys[i], rss_list[i], err_val)

def _sparse_grid_from_traj_csv(rss: np.ndarray, traj_key: str, rp: str) -> Optional[Tuple[np.ndarray, int]]:
    segs = parse_traj_folder_name(traj_key, rp)
    if not segs:
        segs = loc.parse_path_phase_segments(traj_key, rp)
    if not segs:
        return None
    return loc.build_sparse_phase_path_grid(rss, segs)

@torch.no_grad()
def localization_error_m(model: nn.Module, mean: np.ndarray, std: np.ndarray, coord: pd.DataFrame, xy_lut: torch.Tensor, valid_mask: torch.Tensor, rss: np.ndarray, true_rp: str, traj_key: str, device: torch.device, *, engine: Optional[LocalizerInferEngine]=None) -> float:
    if engine is not None:
        return engine.infer_one(rss, true_rp, traj_key)
    built = _sparse_grid_from_traj_csv(rss, traj_key, true_rp)
    if built is None:
        return 1000.0
    grid, L = built
    z = loc.z_score_apply_phase_grid(grid[:, :L], mean, std)
    n_ph = int(loc.Config.NUM_PHASES)
    x = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    if x.shape[1] != n_ph:
        return 1000.0
    lt = torch.tensor([L], dtype=torch.long)
    lam = float(loc.Config.EVAL_FUSE_XY_WEIGHT)
    with loc._amp_autocast_ctx(bool(getattr(loc.Config, 'USE_AMP', True)) and device.type == 'cuda'):
        outputs, pred_xy, _ = model(x, lt)
    masked = _masked_logits(outputs, valid_mask)
    pxy_soft = loc._soft_weighted_xy_from_logits_batch(masked, xy_lut)[0]
    pxy_fused = (lam * pred_xy[0] + (1.0 - lam) * pxy_soft).float()
    px = float(pxy_fused[0].cpu())
    py = float(pxy_fused[1].cpu())
    tx = float(coord.loc[0, true_rp])
    ty = float(coord.loc[1, true_rp])
    return float(math.hypot(px - tx, py - ty))

@torch.no_grad()
def localization_errors_batched(model: nn.Module, mean: np.ndarray, std: np.ndarray, coord: pd.DataFrame, xy_lut: torch.Tensor, valid_mask: torch.Tensor, rss_list: List[np.ndarray], rp_names: List[str], traj_keys: List[str], device: torch.device, *, invalid_err_m: float=50.0, loc_batch_size: int=LOC_INFER_BATCH_SIZE, engine: Optional[LocalizerInferEngine]=None) -> List[float]:
    if engine is not None:
        return engine.infer_errors(rss_list, rp_names, traj_keys, invalid_err_m=float(invalid_err_m))
    cfg = default_ppo_config()
    cfg.device = str(device)
    cfg.invalid_rp_err_m = float(invalid_err_m)
    cfg.loc_infer_batch_size = max(1, int(loc_batch_size))
    eng = LocalizerInferEngine((model, mean, std, coord, xy_lut, valid_mask), cfg)
    return eng.infer_errors(rss_list, rp_names, traj_keys, invalid_err_m=float(invalid_err_m))
