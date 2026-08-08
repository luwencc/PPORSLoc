#!/usr/bin/env python3
"""Trajectory folder scanning and path-phase key helpers."""

from __future__ import annotations
import math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from .config import NUM_PHASES, OBS_META_DIM, PPOConfig
import sys
from pathlib import Path as _Path
_RISFLOC_ROOT = _Path(__file__).resolve().parents[2] / "RIS-FLoc"
if str(_RISFLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISFLOC_ROOT))
import RISFLoc as loc  # noqa: E402

def _read_rssi_csv(fp: Path) -> np.ndarray:
    return loc.load_csv_rssi_column(str(fp))

def _list_traj_folder_csvs(folder: Path) -> Tuple[Path, ...]:
    return tuple(sorted((p for p in folder.glob('*.csv') if p.is_file())))

def parse_traj_folder_name(dirname: str, rp: str) -> Optional[List[Tuple[int, int]]]:
    prefix = rp + '_'
    if not dirname.startswith(prefix):
        return None
    rest = dirname[len(prefix):]
    if not rest:
        return None
    toks = rest.split('_')
    out: List[Tuple[int, int]] = []
    i = 0
    while i + 1 < len(toks):
        if not toks[i].startswith('path'):
            return None
        m1 = re.match('path(\\d+)$', toks[i])
        if not m1:
            return None
        pid = int(m1.group(1))
        if not toks[i + 1].startswith('phase'):
            return None
        m2 = re.match('phase(\\d+)$', toks[i + 1])
        if not m2:
            return None
        ph = int(m2.group(1))
        out.append((pid, ph))
        i += 2
    return out if out else None

def path1_key_for_phase(rp: str, phase: int) -> str:
    return f'{rp}_path1_phase{int(phase)}'

def iter_path1_traj_keys(rp: str, traj_index: Dict[str, Set[str]]) -> List[str]:
    out: List[str] = []
    for key in sorted(traj_index.get(rp, set())):
        segs = parse_traj_folder_name(key, rp)
        if segs and len(segs) == 1 and (segs[0][0] == 1):
            out.append(key)
    return out

def extend_traj_key(rp: str, parent_key: Optional[str], phase: int) -> str:
    if parent_key is None:
        return f'{rp}_path1_phase{int(phase)}'
    segs = parse_traj_folder_name(parent_key, rp)
    if not segs:
        return f'{rp}_path1_phase{int(phase)}'
    last_path = segs[-1][0]
    nxt = last_path + 1
    return f'{parent_key}_path{nxt}_phase{int(phase)}'

def scan_traj_folders(cfg: PPOConfig) -> Dict[str, Set[str]]:
    rp_list = list(loc.Config.RP_LIST)
    out: Dict[str, Set[str]] = {rp: set() for rp in rp_list}
    root = cfg.drl_data_root
    if not root.is_dir():
        return out
    for rp in rp_list:
        d = root / rp
        if not d.is_dir():
            continue
        for ch in d.iterdir():
            if not ch.is_dir():
                continue
            name = ch.name
            if parse_traj_folder_name(name, rp) is not None:
                out[rp].add(name)
    return out

def synthetic_rss(seg_len: int=152) -> np.ndarray:
    t = np.linspace(0, 4 * math.pi, seg_len, dtype=np.float32)
    return (np.sin(t) + 0.05 * np.random.randn(seg_len).astype(np.float32)).astype(np.float32)

@dataclass
class Path1CsvTask:
    rp: str
    path1_key: str
    csv_path: Path
    csv_index: int

def step_reward(cfg: PPOConfig, err_m: float, rp_step_t: int, rp_sub_done: bool) -> float:
    if rp_sub_done:
        d = max(float(cfg.d_norm_m), 1e-06)
        err_norm = min(float(err_m), d)
        return float(-err_norm / d)
    t = max(int(rp_step_t), 1)
    return float(-float(cfg.reward_eta) * t / max(int(cfg.reward_max_steps), 1))

def path_depth_from_obs(obs: np.ndarray, num_path_segments: int) -> int:
    if obs.size < OBS_META_DIM:
        return 1
    depth_norm = float(obs[-OBS_META_DIM])
    return max(1, int(round(depth_norm * max(int(num_path_segments), 1))))

def drl_action_allow_mask(cfg: PPOConfig, obs: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    depth = path_depth_from_obs(obs, cfg.num_path_segments)
    max_d = int(cfg.num_path_segments)
    min_d = max(1, int(cfg.min_path_depth_for_stop))
    block_stop = min_d > 1 and depth < min_d
    block_phases = bool(cfg.require_explicit_stop) and depth >= max_d
    if not block_stop and (not block_phases):
        return None
    mask_np = np.ones(cfg.num_phases + 1, dtype=np.float32)
    if block_stop:
        mask_np[-1] = 0.0
    if block_phases:
        mask_np[:cfg.num_phases] = 0.0
    return torch.from_numpy(mask_np).to(device=device)
