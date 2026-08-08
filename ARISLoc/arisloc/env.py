#!/usr/bin/env python3
"""Gymnasium environment for multi-RP path-phase PPO training."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise SystemExit('Please install gymnasium: pip install gymnasium') from e
import sys
from pathlib import Path as _Path
_RISFLOC_ROOT = _Path(__file__).resolve().parents[2] / "RIS-FLoc"
if str(_RISFLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISFLOC_ROOT))
import RISFLoc as loc  # noqa: E402
from .caches import TrajCsvLRUCache, TrajFolderCatalog
from .config import (ACTION_STOP_ID, NUM_DRL_ACTIONS, NUM_PATH_SEGMENTS, NUM_PHASES, OBS_GRID_ZSCORE, OBS_META_DIM, OBS_RSS_DIM, PER_STEP_LOCALIZE, PRECOMPUTE_PATH1_LOCS, SUB_EPISODE_ALIGNED_CSV, PPOConfig)
from .localizer import LocalizerInferEngine, _sparse_grid_from_traj_csv, localization_error_m, localization_errors_batched
from .traj import (Path1CsvTask, _read_rssi_csv, drl_action_allow_mask, extend_traj_key, iter_path1_traj_keys, path1_key_for_phase, parse_traj_folder_name, path_depth_from_obs, step_reward, synthetic_rss)

class PathPhaseMultiRPEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg: PPOConfig, traj_index: Dict[str, Set[str]], localizer_bundle: Tuple[Any, ...], catalog: TrajFolderCatalog, traj_cache: Optional[TrajCsvLRUCache]=None):
        super().__init__()
        self.cfg = cfg
        self.traj_index = traj_index
        self.catalog = catalog
        self.traj_cache = traj_cache
        self.model, self.mean, self.std, self.coord, self._xy_lut, self._valid_mask = localizer_bundle
        self.device = torch.device(cfg.device)
        self._loc_mean = np.asarray(self.mean, dtype=np.float32)
        self._loc_std = np.asarray(self.std, dtype=np.float32)
        self.all_rps = list(loc.Config.RP_LIST)
        self.n_rps = len(self.all_rps)
        self.obs_rss_dim = cfg.obs_rss_dim
        self.obs_meta_dim = OBS_META_DIM
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.obs_rss_dim + self.obs_meta_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(int(cfg.num_phases) + 1)
        self._rp_queue: List[str] = []
        self._rp_cursor = 0
        self._rp: str = self.all_rps[0]
        self._parent_key: Optional[str] = None
        self._last_rss: np.ndarray = np.zeros(0, dtype=np.float32)
        self._last_err: float = 0.0
        self._rp_step: int = 0
        self._eval_hold_rp: bool = False
        self._path1_tasks: List[Path1CsvTask] = []
        self._path1_task_i: int = 0
        self._sub_episode_csv_idx: Optional[int] = None
        self._path1_err_cache: Optional[List[float]] = None

    def set_path1_err_cache(self, errs: Optional[List[float]]) -> None:
        self._path1_err_cache = None if errs is None else [float(e) for e in errs]

    def obs_with_cached_path1_err(self) -> np.ndarray:
        if self._path1_err_cache is not None and 0 <= int(self._path1_task_i) < len(self._path1_err_cache):
            self._last_err = float(self._path1_err_cache[int(self._path1_task_i)])
        return self._rss_to_obs(self._last_rss)

    def _load_traj_rss(self, rp: str, traj_key: str) -> np.ndarray:
        if SUB_EPISODE_ALIGNED_CSV and self._sub_episode_csv_idx is not None:
            return self.catalog.rss_at_index(rp, traj_key, int(self._sub_episode_csv_idx), self.traj_cache)
        return self.catalog.sample_rss(rp, traj_key, self.np_random, self.traj_cache)

    def count_path1_tasks_per_episode(self) -> int:
        return int(self.n_rps * self.cfg.num_phases)

    def _read_task_rss(self, csv_path: Path) -> np.ndarray:
        try:
            if self.traj_cache is not None:
                return self.traj_cache.get(csv_path)
            return _read_rssi_csv(csv_path).ravel().astype(np.float32, copy=False)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def _build_path1_task_queue(self) -> None:
        tasks: List[Path1CsvTask] = []
        rng = self.np_random
        for rp in self._rp_queue:
            for phase in range(self.cfg.num_phases):
                path1_key = path1_key_for_phase(rp, phase)
                if path1_key not in self.traj_index.get(rp, set()):
                    continue
                csv_paths = self.catalog.list_csv_paths(rp, path1_key)
                if not csv_paths:
                    continue
                idx = int(rng.integers(0, len(csv_paths)))
                tasks.append(Path1CsvTask(rp=rp, path1_key=path1_key, csv_path=Path(csv_paths[idx]), csv_index=idx))
        self._path1_tasks = tasks
        self._path1_task_i = 0

    def _start_path1_task(self, task: Path1CsvTask) -> np.ndarray:
        rss = self._read_task_rss(task.csv_path)
        return self.reset_from_path1_csv(task.rp, task.path1_key, rss, eval_mode=False, csv_index=int(task.csv_index))

    def _advance_path1_task(self) -> Tuple[bool, np.ndarray]:
        self._path1_task_i += 1
        if self._path1_task_i >= len(self._path1_tasks):
            return (True, self._rss_to_obs(self._last_rss))
        return (False, self._start_path1_task(self._path1_tasks[self._path1_task_i]))

    def _rp_index_norm(self) -> float:
        if self.n_rps <= 1:
            return 0.0
        return float(self._rp_cursor) / float(self.n_rps - 1)

    def _path1_phase_norm(self) -> float:
        if not self._parent_key:
            return 0.0
        segs = parse_traj_folder_name(self._parent_key, self._rp)
        if not segs:
            return 0.0
        ph = int(segs[0][1])
        denom = max(int(self.cfg.num_phases) - 1, 1)
        return float(np.clip(ph, 0, int(self.cfg.num_phases) - 1)) / float(denom)

    def _rss_to_phase_grid_feat(self, rss: np.ndarray) -> np.ndarray:
        n_ph = int(loc.Config.NUM_PHASES)
        lmax = int(self.cfg.max_rss_len)
        grid = np.zeros((n_ph, lmax), dtype=np.float32)
        rss = np.asarray(rss, dtype=np.float32).ravel()
        if rss.size > 0 and self._parent_key:
            built = _sparse_grid_from_traj_csv(rss, self._parent_key, self._rp)
            if built is not None:
                mat, _L = built
                grid[:, :mat.shape[1]] = mat[:, :mat.shape[1]]
                if OBS_GRID_ZSCORE:
                    grid = loc.z_score_apply_phase_grid(grid, self._loc_mean, self._loc_std)
        return grid.ravel()

    def _rss_to_obs(self, rss: np.ndarray) -> np.ndarray:
        feat = self._rss_to_phase_grid_feat(rss)
        depth = 0.0
        if self._parent_key:
            segs = parse_traj_folder_name(self._parent_key, self._rp)
            depth = len(segs) / float(self.cfg.num_path_segments) if segs else 0.0
        rss = np.asarray(rss, dtype=np.float32).ravel()
        meta = np.array([depth, min(self._last_err, self.cfg.d_norm_m) / float(self.cfg.d_norm_m), min(rss.size, self.cfg.max_rss_len) / float(self.cfg.max_rss_len), self._rp_index_norm(), self._path1_phase_norm()], dtype=np.float32)
        return np.concatenate([feat, meta], axis=0)

    def _begin_rp(self, rp: str) -> None:
        self._rp = rp
        self._parent_key = None
        self._last_rss = np.zeros(0, dtype=np.float32)
        self._last_err = 0.0
        self._rp_step = 0
        self._sub_episode_csv_idx = None

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self._rp_queue = list(self.all_rps)
        if self.cfg.shuffle_rp_order:
            self.np_random.shuffle(self._rp_queue)
        self._rp_cursor = 0
        self._eval_hold_rp = False
        if self.cfg.enumerate_path1_train and (not self.cfg.synthetic_rss):
            self._build_path1_task_queue()
            if not self._path1_tasks:
                raise RuntimeError('enumerate_path1_train on but no path1 tasks built')
            return (self._start_path1_task(self._path1_tasks[0]), {})
        self._begin_rp(self._rp_queue[0])
        return (self._rss_to_obs(self._last_rss), {})

    def reset_from_path1_csv(self, rp: str, path1_key: str, rss: np.ndarray, *, eval_mode: bool=False, csv_path: Optional[Path]=None, csv_index: Optional[int]=None) -> np.ndarray:
        self._eval_hold_rp = bool(eval_mode)
        self._rp = rp
        self._parent_key = path1_key
        self._last_rss = np.asarray(rss, dtype=np.float32).ravel()
        self._last_err = 0.0
        self._rp_step = 1
        if csv_index is not None:
            self._sub_episode_csv_idx = int(csv_index)
        elif csv_path is not None and SUB_EPISODE_ALIGNED_CSV:
            self._sub_episode_csv_idx = self.catalog.csv_index_of_path(rp, path1_key, csv_path)
        else:
            self._sub_episode_csv_idx = None
        if rp in self._rp_queue:
            self._rp_cursor = self._rp_queue.index(rp)
        return self._rss_to_obs(self._last_rss)

    def _current_path_depth(self) -> int:
        if not self._parent_key:
            return 0
        segs = parse_traj_folder_name(self._parent_key, self._rp)
        return len(segs) if segs else 0

    def step(self, action):
        act = int(action) if np.isscalar(action) else int(np.asarray(action).ravel()[0])
        num_ph = int(self.cfg.num_phases)
        stop_now = act >= num_ph
        if not stop_now and self.cfg.require_explicit_stop and (self._current_path_depth() >= int(self.cfg.num_path_segments)):
            stop_now = True
        if stop_now:
            new_key = str(self._parent_key or '')
            rss = np.asarray(self._last_rss, dtype=np.float32).ravel()
            segs_after = parse_traj_folder_name(new_key, self._rp) if new_key else None
            depth = len(segs_after) if segs_after else 1 if self._parent_key else 0
            valid = bool(self._parent_key)
            self._rp_step += 1
            path_len_m = float(depth) * float(self.cfg.seg_len_m)
            rp_sub_done = True
        else:
            phase_id = act % num_ph
            new_key = extend_traj_key(self._rp, self._parent_key, phase_id)
            folder = self.cfg.drl_data_root / self._rp / new_key
            valid = new_key in self.traj_index.get(self._rp, set()) and folder.is_dir()
            if valid:
                rss = self._load_traj_rss(self._rp, new_key)
            elif self.cfg.synthetic_rss:
                rss = synthetic_rss()
            else:
                rss = np.zeros(0, dtype=np.float32)
            self._last_rss = rss
            self._rp_step += 1
            segs_after = parse_traj_folder_name(new_key, self._rp) if valid else None
            depth = len(segs_after) if segs_after else 0
            path_len_m = float(depth) * float(self.cfg.seg_len_m)
            rp_sub_done = False
            if not valid and (not self.cfg.synthetic_rss):
                rp_sub_done = True
            else:
                segs = parse_traj_folder_name(new_key, self._rp) if valid else []
                last_path = segs[-1][0] if segs else 0
                if not valid:
                    rp_sub_done = True
                elif last_path >= self.cfg.num_path_segments:
                    self._parent_key = new_key
                    if self.cfg.require_explicit_stop:
                        rp_sub_done = False
                    else:
                        rp_sub_done = True
                else:
                    self._parent_key = new_key
        reward = 0.0
        terminated = False
        truncated = False
        rp_path_len_m = float(path_len_m)
        info = {'loc_err_m': float('nan'), 'rp': self._rp, 'traj_key': new_key, 'folder_ok': valid, 'path_depth': depth, 'path_len_m': path_len_m, 'rp_step_t': int(self._rp_step), 'rp_sub_done': rp_sub_done, 'rp_final_err_m': float('nan'), 'rp_path_len_m': rp_path_len_m if rp_sub_done else float('nan'), 'rss': rss, 'needs_loc': bool(rp_sub_done), 'action_id': int(act), 'is_stop': bool(stop_now), 'phase_id': int(act % num_ph) if not stop_now else -1}
        if rp_sub_done:
            if self._eval_hold_rp:
                pass
            elif self.cfg.enumerate_path1_train and (not self.cfg.synthetic_rss):
                terminated, obs_after = self._advance_path1_task()
                return (obs_after, reward, terminated, truncated, info)
            elif self._rp_cursor + 1 >= self.n_rps:
                terminated = True
            else:
                self._rp_cursor += 1
                self._begin_rp(self._rp_queue[self._rp_cursor])
        obs = self._rss_to_obs(self._last_rss)
        return (obs, reward, terminated, truncated, info)

    def obs_after_loc(self, err_m: float) -> np.ndarray:
        self._last_err = float(err_m)
        return self._rss_to_obs(self._last_rss)
