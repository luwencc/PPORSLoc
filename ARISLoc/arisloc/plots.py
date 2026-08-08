#!/usr/bin/env python3
"""Training curves, phase/stop statistics, and error CDF plots."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from .config import (CDF_XLIM_MAX_M, COMPOSITE_LAMBDA, CURVE_COMBINED_PNG, CURVE_ERROR_CDF_PNG, CURVE_MEAN_ERR_PNG, CURVE_PATH_LEN_PNG, CURVE_PHASE_DIST_PNG, CURVE_PLOT_MA_WINDOW, CURVE_REWARD_PNG, DRL_EVAL_DETERMINISTIC, EVAL_ACTION_SEED, NUM_PATH_SEGMENTS, NUM_PHASES, PHASE_EPISODE_FRAC_CSV, PHASE_STATS_CSV, SAVE_SEPARATE_CURVE_PLOTS, STOP_DEPTH_STATS_CSV, PPOConfig)
from .localizer import LocalizerInferEngine
from .policy import ActorCritic, policy_greedy_action, policy_sample_action, _seed_eval_action_rng
from .traj import _read_rssi_csv, drl_action_allow_mask, iter_path1_traj_keys, path1_key_for_phase
from .caches import TrajCsvLRUCache, TrajFolderCatalog
from .env import PathPhaseMultiRPEnv

def _setup_matplotlib_zh() -> None:
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

class PhaseActionStats:

    def __init__(self, num_phases: int, depth_keys: Tuple[int, ...]=(2, 3, 4)):
        self.num_phases = int(num_phases)
        self.depth_keys = tuple((int(d) for d in depth_keys))
        self.total = np.zeros(self.num_phases, dtype=np.int64)
        self.by_depth: Dict[int, np.ndarray] = {d: np.zeros(self.num_phases, dtype=np.int64) for d in self.depth_keys}
        self._episode = np.zeros(self.num_phases, dtype=np.int64)

    def begin_episode(self) -> None:
        self._episode.fill(0)

    def record(self, phase_id: int, path_depth: int) -> None:
        p = int(phase_id) % self.num_phases
        self.total[p] += 1
        self._episode[p] += 1
        d = int(path_depth)
        if d in self.by_depth:
            self.by_depth[d][p] += 1

    def episode_fractions(self) -> np.ndarray:
        s = float(self._episode.sum())
        if s <= 0:
            return np.zeros(self.num_phases, dtype=np.float64)
        return self._episode.astype(np.float64) / s

    def fractions(self, counts: Optional[np.ndarray]=None) -> np.ndarray:
        c = self.total if counts is None else counts
        s = float(np.sum(c))
        if s <= 0:
            return np.zeros(self.num_phases, dtype=np.float64)
        return np.asarray(c, dtype=np.float64) / s

    def depth_fractions(self, depth: int) -> np.ndarray:
        if depth not in self.by_depth:
            return np.zeros(self.num_phases, dtype=np.float64)
        return self.fractions(self.by_depth[depth])

    def total_actions(self) -> int:
        return int(self.total.sum())

def _phase_frac_csv_header(num_phases: int) -> List[str]:
    return [f'phase{p}_frac' for p in range(num_phases)]

def save_phase_stats_csv(out_path: Path, stats: PhaseActionStats, *, label: str='train') -> None:
    rows: List[List[Any]] = []
    header = ['scope', 'label', 'phase', 'count', 'fraction', 'total_actions']
    for p in range(stats.num_phases):
        c = int(stats.total[p])
        rows.append(['global', label, p, c, float(stats.fractions()[p]), stats.total_actions()])
    for d in stats.depth_keys:
        sub = stats.by_depth[d]
        tot = int(sub.sum())
        fr = stats.depth_fractions(d)
        for p in range(stats.num_phases):
            rows.append([f'path{d}', label, p, int(sub[p]), float(fr[p]), tot])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_phase_distribution_plot(out_path: Path, stats: PhaseActionStats, *, title: str='PPO DRL phase selection (training)') -> bool:
    if stats.total_actions() <= 0:
        return False
    import matplotlib.pyplot as plt
    _setup_matplotlib_zh()
    n_ph = stats.num_phases
    phases = np.arange(n_ph, dtype=np.int64)
    labels = [f'phase{p}' for p in phases]
    panels: List[Tuple[str, np.ndarray]] = [('all steps', stats.fractions())]
    for d in stats.depth_keys:
        if int(stats.by_depth[d].sum()) > 0:
            panels.append((f'path{d}', stats.depth_fractions(d)))
    n_pan = len(panels)
    fig, axes = plt.subplots(1, n_pan, figsize=(3.2 * n_pan, 4.0), sharey=True)
    if n_pan == 1:
        axes = [axes]
    bar_c = '#5c6bc0'
    for ax, (subtitle, fr) in zip(axes, panels):
        ax.bar(phases, fr * 100.0, color=bar_c, edgecolor='black', alpha=0.88, width=0.72)
        ax.set_xticks(phases)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel('Phase')
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(0.0, max(100.0, float(np.max(fr)) * 100.0 * 1.15))
        ax.grid(True, axis='y', alpha=0.25)
    axes[0].set_ylabel('Selection ratio (%)')
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True

def _format_phase_frac_line(stats: PhaseActionStats, *, prefix: str='phase') -> str:
    fr = stats.fractions()
    parts = [f'{prefix}{p}={fr[p]:.1%}' for p in range(stats.num_phases)]
    return '  '.join(parts)

class StopDepthStats:
    DEPTHS: Tuple[int, ...] = (1, 2, 3, 4)

    def __init__(self) -> None:
        self.total = np.zeros(len(self.DEPTHS), dtype=np.int64)
        self._episode = np.zeros(len(self.DEPTHS), dtype=np.int64)

    def begin_episode(self) -> None:
        self._episode.fill(0)

    def record(self, path_depth: int) -> None:
        d = int(path_depth)
        if d not in self.DEPTHS:
            return
        i = self.DEPTHS.index(d)
        self.total[i] += 1
        self._episode[i] += 1

    def _fractions_from(self, counts: np.ndarray) -> np.ndarray:
        s = float(np.sum(counts))
        if s <= 0:
            return np.zeros(len(self.DEPTHS), dtype=np.float64)
        return counts.astype(np.float64) / s

    def fractions(self) -> np.ndarray:
        return self._fractions_from(self.total)

    def episode_fractions(self) -> np.ndarray:
        return self._fractions_from(self._episode)

    def total_stops(self) -> int:
        return int(self.total.sum())

def composite_score_j(mean_err_m: float, mean_path_len_m: float, lam: float=COMPOSITE_LAMBDA) -> float:
    if not (np.isfinite(mean_err_m) and np.isfinite(mean_path_len_m)):
        return float('nan')
    return float(mean_err_m) + float(lam) * float(mean_path_len_m)

def save_stop_depth_stats_csv(out_path: Path, stats: StopDepthStats, *, label: str='train') -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fr = stats.fractions()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'label', 'path_depth', 'count', 'fraction', 'total_stops'])
        tot = stats.total_stops()
        for i, d in enumerate(StopDepthStats.DEPTHS):
            w.writerow(['sub_episode_end', label, d, int(stats.total[i]), float(fr[i]), tot])

def _localize_error_m(engine: LocalizerInferEngine, rss: np.ndarray, rp: str, traj_key: str) -> float:
    return engine.infer_one(rss, rp, traj_key)

def _cached_or_localize_error_m(env: PathPhaseMultiRPEnv, engine: LocalizerInferEngine, rss: np.ndarray, rp: str, traj_key: str) -> float:
    if PER_STEP_LOCALIZE:
        lr = np.asarray(rss, dtype=np.float32).ravel()
        er = np.asarray(env._last_rss, dtype=np.float32).ravel()
        if lr.size > 0 and lr.size == er.size and np.array_equal(lr, er) and np.isfinite(float(env._last_err)):
            return float(env._last_err)
    return _localize_error_m(engine, rss, rp, traj_key)

def fill_path1_loc_cache(env: PathPhaseMultiRPEnv, engine: LocalizerInferEngine, cfg: PPOConfig) -> None:
    if not PRECOMPUTE_PATH1_LOCS or not PER_STEP_LOCALIZE:
        env.set_path1_err_cache(None)
        return
    if not cfg.enumerate_path1_train or cfg.synthetic_rss:
        env.set_path1_err_cache(None)
        return
    tasks = env._path1_tasks
    if not tasks:
        env.set_path1_err_cache(None)
        return
    rss_list = [env._read_task_rss(t.csv_path) for t in tasks]
    rp_list = [t.rp for t in tasks]
    key_list = [t.path1_key for t in tasks]
    env.set_path1_err_cache(engine.infer_errors(rss_list, rp_list, key_list))

def _obs_after_step_localization(env: PathPhaseMultiRPEnv, engine: LocalizerInferEngine, rss: np.ndarray, rp: str, traj_key: str) -> Tuple[float, np.ndarray]:
    err = _localize_error_m(engine, rss, rp, traj_key)
    return (err, env.obs_after_loc(err))

def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = int(window)
    if w < 2 or y.size < w:
        return (np.array([]), np.array([]), np.array([]))
    s = pd.Series(np.asarray(y, dtype=np.float64))
    m = s.rolling(w, min_periods=w).mean()
    st = s.rolling(w, min_periods=w).std(ddof=0).fillna(0.0)
    start = w - 1
    x = np.arange(start + 1, len(y) + 1, dtype=np.float64)
    mv = m.iloc[start:].to_numpy(dtype=np.float64)
    sv = st.iloc[start:].to_numpy(dtype=np.float64)
    return (x, mv, sv)

def save_training_curve_plots(episode_mean_rewards: List[float], episode_mean_err_m: List[float], episode_mean_path_len_m: List[float], ma_window: int, combined_path: Path, separate_paths: Optional[Tuple[Path, Path, Path]]) -> None:
    import matplotlib.pyplot as plt
    _setup_matplotlib_zh()
    er = np.asarray(episode_mean_rewards, dtype=np.float64)
    em = np.asarray(episode_mean_err_m, dtype=np.float64)
    pl = np.asarray(episode_mean_path_len_m, dtype=np.float64)
    x_all = np.arange(1, len(er) + 1, dtype=np.float64)
    wnd = max(2, int(ma_window))
    raw_c = '#e57373'
    ma_c = '#b71c1c'

    def _band(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _rolling_mean_std(y, wnd)

    def _one_axis(ax, y: np.ndarray, ylabel: str, *, panel: Optional[str]=None, title: Optional[str]=None) -> None:
        ax.plot(x_all, y, color=raw_c, alpha=0.75, linewidth=0.9)
        xb, mb, sb = _band(y)
        if xb.size > 0:
            ax.fill_between(xb, mb - sb, mb + sb, color=raw_c, alpha=0.22, linewidth=0)
            ax.plot(xb, mb, color=ma_c, linestyle='--', linewidth=1.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        elif panel:
            ax.set_title(f'({panel})')
        ax.grid(True, alpha=0.25)

    def _ensure_max_path_len_ytick(ax) -> None:
        max_m = float(MAX_PATH_LEN_M)
        ax.relim()
        ax.autoscale(axis='y')
        y_lo, y_hi = ax.get_ylim()
        if y_hi < max_m:
            ax.set_ylim(y_lo, max_m)
            y_hi = max_m
        ticks = [float(t) for t in ax.get_yticks()]
        if not any((abs(t - max_m) < 0.001 for t in ticks)):
            ticks.append(max_m)
        ticks = sorted((t for t in ticks if y_lo - 1e-06 <= t <= y_hi + 1e-06))
        ax.set_yticks(ticks)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    _one_axis(axes[0], er, 'Average reward', panel='a')
    _one_axis(axes[1], em, 'Mean loc. error (m)', panel='b')
    _one_axis(axes[2], pl, 'Mean path length (m)', panel='c')
    _ensure_max_path_len_ytick(axes[2])
    fig.suptitle('PPO DRL training curves', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if separate_paths is not None:
        p_r, p_e, p_p = separate_paths

        def _solo(path: Path, y: np.ndarray, ylabel: str, title: str) -> None:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            _one_axis(ax2, y, ylabel, title=title)
            fig2.tight_layout()
            fig2.savefig(path, dpi=120)
            plt.close(fig2)
        _solo(p_r, er, 'Average reward', 'Average reward per step')
        _solo(p_e, em, 'Mean loc. error (m)', 'Mean localization error')
        fig_p, ax_p = plt.subplots(figsize=(8, 4))
        _one_axis(ax_p, pl, 'Mean path length (m)', title='Mean path length')
        _ensure_max_path_len_ytick(ax_p)
        fig_p.tight_layout()
        fig_p.savefig(p_p, dpi=120)
        plt.close(fig_p)

def save_error_cdf_plot(errors: List[float], out_path: Path, *, xlim_max_m: float=CDF_XLIM_MAX_M, label: str='PPO DRL (path1 csv × trained policy)') -> bool:
    import matplotlib.pyplot as plt
    arr = np.sort(np.asarray(errors, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    _setup_matplotlib_zh()
    cdf = np.arange(1, arr.size + 1, dtype=np.float64) / float(arr.size)
    x_max = float(arr.max()) * 1.02 if float(xlim_max_m) <= 0 else float(xlim_max_m)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(arr, cdf, color='#1f77b4', linewidth=2, label=label)
    ax.set_xlabel('Localization error (m)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f'PPO DRL localization error CDF — path1 csv eval (n={arr.size})')
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True

@torch.no_grad()
def evaluate_path1_csv_cdf(cfg: PPOConfig, policy: ActorCritic, env: PathPhaseMultiRPEnv, engine: LocalizerInferEngine, catalog: TrajFolderCatalog, device: torch.device, phase_stats: Optional[PhaseActionStats]=None) -> List[float]:
    policy.eval()
    errors: List[float] = []
    use_greedy = bool(DRL_EVAL_DETERMINISTIC)
    if not use_greedy:
        _seed_eval_action_rng(device, EVAL_ACTION_SEED)
    rp_items = [(rp, iter_path1_traj_keys(rp, env.traj_index)) for rp in env.all_rps]
    total_csv = sum((len(catalog.list_csv_paths(rp, k)) for rp, keys in rp_items for k in keys))
    bar = None
    if SHOW_TRAIN_PROGRESS and _tqdm_bar is not None and (total_csv > 0):
        bar = _tqdm_bar(total=total_csv, desc='CDF eval (path1 csv)', unit='csv', ascii=True)
    for rp, path1_keys in rp_items:
        for path1_key in path1_keys:
            for csv_path in catalog.list_csv_paths(rp, path1_key):
                try:
                    rss0 = _read_rssi_csv(csv_path).ravel().astype(np.float32, copy=False)
                except Exception:
                    if bar is not None:
                        bar.update(1)
                    continue
                obs = env.reset_from_path1_csv(rp, path1_key, rss0, eval_mode=True, csv_path=Path(csv_path))
                if PER_STEP_LOCALIZE:
                    _, obs = _obs_after_step_localization(env, engine, rss0, rp, path1_key)
                final_rss = rss0
                final_rp = rp
                rp_sub_done = False
                while not rp_sub_done:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    ev_mask = drl_action_allow_mask(cfg, obs, device)
                    if use_greedy:
                        action_id = policy_greedy_action(policy, obs_t, ev_mask)
                    else:
                        action_t, _, _ = policy_sample_action(policy, obs_t, ev_mask)
                        action_id = int(action_t.item())
                    obs, _, _term, _trunc, info = env.step(action_id)
                    if phase_stats is not None and (not bool(info.get('is_stop'))):
                        phase_stats.record(int(info.get('phase_id', action_id)), int(info.get('path_depth', 0)))
                    final_rss = np.asarray(info.get('rss'), dtype=np.float32).ravel()
                    if PER_STEP_LOCALIZE and (not bool(info.get('rp_sub_done'))):
                        _, obs = _obs_after_step_localization(env, engine, final_rss, str(info.get('rp', rp)), str(info.get('traj_key', '')))
                    final_rp = str(info.get('rp', rp))
                    rp_sub_done = bool(info.get('rp_sub_done'))
                traj_key = str(env._parent_key or path1_key)
                err = _cached_or_localize_error_m(env, engine, final_rss, final_rp, traj_key)
                errors.append(float(err))
                env._eval_hold_rp = False
                if bar is not None:
                    bar.update(1)
    if bar is not None:
        bar.close()
    return errors
