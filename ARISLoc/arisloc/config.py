#!/usr/bin/env python3
"""PPO hyperparameters and episode schedule helpers."""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import torch
import sys
from pathlib import Path as _Path
_RISFLOC_ROOT = _Path(__file__).resolve().parents[2] / "RIS-FLoc"
if str(_RISFLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISFLOC_ROOT))
import RISFLoc as loc  # noqa: E402

DRL_DATA_ROOT = Path('Test_Data')
ORDINATE_CSV: Optional[str] = 'ordinate.csv'
WEIGHTS_PATH = Path(loc.Config.OUTPUT_MODEL_PTH)
MEAN_NPY_PATH = Path(loc.Config.OUTPUT_MEAN_NPY)
STD_NPY_PATH = Path(loc.Config.OUTPUT_STD_NPY)
POLICY_SAVE_PATH = Path('drl_ppo_path_phase.pt')
DEVICE: Optional[str] = 'cuda:3'
NUM_PHASES = 6
ACTION_STOP_ID = NUM_PHASES
NUM_DRL_ACTIONS = NUM_PHASES + 1
NUM_PATH_SEGMENTS = 4
MAX_RSS_LEN = int(loc.Config.MAX_SEQ_LEN)
OBS_RSS_DIM = int(loc.Config.NUM_PHASES) * int(loc.Config.MAX_SEQ_LEN)
OBS_META_DIM = 5
OBS_GRID_ZSCORE = True
SEG_LEN_M = 4.0
MAX_PATH_LEN_M = float(NUM_PATH_SEGMENTS * SEG_LEN_M)
REWARD_ETA = 0.08
REWARD_MAX_STEPS = NUM_PATH_SEGMENTS
REWARD_D_NORM_M = 40.0
D_NORM_M = REWARD_D_NORM_M
INVALID_RP_ERR_M = REWARD_D_NORM_M
REQUIRE_EXPLICIT_STOP = True
MIN_PATH_DEPTH_FOR_STOP = 1
SHUFFLE_RP_ORDER = True
ENUMERATE_PATH1_TRAIN = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 0.0002
LR_MIN = 2e-05
CONVERGE_TARGET_EP = 1500
LR_DECAY_START_EP = 1000
LR_DECAY_END_EP = 1600
TRAIN_TOTAL_EPISODES = 2500
LOG_EVERY = 50
PPO_ROLLOUT_STEPS = 2048
PPO_EPOCHS = 5
PPO_EPOCHS_LATE = 2
PPO_EPOCHS_LATE_START_EP = 1400
PPO_CLIP = 0.15
PPO_MINIBATCH = 256
ENT_COEF = 0.003
ENT_START = 0.2
ENT_PHASE_COEF = ENT_COEF
ENT_CONT_COEF = ENT_COEF
ENT_PHASE_START = ENT_START
ENT_CONT_START = ENT_START
ENT_ANNEAL_ENABLED = True
ENT_ANNEAL_DECAY = 250
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPSILON_GREEDY_ENABLED = True
EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 250.0
SYNTHETIC_RSS = False
SUB_EPISODE_ALIGNED_CSV = True
TRAJ_CACHE_ENABLED = True
TRAJ_CACHE_MAX_GB = 16.0
LOC_INFER_BATCH_SIZE = 1024
LOC_INFER_CACHE_ENABLED = True
LOC_INFER_CACHE_MAX_ENTRIES = 16384
EPISODE_CSV_FLUSH_EVERY = 20
EPISODE_RETURNS_CSV = Path('drl_ppo_episode_returns.csv')
COMPOSITE_LAMBDA = 0.1
STOP_DEPTH_STATS_CSV = Path('drl_ppo_stop_depth_stats.csv')
PER_STEP_LOCALIZE = True
PRECOMPUTE_PATH1_LOCS = True
REWARD_ROLL_EPISODES = 20
SAVE_TRAIN_CURVE_PLOTS = True
CURVE_PLOT_MA_WINDOW = 40
CURVE_COMBINED_PNG = Path('drl_ppo_train_curves_panel.png')
SAVE_SEPARATE_CURVE_PLOTS = False
CURVE_REWARD_PNG = Path('drl_ppo_curve_mean_reward.png')
CURVE_MEAN_ERR_PNG = Path('drl_ppo_curve_mean_loc_err_m.png')
CURVE_PATH_LEN_PNG = Path('drl_ppo_curve_mean_path_len_m.png')
PHASE_STATS_CSV = Path('drl_ppo_phase_stats.csv')
PHASE_EPISODE_FRAC_CSV = Path('drl_ppo_phase_episode_frac.csv')
CURVE_PHASE_DIST_PNG = Path('drl_ppo_phase_distribution.png')
SAVE_PHASE_STATS = True
SAVE_ERROR_CDF = True
CURVE_ERROR_CDF_PNG = Path('drl_ppo_error_cdf.png')
CDF_XLIM_MAX_M = 10.0
DRL_EVAL_DETERMINISTIC = False
EVAL_ACTION_SEED: Optional[int] = 0
EVAL_TOPK_WEIGHT_MODE = 'full_softmax_no_renorm'
EVAL_TOPK_PROB_MASS_DIV = 1.0
SHOW_TRAIN_PROGRESS = True

@dataclass
class PPOConfig:
    drl_data_root: Path
    ordinate_csv: Path
    weights: Path
    mean_npy: Path
    std_npy: Path
    device: str
    num_phases: int
    num_path_segments: int
    max_rss_len: int
    obs_rss_dim: int
    seg_len_m: float
    max_path_len_m: float
    d_norm_m: float
    reward_eta: float
    reward_max_steps: int
    invalid_rp_err_m: float
    require_explicit_stop: bool
    min_path_depth_for_stop: int
    epsilon_greedy_enabled: bool
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    shuffle_rp_order: bool
    enumerate_path1_train: bool
    gamma: float
    gae_lambda: float
    lr: float
    train_total_episodes: int
    log_every: int
    ppo_rollout_steps: int
    ppo_epochs: int
    ppo_clip: float
    ppo_minibatch: int
    ent_phase_coef: float
    ent_cont_coef: float
    vf_coef: float
    max_grad_norm: float
    synthetic_rss: bool
    policy_save_path: Path
    traj_cache_enabled: bool
    traj_cache_max_gb: float
    loc_infer_batch_size: int

def _resolve_train_device(dev: str) -> str:
    d = str(dev).strip()
    if d.startswith('cuda') and (not torch.cuda.is_available()):
        print('[warn] CUDA unavailable; falling back to cpu', flush=True)
        return 'cpu'
    if d.startswith('cuda:'):
        try:
            idx = int(d.split(':', 1)[1])
        except ValueError:
            return d
        if torch.cuda.is_available() and idx >= torch.cuda.device_count():
            print(f'[warn] requested {d} but only {torch.cuda.device_count()}  GPU(s); falling back to cuda:0', flush=True)
            return 'cuda:0'
    return d

def default_ppo_config() -> PPOConfig:
    raw_dev = DEVICE if DEVICE else 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = _resolve_train_device(raw_dev)
    ord_csv = Path(ORDINATE_CSV) if ORDINATE_CSV else Path(loc.Config.ORDINATE_CSV)
    return PPOConfig(drl_data_root=Path(DRL_DATA_ROOT), ordinate_csv=ord_csv, weights=Path(WEIGHTS_PATH), mean_npy=Path(MEAN_NPY_PATH), std_npy=Path(STD_NPY_PATH), device=dev, num_phases=NUM_PHASES, num_path_segments=NUM_PATH_SEGMENTS, max_rss_len=MAX_RSS_LEN, obs_rss_dim=OBS_RSS_DIM, seg_len_m=float(SEG_LEN_M), max_path_len_m=float(MAX_PATH_LEN_M), d_norm_m=float(D_NORM_M), reward_eta=float(REWARD_ETA), reward_max_steps=int(REWARD_MAX_STEPS), invalid_rp_err_m=float(INVALID_RP_ERR_M), require_explicit_stop=bool(REQUIRE_EXPLICIT_STOP), min_path_depth_for_stop=int(MIN_PATH_DEPTH_FOR_STOP), epsilon_greedy_enabled=bool(EPSILON_GREEDY_ENABLED), epsilon_start=float(EPSILON_START), epsilon_min=float(EPSILON_MIN), epsilon_decay=float(EPSILON_DECAY), shuffle_rp_order=bool(SHUFFLE_RP_ORDER), enumerate_path1_train=bool(ENUMERATE_PATH1_TRAIN), gamma=float(GAMMA), gae_lambda=float(GAE_LAMBDA), lr=float(LR), train_total_episodes=int(TRAIN_TOTAL_EPISODES), log_every=int(LOG_EVERY), ppo_rollout_steps=int(PPO_ROLLOUT_STEPS), ppo_epochs=int(PPO_EPOCHS), ppo_clip=float(PPO_CLIP), ppo_minibatch=int(PPO_MINIBATCH), ent_phase_coef=float(ENT_COEF), ent_cont_coef=float(ENT_COEF), vf_coef=float(VF_COEF), max_grad_norm=float(MAX_GRAD_NORM), synthetic_rss=bool(SYNTHETIC_RSS), policy_save_path=Path(POLICY_SAVE_PATH), traj_cache_enabled=bool(TRAJ_CACHE_ENABLED), traj_cache_max_gb=float(TRAJ_CACHE_MAX_GB), loc_infer_batch_size=max(1, int(LOC_INFER_BATCH_SIZE)))

def entropy_anneal_decay_episodes(total_episodes: int, decay: float=ENT_ANNEAL_DECAY) -> float:
    if float(decay) > 0.0:
        return float(decay)
    tot = max(int(total_episodes), 1)
    return max(float(tot - 1) / 4.0, 1.0)

def entropy_coef_for_episode(episode: int, total_episodes: int, *, ent_end: Optional[float]=None, ent_start: float=ENT_START, anneal_decay: float=ENT_ANNEAL_DECAY, enabled: bool=ENT_ANNEAL_ENABLED) -> float:
    e_end = float(ENT_COEF if ent_end is None else ent_end)
    if not enabled:
        return e_end
    tot = max(int(total_episodes), 1)
    ep = max(1, min(int(episode), tot))
    tau = entropy_anneal_decay_episodes(tot, anneal_decay)
    decay_factor = math.exp(-float(ep - 1) / tau)
    return e_end + (float(ent_start) - e_end) * decay_factor

def epsilon_for_episode(episode: int, total_episodes: int, cfg: PPOConfig) -> float:
    if not cfg.epsilon_greedy_enabled:
        return 0.0
    tot = max(int(total_episodes), 1)
    ep = max(1, min(int(episode), tot))
    eps_min = float(cfg.epsilon_min)
    eps_start = float(cfg.epsilon_start)
    tau = float(cfg.epsilon_decay)
    if tau <= 0.0:
        tau = max(float(tot - 1) / 4.0, 1.0)
    return eps_min + (eps_start - eps_min) * math.exp(-float(ep - 1) / tau)

def lr_for_episode(episode: int, total_episodes: int, *, lr_start: float=LR, lr_min: float=LR_MIN, decay_start: int=LR_DECAY_START_EP, decay_end: int=LR_DECAY_END_EP) -> float:
    ep = max(1, int(episode))
    lr0 = float(lr_start)
    lr1 = float(lr_min)
    t0 = max(1, int(decay_start))
    t1 = max(t0, int(decay_end))
    if ep < t0:
        return lr0
    if ep >= t1:
        return lr1
    span = max(t1 - t0, 1)
    progress = float(ep - t0) / float(span)
    return lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * progress))

def ppo_epochs_for_episode(episode: int) -> int:
    if int(episode) >= int(PPO_EPOCHS_LATE_START_EP):
        return int(PPO_EPOCHS_LATE)
    return int(PPO_EPOCHS)

def entropy_coefs_for_episode(episode: int, total_episodes: int, **kwargs) -> Tuple[float, float]:
    e = entropy_coef_for_episode(episode, total_episodes, **kwargs)
