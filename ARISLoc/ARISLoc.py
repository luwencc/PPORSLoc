#!/usr/bin/env python3
"""ARISLoc entry point: PPO training for path-phase RIS localization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RISFLOC = Path(__file__).resolve().parent.parent / "RIS-FLoc"
if str(_RISFLOC) not in sys.path:
    sys.path.insert(0, str(_RISFLOC))

import RISFLoc as loc
from arisloc.config import (
    DRL_DATA_ROOT,
    ENT_ANNEAL_ENABLED,
    EPSILON_GREEDY_ENABLED,
    EVAL_ACTION_SEED,
    EVAL_TOPK_PROB_MASS_DIV,
    EVAL_TOPK_WEIGHT_MODE,
    LOC_INFER_BATCH_SIZE,
    REQUIRE_EXPLICIT_STOP,
    SYNTHETIC_RSS,
    TRAJ_CACHE_MAX_GB,
    TRAIN_TOTAL_EPISODES,
    default_ppo_config,
)
from arisloc.localizer import apply_drl_localizer_eval_config
from arisloc.train import train_ppo


def main():
    ap = argparse.ArgumentParser(description="ARISLoc: PPO over all DRL reference points.")
    ap.add_argument("--drl-root", type=str, default=None, help=f"Override DRL_DATA_ROOT (default {DRL_DATA_ROOT})")
    ap.add_argument("--episodes", type=int, default=None, help=f"Override TRAIN_TOTAL_EPISODES (default {TRAIN_TOTAL_EPISODES})")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic RSS")
    ap.add_argument("--device", type=str, default=None, help="Override DEVICE")
    ap.add_argument("--cache-gb", type=float, default=None, help=f"Override TRAJ_CACHE_MAX_GB (default {TRAJ_CACHE_MAX_GB})")
    ap.add_argument("--loc-batch", type=int, default=None, help=f"Override LOC_INFER_BATCH_SIZE (default {LOC_INFER_BATCH_SIZE})")
    ap.add_argument("--topk-weight-mode", choices=("topk_softmax", "full_softmax_no_renorm"), default=None)
    ap.add_argument("--prob-mass-div", type=float, default=None, help=f"Softmax mass divisor (default {EVAL_TOPK_PROB_MASS_DIV})")
    ap.add_argument("--eval-greedy", action="store_true", help="Greedy argmax for CDF/phase eval")
    ap.add_argument("--eval-action-seed", type=int, default=None, help=f"Eval sampling seed (default {EVAL_ACTION_SEED}); -1=unset")
    args = ap.parse_args()

    import arisloc.config as C
    if args.eval_greedy:
        C.DRL_EVAL_DETERMINISTIC = True
    if args.eval_action_seed is not None:
        C.EVAL_ACTION_SEED = None if int(args.eval_action_seed) < 0 else int(args.eval_action_seed)

    apply_drl_localizer_eval_config(
        topk_weight_mode=args.topk_weight_mode if args.topk_weight_mode is not None else EVAL_TOPK_WEIGHT_MODE,
        prob_mass_div=args.prob_mass_div,
    )
    cfg = default_ppo_config()
    if args.drl_root is not None:
        cfg.drl_data_root = Path(args.drl_root)
    if args.episodes is not None:
        cfg.train_total_episodes = int(args.episodes)
    if args.device is not None:
        cfg.device = str(args.device)
    if args.cache_gb is not None:
        cfg.traj_cache_max_gb = float(args.cache_gb)
        cfg.traj_cache_enabled = float(args.cache_gb) > 0
    if args.loc_batch is not None:
        cfg.loc_infer_batch_size = max(1, int(args.loc_batch))
    cfg.synthetic_rss = bool(SYNTHETIC_RSS or args.synthetic)

    print(
        f"RPs={len(loc.Config.RP_LIST)} | PPO 7 actions (phase0-5+stop) | "
        f"ent={'exp-anneal' if ENT_ANNEAL_ENABLED else 'fixed'} | "
        f"explicit_stop={'on' if REQUIRE_EXPLICIT_STOP else 'off'} | "
        f"eps-greedy={'on' if EPSILON_GREEDY_ENABLED else 'off'} | "
        f"seg={cfg.seg_len_m}m | root={cfg.drl_data_root.resolve()}"
    )
    train_ppo(cfg)


if __name__ == "__main__":
    main()
