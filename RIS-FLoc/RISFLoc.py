#!/usr/bin/env python3
"""RIS-FLoc entry point and public API for the localization network."""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

CUDA_DEVICE_IDS = "0,1"
if __name__ == "__main__" and CUDA_DEVICE_IDS.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_IDS.strip()

import random

import numpy as np
import torch

from risfloc.config import Config
from risfloc.data import (
    build_sparse_phase_path_grid,
    load_csv_rssi_column,
    parse_path_phase_segments,
)
from risfloc.evaluate import (
    _build_rp_xy_lut_tensor,
    _eval_topk_prob_mass_div,
    _eval_topk_weight_mode,
    _soft_weighted_xy_from_logits_batch,
    _valid_class_mask_tensor,
    apply_eval_localization_config,
    evaluate_localization,
)
from risfloc.model import MixedModel
from risfloc.normalize import z_score_apply_phase_grid, z_score_apply_vector
from risfloc.train import train_model
from risfloc.utils import _amp_autocast_ctx

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
torch.set_num_threads(int(getattr(Config, "TORCH_NUM_INTRAOP_THREADS", 8)))

__all__ = [
    "Config",
    "MixedModel",
    "train_model",
    "evaluate_localization",
    "apply_eval_localization_config",
    "build_sparse_phase_path_grid",
    "load_csv_rssi_column",
    "parse_path_phase_segments",
    "z_score_apply_phase_grid",
    "z_score_apply_vector",
    "_amp_autocast_ctx",
    "_build_rp_xy_lut_tensor",
    "_eval_topk_prob_mass_div",
    "_eval_topk_weight_mode",
    "_soft_weighted_xy_from_logits_batch",
    "_valid_class_mask_tensor",
]


if __name__ == "__main__":
    RETRAIN = True
    if RETRAIN:
        print("[mode] train RIS-FLoc sparse grid")
        _, mean_val, std_val = train_model()
    else:
        print("[mode] eval only")
        mean_val = np.load(Config.OUTPUT_MEAN_NPY)
        std_val = np.load(Config.OUTPUT_STD_NPY)
    evaluate_localization(mean_val, std_val)
