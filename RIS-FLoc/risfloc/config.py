#!/usr/bin/env python3
"""Hyperparameters for RIS-FLoc sparse path-phase localization."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch


class _ConfigMeta(type):
    @property
    def RP_LIST(cls) -> List[str]:
        return cls._ensure_rp_list()

    @property
    def NUM_CLASSES(cls) -> int:
        return len(cls._ensure_rp_list())


class Config(metaclass=_ConfigMeta):
    TRAIN_PATH = "Trained_Data"
    VAL_PATH = "Validation_Data"
    TEST_PATH = "Test_Data"
    ORDINATE_CSV = "ordinate.csv"
    VALID_RP_CLASSES_JSON = "valid_rp_classes_DRL.json"
    OUTPUT_MODEL_PTH = "best_model_DRL.pth"
    OUTPUT_MEAN_NPY = "mean_DRL.npy"
    OUTPUT_STD_NPY = "std_DRL.npy"
    OUTPUT_TRAINING_CURVES_PNG = "training_curves_DRL.png"
    OUTPUT_VAL_CONFUSION_PNG = "val_confusion_matrix_DRL.png"
    OUTPUT_VAL_PER_CLASS_PNG = "val_per_class_accuracy_support_DRL.png"
    OUTPUT_VAL_ERROR_SCATTER_PNG = "val_error_scatter_DRL.png"
    OUTPUT_ERROR_CDF_PNG = "error_cdf_global_DRL.png"
    OUTPUT_ERROR_CDF_BY_PATH_DEPTH_PNG = "error_cdf_by_path_depth_DRL.png"
    OUTPUT_ERROR_CDF_TRAIN_TEST_PNG = "error_cdf_k{k}_DRL_train_test.png"
    OUTPUT_TEST_BAR_PNG = "test_localization_error_bar_DRL_train_test.png"
    OUTPUT_PICTURE_DIR = "Test_picture_DRL"
    MAX_SEQ_LEN = 152
    SEG_LEN = 38
    NUM_PHASES = 6
    NUM_PATHS = 4
    PATH_DEPTHS: Tuple[int, ...] = (1, 2, 3, 4)
    CACHE_DRL_TAG = "DRL_phgrid"
    CACHE_DRL_VERSION = "v1"
    BATCH_SIZE = 4096
    EPOCHS = 50
    LR = 0.004
    K_VALUE = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_DATA_PARALLEL = True
    USE_AMP = True
    USE_TORCH_COMPILE = False
    LABEL_SMOOTHING = 0.0
    USE_WEIGHTED_SAMPLER = True
    USE_CLASS_WEIGHTS = True
    LOSS_XY_WEIGHT = 1.8
    EVAL_FUSE_XY_WEIGHT = 0.72
    EVAL_TOPK_WEIGHT_MODE = "full_softmax_no_renorm"
    EVAL_TOPK_PROB_MASS_DIV = 1.0
    RP_SPACING_M = 1.6
    EVAL_ERROR_OUTLIER_MAX_M = 50.0 * RP_SPACING_M
    DATALOADER_NUM_WORKERS = 16
    TORCH_NUM_INTRAOP_THREADS = 16
    ZSCORE_INPLACE_CHUNK_ROWS = 4096
    TRAIN_SCAN_READ_WORKERS = 0
    TRAIN_SCAN_USE_THREADS = True
    TEST_SCAN_READ_WORKERS = 64
    TEST_SCAN_BATCH_SIZE = 50000
    TEST_EVAL_BATCH_SIZE: Optional[int] = 4096
    TEST_EVAL_SINGLE_GPU = True
    TEST_DATALOADER_NUM_WORKERS: Optional[int] = 0
    TEST_DATALOADER_PREFETCH_FACTOR = 2

    _rp_list_cache: Optional[List[str]] = None

    @classmethod
    def _ensure_rp_list(cls) -> List[str]:
        if cls._rp_list_cache is None:
            import pandas as pd

            path = Path(cls.ORDINATE_CSV)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path.resolve()}. Place ordinate.csv next to the training script "
                    "(download it with the dataset release if needed)."
                )
            cls._rp_list_cache = [str(c) for c in pd.read_csv(path).columns]
        return cls._rp_list_cache
