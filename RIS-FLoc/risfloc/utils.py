#!/usr/bin/env python3
"""Helpers for DataParallel unwrap and AMP autocast/scaler."""

from __future__ import annotations
import torch
import torch.nn as nn

def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m

def _amp_autocast_ctx(enabled: bool):
    import contextlib
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)

def _make_grad_scaler(enabled: bool):
    if not enabled or not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler('cuda')
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()
