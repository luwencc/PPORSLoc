#!/usr/bin/env python3
"""Masked z-score normalization for RSS sequences and phase grids."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from .config import Config

def z_score_masked_fit(padded: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, Lmax = padded.shape
    cnt = np.zeros(Lmax, dtype=np.float64)
    s1 = np.zeros(Lmax, dtype=np.float64)
    s2 = np.zeros(Lmax, dtype=np.float64)
    chunk = int(getattr(Config, 'ZSCORE_INPLACE_CHUNK_ROWS', 4096))
    chunk = max(chunk, 1)
    t_idx = np.arange(Lmax, dtype=np.int32)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        sl = padded[i0:i1].astype(np.float64, copy=False)
        Ls = lengths[i0:i1]
        m = t_idx.reshape(1, -1) < Ls.reshape(-1, 1)
        cnt += np.sum(m, axis=0, dtype=np.float64)
        s1 += np.sum(sl * m, axis=0)
        s2 += np.sum(sl * sl * m, axis=0)
    mean = s1 / np.maximum(cnt, 1.0)
    var = s2 / np.maximum(cnt, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    std[std < 1e-06] = 1.0
    return (mean.astype(np.float32), std.astype(np.float32))

def z_score_apply_padded_inplace(padded: np.ndarray, lengths: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    N, Lmax = padded.shape
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.size < Lmax:
        mean = np.pad(mean, (0, int(Lmax - mean.size))).astype(np.float32)
    else:
        mean = mean[:Lmax]
    if std.size < Lmax:
        std = np.pad(std, (0, int(Lmax - std.size)), constant_values=1.0).astype(np.float32)
    else:
        std = std[:Lmax]
    st = np.maximum(std.reshape(1, -1), 1e-06)
    m = mean.reshape(1, -1)
    chunk = int(getattr(Config, 'ZSCORE_INPLACE_CHUNK_ROWS', 4096))
    chunk = max(chunk, 1)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        sl = padded[i0:i1]
        Ls = lengths[i0:i1]
        mask = np.arange(Lmax, dtype=np.int32).reshape(1, -1) < Ls.reshape(-1, 1)
        padded[i0:i1] = np.where(mask, (sl.astype(np.float32, copy=False) - m) / st, 0.0).astype(np.float32)

def z_score_apply_vector(rss: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    L = rss.size
    return (rss.astype(np.float32) - mean[:L]) / np.maximum(std[:L], 1e-06).astype(np.float32)

def z_score_apply_phase_grid(rss_2d: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.empty_like(rss_2d, dtype=np.float32)
    for ph in range(int(rss_2d.shape[0])):
        out[ph] = z_score_apply_vector(rss_2d[ph], mean, std)
    return out

def z_score_masked_fit_2d(padded: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, p, lmax = padded.shape
    flat = padded.reshape(n * p, lmax)
    lens_rep = np.repeat(np.asarray(lengths, dtype=np.int32), int(p))
    return z_score_masked_fit(flat, lens_rep)

def z_score_apply_padded_2d_inplace(padded: np.ndarray, lengths: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    p = int(padded.shape[1])
    for ph in range(p):
        z_score_apply_padded_inplace(padded[:, ph, :], lengths, mean, std)
