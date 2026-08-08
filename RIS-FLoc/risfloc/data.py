#!/usr/bin/env python3
"""CSV loading, sparse path-phase grids, caches, and RSS datasets."""

from __future__ import annotations
import gc, hashlib, os, re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from .config import Config
from .normalize import z_score_apply_phase_grid, z_score_masked_fit_2d

def parse_path_phase_segments(dirname: str, rp: Optional[str]=None) -> Optional[List[Tuple[int, int]]]:
    rest = str(dirname).strip()
    if rp:
        prefix = str(rp) + '_'
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
    if not rest:
        return None
    toks = rest.split('_')
    out: List[Tuple[int, int]] = []
    i = 0
    while i + 1 < len(toks):
        if not toks[i].lower().startswith('path'):
            return None
        m1 = re.match('path(\\d+)$', toks[i], re.IGNORECASE)
        if not m1:
            return None
        if not toks[i + 1].lower().startswith('phase'):
            return None
        m2 = re.match('phase(\\d+)$', toks[i + 1], re.IGNORECASE)
        if not m2:
            return None
        out.append((int(m1.group(1)), int(m2.group(1))))
        i += 2
    return out if out else None

def _effective_length_from_segments(segments: Sequence[Tuple[int, int]]) -> int:
    seg = int(Config.SEG_LEN)
    max_path = max((int(p) for p, _ in segments))
    return int(np.clip(max_path * seg, seg, int(Config.MAX_SEQ_LEN)))

def _split_rss_for_segments(rss: np.ndarray, n_seg: int) -> Optional[List[np.ndarray]]:
    seg = int(Config.SEG_LEN)
    if n_seg <= 0:
        return None
    rss = np.asarray(rss, dtype=np.float32).ravel()
    if rss.size == 0:
        return None
    if rss.size != n_seg * seg:
        return None
    return [rss[i * seg:(i + 1) * seg].copy() for i in range(n_seg)]

def build_sparse_phase_path_grid(rss: np.ndarray, segments: Sequence[Tuple[int, int]]) -> Optional[Tuple[np.ndarray, int]]:
    seg = int(Config.SEG_LEN)
    n_ph = int(Config.NUM_PHASES)
    n_paths = int(Config.NUM_PATHS)
    Lmax = int(Config.MAX_SEQ_LEN)
    n_seg = len(segments)
    if n_seg == 0:
        return None
    rss = np.asarray(rss, dtype=np.float32).ravel()
    if rss.size != n_seg * seg:
        return None
    grid = np.zeros((n_ph, Lmax), dtype=np.float32)
    for i, (path_n, phase_n) in enumerate(segments):
        if not 1 <= int(path_n) <= n_paths:
            return None
        if not 0 <= int(phase_n) < n_ph:
            return None
        col0 = (int(path_n) - 1) * seg
        col1 = col0 + seg
        if col1 > Lmax:
            return None
        grid[int(phase_n), col0:col1] = rss[i * seg:(i + 1) * seg]
    L_eff = _effective_length_from_segments(segments)
    return (grid, L_eff)

def _sparse_labeled_scan_one_csv_task(task: Tuple[str, int, Tuple[Tuple[int, int], ...]]) -> Optional[Tuple[np.ndarray, int, int, int]]:
    csv_path, lab, segments = task
    try:
        rss = load_csv_rssi_column(csv_path)
    except Exception:
        return None
    if rss.size == 0:
        return None
    built = build_sparse_phase_path_grid(rss, segments)
    if built is None:
        return None
    mat, L_eff = built
    depth = int(max((int(p) for p, _ in segments)))
    return (mat.astype(np.float32, copy=False), int(lab), depth, int(L_eff))

def _scan_executor_for_tasks(n_tasks: int, num_workers: int):
    n_par = int(num_workers)
    if n_par <= 1:
        return (None, 0, 0)
    max_w = min(n_par, n_tasks, max(1, os.cpu_count() or 1))
    chunksize = max(1, min(1024, n_tasks // max(1, max_w * 4)))
    use_threads = bool(getattr(Config, 'TRAIN_SCAN_USE_THREADS', True))
    if use_threads or n_tasks >= 500000:
        tqdm.write(f'[scan] ThreadPoolExecutor workers={max_w} chunksize={chunksize} (I/O parallel reads)')
        return (ThreadPoolExecutor(max_workers=max_w), max_w, chunksize)
    tqdm.write(f'[scan] ProcessPoolExecutor workers={max_w} chunksize={chunksize} (small data; use TRAIN_SCAN_USE_THREADS=True for huge sets)')
    return (ProcessPoolExecutor(max_workers=max_w), max_w, chunksize)

def enumerate_sparse_labeled_samples(data_root: Path, num_workers: int=0, sequences_npy_out: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tag = data_root.name or str(data_root)
    n_ph = int(Config.NUM_PHASES)
    Lmax = int(Config.MAX_SEQ_LEN)
    tasks: List[Tuple[str, int, Tuple[Tuple[int, int], ...]]] = []
    rp_pbar = tqdm(Config.RP_LIST, desc=f'enum CSV [{tag}]', unit='rp', dynamic_ncols=True)
    for rp in rp_pbar:
        lab = Config.RP_LIST.index(rp)
        rp_dir = data_root / rp
        if not rp_dir.is_dir():
            continue
        subdirs = sorted((d for d in rp_dir.iterdir() if d.is_dir()))
        for sub in subdirs:
            segments = parse_path_phase_segments(sub.name, rp)
            if not segments:
                continue
            seg_t = tuple(((int(p), int(ph)) for p, ph in segments))
            files = sorted((p for p in sub.glob('*.csv') if p.is_file()))
            for csv_p in files:
                tasks.append((str(csv_p), lab, seg_t))
        rp_pbar.set_postfix(files=len(tasks))
    if not tasks:
        empty3 = np.zeros((0, n_ph, Lmax), dtype=np.float32)
        return (empty3, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int32))
    n_tasks = len(tasks)
    tqdm.write(f'[scan] {n_tasks} CSV files with sparse path×phase labels under {data_root.resolve()}')
    mmap_tmp = _cache_prefix(str(data_root.resolve())) + f'_scan_tmp_{Config.CACHE_DRL_TAG}_{int(Config.MAX_SEQ_LEN)}.mmap'
    if os.path.isfile(mmap_tmp):
        try:
            os.remove(mmap_tmp)
        except OSError:
            pass
    mm = np.lib.format.open_memmap(mmap_tmp, mode='w+', dtype=np.float32, shape=(n_tasks, n_ph, Lmax))
    labels_buf = np.zeros(n_tasks, dtype=np.int64)
    depths_buf = np.zeros(n_tasks, dtype=np.int8)
    lengths_buf = np.zeros(n_tasks, dtype=np.int32)
    valid_idx = 0

    def _store_result(r: Optional[Tuple[np.ndarray, int, int, int]]) -> int:
        nonlocal valid_idx
        if r is None:
            return valid_idx
        mat, lab, depth, L_eff = r
        i = valid_idx
        mm[i, :, :L_eff] = mat[:, :L_eff]
        labels_buf[i] = int(lab)
        depths_buf[i] = int(depth)
        lengths_buf[i] = int(L_eff)
        valid_idx += 1
        return valid_idx
    ex, max_w, chunksize = _scan_executor_for_tasks(n_tasks, num_workers)
    try:
        if ex is None:
            pbar = tqdm(tasks, desc=f'read CSV [{tag}]', unit='file', dynamic_ncols=True)
            for task in pbar:
                _store_result(_sparse_labeled_scan_one_csv_task(task))
                pbar.set_postfix(ok=valid_idx)
        else:
            with ex:
                for r in tqdm(ex.map(_sparse_labeled_scan_one_csv_task, tasks, chunksize=chunksize), total=n_tasks, desc=f'read CSV parallel [{tag}]', unit='file', dynamic_ncols=True):
                    _store_result(r)
    finally:
        try:
            mm.flush()
        except Exception:
            pass
    n_valid = valid_idx
    skipped = n_tasks - n_valid
    tqdm.write(f'[scan] loaded {n_valid} valid sparse grids ({skipped} skipped)')
    if n_valid == 0:
        try:
            del mm
            os.remove(mmap_tmp)
        except OSError:
            pass
        return (np.zeros((0, n_ph, Lmax), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int32))
    labels_out = labels_buf[:n_valid].copy()
    depths_out = depths_buf[:n_valid].copy()
    lengths_out = lengths_buf[:n_valid].copy()
    try:
        del mm
    except Exception:
        pass
    if sequences_npy_out:
        _finalize_scan_memmap_to_npy(mmap_tmp, n_valid, sequences_npy_out, n_ph, Lmax)
        sequences_out = np.zeros((0, n_ph, Lmax), dtype=np.float32)
    else:
        sequences_out = np.asarray(np.load(mmap_tmp, mmap_mode='r')[:n_valid])
        try:
            os.remove(mmap_tmp)
        except OSError:
            pass
    return (sequences_out, labels_out, depths_out, lengths_out)

def _test_sparse_scan_one_csv_task(task: Tuple[str, Tuple[Tuple[int, int], ...], str, float, float]) -> Optional[Tuple[np.ndarray, int, int, str, List[float]]]:
    csv_path, segments, rp_name, tx, ty = task
    try:
        rss = load_csv_rssi_column(csv_path)
    except Exception:
        return None
    if rss.size == 0:
        return None
    built = build_sparse_phase_path_grid(rss, segments)
    if built is None:
        return None
    mat, L_eff = built
    depth = int(max((int(p) for p, _ in segments)))
    return (mat.astype(np.float32, copy=False), int(L_eff), depth, rp_name, [tx, ty])

def _count_test_sparse_csvs(test_root: Path, coord: pd.DataFrame) -> int:
    cols = set(coord.columns.astype(str).tolist())
    n = 0
    rp_dirs = sorted((d for d in test_root.iterdir() if d.is_dir()))
    for true_rp in tqdm(rp_dirs, desc='count test CSV', unit='rp', dynamic_ncols=True):
        rp_name = true_rp.name
        if rp_name not in cols:
            continue
        try:
            float(coord.loc[0, rp_name])
            float(coord.loc[1, rp_name])
        except (KeyError, TypeError, ValueError):
            continue
        for sub in true_rp.iterdir():
            if not sub.is_dir():
                continue
            if not parse_path_phase_segments(sub.name, rp_name):
                continue
            for csv_p in sub.glob('*.csv'):
                if csv_p.is_file():
                    n += 1
    return n

def build_test_drl_cache_from_scan(test_root: Path, coord: pd.DataFrame, lens_f: str, depth_f: str, rss_pad_f: str, rp_f: str, xy_f: str) -> int:
    n_tasks = _count_test_sparse_csvs(test_root, coord)
    if n_tasks == 0:
        tqdm.write(f'[scan] 0 test CSV under {test_root.resolve()}')
        return 0
    est_gb = n_tasks * int(Config.NUM_PHASES) * int(Config.MAX_SEQ_LEN) * 4 / 1000000000.0
    tqdm.write(f'[scan] {n_tasks} test CSV files under {test_root.resolve()} (rss_padded ~ {est_gb:.1f} GB cap, streaming write)')
    n_ph = int(Config.NUM_PHASES)
    Lmax = int(Config.MAX_SEQ_LEN)
    mmap_tmp = _cache_prefix(str(test_root.resolve())) + f'_scan_tmp_{Config.CACHE_DRL_TAG}_{Lmax}.mmap'
    if os.path.isfile(mmap_tmp):
        try:
            os.remove(mmap_tmp)
        except OSError:
            pass
    mm = np.lib.format.open_memmap(mmap_tmp, mode='w+', dtype=np.float32, shape=(n_tasks, n_ph, Lmax))
    lens_buf = np.zeros(n_tasks, dtype=np.int32)
    depths_buf = np.zeros(n_tasks, dtype=np.int8)
    rp_buf = np.empty(n_tasks, dtype='U16')
    xy_buf = np.zeros((n_tasks, 2), dtype=np.float32)
    valid_idx = 0
    cols = set(coord.columns.astype(str).tolist())
    batch_size = max(1000, int(getattr(Config, 'TEST_SCAN_BATCH_SIZE', 50000)))
    nw = int(getattr(Config, 'TEST_SCAN_READ_WORKERS', 0))
    ex, _max_w, chunksize = _scan_executor_for_tasks(n_tasks, nw)
    batch: List[Tuple[str, Tuple[Tuple[int, int], ...], str, float, float]] = []
    pbar = tqdm(total=n_tasks, desc='read test CSV', unit='file', dynamic_ncols=True, mininterval=1.0)

    def _store_result(r: Optional[Tuple[np.ndarray, int, int, str, List[float]]]) -> None:
        nonlocal valid_idx
        if r is None:
            return
        mat, L_eff, depth, rp_name, xy = r
        i = valid_idx
        Li = int(L_eff)
        mm[i, :, :Li] = mat[:, :Li]
        lens_buf[i] = Li
        depths_buf[i] = int(depth)
        rp_buf[i] = str(rp_name)
        xy_buf[i, 0] = float(xy[0])
        xy_buf[i, 1] = float(xy[1])
        valid_idx += 1

    def _process_batch(tasks: Sequence[Tuple[str, Tuple[Tuple[int, int], ...], str, float, float]]) -> None:
        if not tasks:
            return
        if ex is None:
            for task in tasks:
                _store_result(_test_sparse_scan_one_csv_task(task))
        else:
            for r in ex.map(_test_sparse_scan_one_csv_task, tasks, chunksize=chunksize):
                _store_result(r)
    try:
        if ex is not None:
            ex.__enter__()
        for true_rp in sorted((d for d in test_root.iterdir() if d.is_dir())):
            rp_name = true_rp.name
            if rp_name not in cols:
                continue
            try:
                tx = float(coord.loc[0, rp_name])
                ty = float(coord.loc[1, rp_name])
            except (KeyError, TypeError, ValueError):
                continue
            for sub in true_rp.iterdir():
                if not sub.is_dir():
                    continue
                segments = parse_path_phase_segments(sub.name, rp_name)
                if not segments:
                    continue
                seg_t = tuple(((int(p), int(ph)) for p, ph in segments))
                for csv_p in sub.glob('*.csv'):
                    if not csv_p.is_file():
                        continue
                    batch.append((str(csv_p), seg_t, rp_name, tx, ty))
                    if len(batch) >= batch_size:
                        _process_batch(batch)
                        pbar.update(len(batch))
                        pbar.set_postfix(ok=valid_idx)
                        batch = []
        if batch:
            _process_batch(batch)
            pbar.update(len(batch))
            pbar.set_postfix(ok=valid_idx)
    finally:
        pbar.close()
        if ex is not None:
            ex.__exit__(None, None, None)
        try:
            mm.flush()
        except Exception:
            pass
        del mm
    n_valid = valid_idx
    skipped = n_tasks - n_valid
    tqdm.write(f'[scan] loaded {n_valid} valid test grids ({skipped} skipped)')
    if n_valid == 0:
        try:
            os.remove(mmap_tmp)
        except OSError:
            pass
        return 0
    _finalize_scan_memmap_to_npy(mmap_tmp, n_valid, rss_pad_f, n_ph, Lmax)
    np.save(lens_f, lens_buf[:n_valid])
    np.save(depth_f, depths_buf[:n_valid])
    np.save(rp_f, rp_buf[:n_valid])
    np.save(xy_f, xy_buf[:n_valid])
    del lens_buf, depths_buf, rp_buf, xy_buf
    gc.collect()
    return n_valid

def enumerate_test_sparse_samples(test_root: Path, coord: pd.DataFrame) -> Tuple[List[int], List[int], List[np.ndarray], List[str], List[List[float]]]:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, 'test_tmp')
        tag = f'{Config.CACHE_DRL_TAG}_{Config.CACHE_DRL_VERSION}_p{int(Config.NUM_PHASES)}_L{int(Config.MAX_SEQ_LEN)}'
        lens_f = f'{prefix}_rp52_{tag}_lens.npy'
        depth_f = f'{prefix}_rp52_{tag}_depths.npy'
        rss_pad_f = f'{prefix}_rp52_{tag}_rss_padded.npy'
        rp_f = f'{prefix}_rp52_{tag}_true_rp.npy'
        xy_f = f'{prefix}_rp52_{tag}_true_xy.npy'
        n_valid = build_test_drl_cache_from_scan(test_root, coord, lens_f, depth_f, rss_pad_f, rp_f, xy_f)
        if n_valid == 0:
            return ([], [], [], [], [])
        if n_valid > 100000:
            raise RuntimeError(f'Test samples {n_valid} too many; use build_test_drl_cache_from_scan')
        lens = np.load(lens_f)
        depths = np.load(depth_f)
        rss = np.load(rss_pad_f, mmap_mode='r')
        rp_names = np.load(rp_f, allow_pickle=True)
        xy = np.load(xy_f)
        rss_list = [np.asarray(rss[i], dtype=np.float32) for i in range(n_valid)]
        return (lens.tolist(), depths.tolist(), rss_list, list(rp_names), xy.tolist())

def _picture_output_path(filename: str) -> str:
    root = str(getattr(Config, 'OUTPUT_PICTURE_DIR', 'Test_picture_DRL'))
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, filename)

def _build_xy_from_labels(labels: np.ndarray) -> np.ndarray:
    coord = pd.read_csv(Config.ORDINATE_CSV)
    cols = set(coord.columns.astype(str).tolist())
    ir = np.asarray(labels, dtype=np.int64)
    xs = np.zeros(len(labels), dtype=np.float32)
    ys = np.zeros(len(labels), dtype=np.float32)
    for j, rp in enumerate(Config.RP_LIST):
        if rp not in cols:
            continue
        m = ir == j
        if not np.any(m):
            continue
        xs[m] = float(coord.loc[0, rp])
        ys[m] = float(coord.loc[1, rp])
    return np.stack([xs, ys], axis=1)

def _load_csv_rssi_column_pandas(csv_path: str) -> np.ndarray:
    path = str(csv_path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=np.float32)
    except (OSError, ValueError, UnicodeDecodeError):
        return np.empty(0, dtype=np.float32)
    if df.empty or len(df.columns) == 0:
        return np.empty(0, dtype=np.float32)
    if 'RSSI' in df.columns:
        rss = df['RSSI'].dropna().values
    elif 'filtered_rss' in df.columns:
        rss = df['filtered_rss'].dropna().values
    else:
        rss = df.iloc[:, 0].dropna().values
    return np.asarray(rss, dtype=np.float32)

def load_csv_rssi_column(csv_path: str) -> np.ndarray:
    path = str(csv_path)
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return np.empty(0, dtype=np.float32)
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            header = f.readline()
        if not header.strip():
            return np.empty(0, dtype=np.float32)
        cols = [c.strip().strip('"').strip("'") for c in header.split(',')]
        col_idx = 0
        if 'RSSI' in cols:
            col_idx = cols.index('RSSI')
        elif 'filtered_rss' in cols:
            col_idx = cols.index('filtered_rss')
        arr = np.loadtxt(path, delimiter=',', skiprows=1, usecols=col_idx, dtype=np.float32, ndmin=1)
        if arr.ndim == 0:
            arr = np.array([float(arr)], dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return arr.astype(np.float32, copy=False)
    except (OSError, ValueError, UnicodeDecodeError):
        return _load_csv_rssi_column_pandas(path)

def _cache_prefix(data_path: str) -> str:
    return os.path.abspath(data_path).replace(os.sep, '_').replace(':', '_')

def _test_drl_cache_filenames(test_prefix: str) -> Tuple[str, str, str, str, str]:
    tag = f'{Config.CACHE_DRL_TAG}_{Config.CACHE_DRL_VERSION}_p{int(Config.NUM_PHASES)}_L{int(Config.MAX_SEQ_LEN)}'
    lens_f = f'{test_prefix}_rp52_{tag}_lens.npy'
    depth_f = f'{test_prefix}_rp52_{tag}_depths.npy'
    rss_pad_f = f'{test_prefix}_rp52_{tag}_rss_padded.npy'
    rp_f = f'{test_prefix}_rp52_{tag}_true_rp.npy'
    xy_f = f'{test_prefix}_rp52_{tag}_true_xy.npy'
    return (lens_f, depth_f, rss_pad_f, rp_f, xy_f)

def _depth_from_seq_length(length: int) -> int:
    seg = int(Config.SEG_LEN)
    if seg <= 0:
        return 1
    L = int(length)
    if L <= 0:
        return 1
    d = L // seg
    if L % seg != 0:
        d = max(1, int(round(L / seg)))
    return int(np.clip(d, 1, int(Config.NUM_PATHS)))

def _resolve_test_depths(lens: np.ndarray, depth_path: Optional[str]=None) -> np.ndarray:
    lens = np.asarray(lens, dtype=np.int32)
    if depth_path and os.path.isfile(depth_path):
        depths = np.asarray(np.load(depth_path), dtype=np.int8).reshape(-1)
        if depths.shape[0] == lens.shape[0]:
            return depths
    return np.asarray([_depth_from_seq_length(int(L)) for L in lens], dtype=np.int8)

def _path_depth_label(depth: int) -> str:
    d = int(depth)
    seg = int(Config.SEG_LEN)
    if d <= 1:
        return f'path1 (L={seg})'
    parts = [f'path{i}' for i in range(1, d + 1)]
    return f'{'+'.join(parts)} (L={d * seg})'

def _write_test_rss_list_to_padded_file(lens_list: Sequence[int], rss_list: Sequence[np.ndarray], out_path: str, Lmax: int) -> None:
    n = len(rss_list)
    n_ph = int(Config.NUM_PHASES)
    tmp_path = out_path + '.tmp'
    mm = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.float32, shape=(n, n_ph, Lmax))
    try:
        for i in tqdm(range(n), desc='write test rss_padded DRL', unit='seq', dynamic_ncols=True):
            Li = int(lens_list[i])
            s = np.asarray(rss_list[i], dtype=np.float32)
            if s.ndim == 1:
                L = min(Li, int(s.size), Lmax)
                if L > 0:
                    mm[i, 0, :L] = s[:L]
            else:
                L = min(Li, int(s.shape[1]), Lmax)
                rows = min(int(s.shape[0]), n_ph)
                if L > 0 and rows > 0:
                    mm[i, :rows, :L] = s[:rows, :L]
        mm.flush()
    finally:
        del mm
    os.replace(tmp_path, out_path)

def _rp_list_fingerprint() -> str:
    return hashlib.sha256(','.join(Config.RP_LIST).encode('utf-8')).hexdigest()[:10]

def _dataset_cache_npz_path(data_path: str) -> str:
    prefix = _cache_prefix(data_path)
    rp_key = _rp_list_fingerprint()
    return f'{prefix}_rp52_{Config.CACHE_DRL_TAG}_{Config.CACHE_DRL_VERSION}_s{int(Config.SEG_LEN)}_L{int(Config.MAX_SEQ_LEN)}_p{int(Config.NUM_PHASES)}_rp{rp_key}.npz'

def _dataset_cache_sequences_npy_path(data_path: str) -> str:
    return _dataset_cache_npz_path(data_path).replace('.npz', '_sequences.npy')

def _finalize_scan_memmap_to_npy(mmap_tmp: str, n_valid: int, out_path: str, n_ph: int, Lmax: int) -> None:
    tmp_out = out_path + '.tmp'
    if os.path.isfile(tmp_out):
        try:
            os.remove(tmp_out)
        except OSError:
            pass
    src = np.load(mmap_tmp, mmap_mode='r')
    dst = np.lib.format.open_memmap(tmp_out, mode='w+', dtype=np.float32, shape=(n_valid, n_ph, Lmax))
    chunk = int(getattr(Config, 'ZSCORE_INPLACE_CHUNK_ROWS', 4096))
    chunk = max(chunk, 1)
    try:
        for i0 in tqdm(range(0, n_valid, chunk), desc='finalize sequences npy', unit='chunk', dynamic_ncols=True):
            i1 = min(i0 + chunk, n_valid)
            dst[i0:i1] = src[i0:i1]
        dst.flush()
    finally:
        del dst, src
    os.replace(tmp_out, out_path)
    try:
        os.remove(mmap_tmp)
    except OSError:
        pass

def _write_sequences_3d_to_npy(sequences: np.ndarray, out_path: str) -> None:
    seqs = np.asarray(sequences, dtype=np.float32)
    if seqs.ndim != 3:
        raise ValueError(f'expected (N,P,L) got {seqs.shape}')
    n, n_ph, Lmax = seqs.shape
    tmp_out = out_path + '.tmp'
    if os.path.isfile(tmp_out):
        try:
            os.remove(tmp_out)
        except OSError:
            pass
    mm = np.lib.format.open_memmap(tmp_out, mode='w+', dtype=np.float32, shape=(n, n_ph, Lmax))
    chunk = int(getattr(Config, 'ZSCORE_INPLACE_CHUNK_ROWS', 4096))
    chunk = max(chunk, 1)
    try:
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            mm[i0:i1] = seqs[i0:i1]
        mm.flush()
    finally:
        del mm
    os.replace(tmp_out, out_path)

def _migrate_legacy_npz_sequences_to_npy(cache_npz: str, seq_npy: str) -> None:
    tqdm.write(f'[cache] migrate legacy npz sequences -> mmap {seq_npy}')
    z = np.load(cache_npz, allow_pickle=True)
    lengths = z['lengths'].astype(np.int32)
    labels = z['labels'].astype(np.int64)
    depths = z.get('depths')
    _write_sequences_3d_to_npy(z['sequences'], seq_npy)
    del z
    gc.collect()
    depths_arr = np.asarray(depths, dtype=np.int8) if depths is not None else np.zeros(len(lengths), dtype=np.int8)
    np.savez_compressed(cache_npz, lengths=lengths, labels=labels, depths=depths_arr)
    tqdm.write(f'[cache] migrated {cache_npz} (meta only) + {seq_npy}')

def rss_collate_fn(batch):
    xs, lens, ys, xys = zip(*batch)
    lens_t = torch.tensor(lens, dtype=torch.long)
    max_l = int(lens_t.max().item())
    B = len(xs)
    n_ph = int(Config.NUM_PHASES)
    padded = torch.zeros(B, n_ph, max_l, dtype=torch.float32)
    for i, (xv, L) in enumerate(zip(xs, lens)):
        L = int(L)
        padded[i, :, :L] = xv[:, :L].float()
    return (padded, lens_t, torch.stack(list(ys)), torch.stack(list(xys)))

class ClassBalancedIndexSampler(Sampler[int]):

    def __init__(self, labels: np.ndarray, num_classes: int, num_samples: Optional[int]=None):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_classes = int(num_classes)
        n = int(self.labels.size)
        self.num_samples = n if num_samples is None else int(num_samples)
        cnt = np.bincount(np.clip(self.labels, 0, self.num_classes - 1), minlength=self.num_classes).astype(np.float64)
        w = 1.0 / np.maximum(cnt, 1.0)
        w = w * self.num_classes / np.sum(w)
        self._class_weights = torch.as_tensor(w, dtype=torch.double)
        tqdm.write(f'[sampler] ClassBalancedIndexSampler: argsort N={n} for class ranges …')
        self._order = np.argsort(self.labels, kind='mergesort').astype(np.int64)
        sy = self.labels[self._order]
        self._offsets = np.searchsorted(sy, np.arange(self.num_classes + 1, dtype=np.int64), side='left').astype(np.int64)

    def __iter__(self):
        nonempty = [c for c in range(self.num_classes) if self._offsets[c] < self._offsets[c + 1]]
        if not nonempty:
            raise RuntimeError('ClassBalancedIndexSampler: no samples in any class')
        for _ in range(self.num_samples):
            c = int(torch.multinomial(self._class_weights, 1, replacement=True).item())
            lo, hi = (int(self._offsets[c]), int(self._offsets[c + 1]))
            if lo >= hi:
                c = int(np.random.choice(nonempty))
                lo, hi = (int(self._offsets[c]), int(self._offsets[c + 1]))
            j = lo + int(np.random.randint(0, hi - lo))
            yield int(self._order[j])

    def __len__(self) -> int:
        return self.num_samples

class RSSDataset(Dataset):

    def __init__(self, data_path: str, mean: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None, is_train: bool=True):
        self.is_train = is_train
        self.augment_noise = 0.01
        self.Lmax = int(Config.MAX_SEQ_LEN)
        self.n_phases = int(Config.NUM_PHASES)
        cache_npz = _dataset_cache_npz_path(data_path)
        seq_npy = _dataset_cache_sequences_npy_path(data_path)
        tag_ds = os.path.basename(os.path.normpath(data_path)) or data_path
        data_root = Path(data_path)
        if os.path.exists(cache_npz) and os.path.exists(seq_npy):
            tqdm.write(f'\n[cache] load meta {cache_npz}')
            z = np.load(cache_npz, allow_pickle=True)
            self.lengths = z['lengths'].astype(np.int32)
            self.labels = z['labels'].astype(np.int64)
            tqdm.write(f'[cache] mmap sequences {seq_npy}')
            self.sequences = np.load(seq_npy, mmap_mode='r')
        elif os.path.exists(cache_npz):
            with np.load(cache_npz, allow_pickle=True) as z_probe:
                if 'sequences' not in z_probe.files:
                    raise FileNotFoundError(f'Found {cache_npz} but missing {seq_npy}; delete cache and rescan.')
            _migrate_legacy_npz_sequences_to_npy(cache_npz, seq_npy)
            z = np.load(cache_npz, allow_pickle=True)
            self.lengths = z['lengths'].astype(np.int32)
            self.labels = z['labels'].astype(np.int64)
            tqdm.write(f'[cache] mmap sequences {seq_npy}')
            self.sequences = np.load(seq_npy, mmap_mode='r')
        else:
            tqdm.write(f'\n[scan] sparse path×phase grids under {data_root.resolve()}')
            n_scan_w = int(getattr(Config, 'TRAIN_SCAN_READ_WORKERS', 0))
            _, self.labels, depths_arr, self.lengths = enumerate_sparse_labeled_samples(data_root, num_workers=n_scan_w, sequences_npy_out=seq_npy)
            n = int(self.lengths.shape[0])
            if n == 0:
                raise RuntimeError(f'No sparse-labeled samples under {data_root}. Need <RP>/path*_phase* folders')
            self.lengths = np.asarray(self.lengths, dtype=np.int32)
            self.labels = np.asarray(self.labels, dtype=np.int64)
            np.savez_compressed(cache_npz, lengths=self.lengths, labels=self.labels, depths=np.asarray(depths_arr, dtype=np.int8))
            tqdm.write(f'[cache] saved meta {cache_npz} + mmap {seq_npy} n={n}')
            self.sequences = np.load(seq_npy, mmap_mode='r')
        if self.sequences.ndim != 3:
            raise ValueError(f'Bad cache dims (expected N×{self.n_phases}×L): {seq_npy}; delete legacy 1D/varlen cache and rescan.')
        self.xy = _build_xy_from_labels(self.labels)
        if mean is None or std is None:
            mean_arr, std_arr = z_score_masked_fit_2d(self.sequences, self.lengths)
            self.mean = mean_arr
            self.std = std_arr
        else:
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.is_train:
            noise = np.random.normal(0, self.augment_noise, x.shape).astype(np.float32)
            x = x + noise
            x = np.clip(x, -5, 5)
        return x

    def __len__(self):
        return int(self.sequences.shape[0])

    def __getitem__(self, i):
        L = int(self.lengths[i])
        x = np.asarray(self.sequences[i, :, :L], dtype=np.float32)
        x = z_score_apply_phase_grid(x, self.mean, self.std)
        x = self._augment(x)
        return (torch.tensor(x, dtype=torch.float32), L, torch.tensor(int(self.labels[i]), dtype=torch.long), torch.tensor(self.xy[i], dtype=torch.float32))
