#!/usr/bin/env python3
"""LRU caches for trajectory CSVs and localization inference results."""

from __future__ import annotations
import csv, time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from .config import EPISODE_CSV_FLUSH_EVERY
from .traj import _list_traj_folder_csvs, _read_rssi_csv

class LocInferResultCache:

    def __init__(self, max_entries: int):
        self.max_entries = max(0, int(max_entries))
        self._data: 'OrderedDict[Tuple[str, str, bytes], float]' = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, rp: str, traj_key: str, rss: np.ndarray) -> Tuple[str, str, bytes]:
        arr = np.ascontiguousarray(rss, dtype=np.float32).ravel()
        return (str(rp), str(traj_key), arr.tobytes())

    def get(self, rp: str, traj_key: str, rss: np.ndarray) -> Optional[float]:
        if self.max_entries <= 0:
            return None
        key = self._key(rp, traj_key, rss)
        val = self._data.get(key)
        if val is None:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return float(val)

    def put(self, rp: str, traj_key: str, rss: np.ndarray, err_m: float) -> None:
        if self.max_entries <= 0:
            return
        key = self._key(rp, traj_key, rss)
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = float(err_m)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

class _EpisodeCsvAppender:

    def __init__(self, path: Path, *, flush_every: int=EPISODE_CSV_FLUSH_EVERY):
        self.path = Path(path)
        self.flush_every = max(1, int(flush_every))
        self._rows: List[List[Any]] = []
        self._file = open(self.path, 'a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)

    def writerow(self, row: List[Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self._writer.writerows(self._rows)
        self._rows.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()

class TrajCsvLRUCache:

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max(0, max_bytes))
        self._data: 'OrderedDict[str, np.ndarray]' = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, csv_path: Path) -> np.ndarray:
        key = str(csv_path.resolve())
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        arr = _read_rssi_csv(csv_path).astype(np.float32, copy=False).ravel()
        self._put(key, arr)
        self.misses += 1
        return arr

    def _put(self, key: str, arr: np.ndarray) -> None:
        b = int(arr.nbytes)
        if self.max_bytes <= 0:
            return
        while self._bytes + b > self.max_bytes and self._data:
            _, old = self._data.popitem(last=False)
            self._bytes -= int(old.nbytes)
        if key in self._data:
            self._bytes -= int(self._data[key].nbytes)
            del self._data[key]
        self._data[key] = arr
        self._bytes += b
        self._data.move_to_end(key)

class TrajFolderCatalog:

    def __init__(self, drl_root: Path, traj_index: Dict[str, Set[str]]):
        self._csv_lists: Dict[Tuple[str, str], Tuple[Path, ...]] = {}
        n_keys = sum((len(v) for v in traj_index.values()))
        t0 = time.perf_counter()
        done = 0
        for rp, keys in traj_index.items():
            for key in keys:
                folder = drl_root / rp / key
                if not folder.is_dir():
                    continue
                paths = _list_traj_folder_csvs(folder)
                if paths:
                    self._csv_lists[rp, key] = paths
                done += 1
                if done % 500 == 0:
                    print(f'[init] csv catalog {done}/{n_keys} folders ... ({time.perf_counter() - t0:.1f}s)', flush=True)
        dt = time.perf_counter() - t0
        n_csv = sum((len(v) for v in self._csv_lists.values()))
        print(f'[init] csv catalog done: {len(self._csv_lists)} folders, {n_csv} csvs in {dt:.1f}s', flush=True)

    def stats(self) -> Tuple[int, int]:
        n_csv = sum((len(v) for v in self._csv_lists.values()))
        return (len(self._csv_lists), n_csv)

    def sample_rss(self, rp: str, traj_key: str, rng: np.random.Generator, cache: Optional[TrajCsvLRUCache]) -> np.ndarray:
        paths = self._csv_lists.get((rp, traj_key))
        if not paths:
            return np.zeros(0, dtype=np.float32)
        fp = paths[int(rng.integers(0, len(paths)))]
        return self._read_csv_path(fp, cache)

    def csv_index_of_path(self, rp: str, traj_key: str, csv_path: Path) -> int:
        paths = self.list_csv_paths(rp, traj_key)
        if not paths:
            return 0
        target = str(Path(csv_path).resolve())
        for i, p in enumerate(paths):
            if str(p.resolve()) == target:
                return int(i)
        return 0

    def rss_at_index(self, rp: str, traj_key: str, csv_index: int, cache: Optional[TrajCsvLRUCache]) -> np.ndarray:
        paths = self._csv_lists.get((rp, traj_key))
        if not paths:
            return np.zeros(0, dtype=np.float32)
        idx = int(csv_index)
        if idx < 0:
            idx = 0
        elif idx >= len(paths):
            idx = len(paths) - 1
        return self._read_csv_path(paths[idx], cache)

    def _read_csv_path(self, fp: Path, cache: Optional[TrajCsvLRUCache]) -> np.ndarray:
        try:
            if cache is not None:
                return cache.get(fp)
            return _read_rssi_csv(fp).ravel().astype(np.float32, copy=False)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def list_csv_paths(self, rp: str, traj_key: str) -> Tuple[Path, ...]:
        return self._csv_lists.get((rp, traj_key), ())
