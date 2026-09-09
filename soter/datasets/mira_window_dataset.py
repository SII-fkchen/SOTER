#!/usr/bin/env python
# -*- coding:utf-8 _*-
import hashlib
import json
import multiprocessing as mp
import os
import random
import tempfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.utils import logging

from soter.datasets.ts_dataset import TimeSeriesDataset


logger = logging.get_logger(__name__)


def _safe_json_loads(line: str):
    return json.loads(
        line,
        parse_constant=lambda s: float("nan")
        if s == "NaN"
        else (float("inf") if s == "Infinity" else float("-inf")),
    )


def _build_window_chunk(args):
    """Worker for parallel valid-window discovery."""
    (
        data_path,
        offset_path,
        seq_start,
        seq_end,
        raw_window_size,
        min_valid_history,
        stride,
    ) = args

    offsets = np.load(offset_path, allow_pickle=False).astype(np.int64)
    valid_windows = []

    with open(data_path, "r", encoding="utf-8") as f:
        for seq_idx in range(seq_start, seq_end):
            f.seek(int(offsets[seq_idx]))
            raw = f.readline().strip()
            if not raw:
                continue
            try:
                record = _safe_json_loads(raw)
            except Exception:
                continue

            mask = record.get("mask", None)
            if mask is None:
                seq = record.get("sequence", [])
                seq_len = len(seq) if hasattr(seq, "__len__") else 0
                time_mask_1d = np.ones(seq_len, dtype=np.int8)
            else:
                mask_arr = np.asarray(mask, dtype=np.int8)
                if mask_arr.ndim > 1:
                    time_mask_1d = (mask_arr == 1).any(axis=-1).astype(np.int8)
                else:
                    time_mask_1d = mask_arr

            seq_len = len(time_mask_1d)
            if seq_len < raw_window_size:
                continue

            if stride == 1:
                view = sliding_window_view(time_mask_1d, raw_window_size)
                valid_starts = np.where(view.sum(axis=1) >= min_valid_history)[0]
            else:
                view = sliding_window_view(time_mask_1d, raw_window_size)
                valid_mask = view.sum(axis=1) >= min_valid_history
                max_start = seq_len - raw_window_size
                if max_start < 0:
                    continue
                candidate_starts = np.arange(0, max_start + 1, stride)
                valid_starts = candidate_starts[valid_mask[candidate_starts]]

            valid_windows.extend(
                [(int(seq_idx), int(s)) for s in valid_starts]
            )

    return valid_windows


class TimeAwareWindowDataset(Dataset):
    """
    Generates training windows from a TimeSeriesDataset (like TimeAwareJSONLDataset).
    Applies masking to remove invalid points and normalizes time.
    Outputs data SUITABLE FOR AUTOREGRESSIVE TRAINING with absolute time.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        time_normalizer=None,
        min_valid_history: int = 1,
        stride: int = 1,
        window_workers: int = None,
    ):
        self.source_dataset = dataset
        self.context_length = context_length
        self.time_normalizer = time_normalizer
        self.min_valid_history = min_valid_history
        self.stride = max(1, int(stride))
        self.raw_window_size = context_length + 1
        self.valid_windows = []

        # --- Precompute valid windows (self.min_valid_history), with cache ---
        cache_tag = getattr(self.source_dataset, "cache_tag", f"id:{id(self.source_dataset)}")
        key = (
            f"{cache_tag}|ctx={self.context_length}|min_valid={self.min_valid_history}"
            f"|raw={self.raw_window_size}|stride={self.stride}"
        )
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        base_path = getattr(self.source_dataset, "data_path", None)
        cache_dir = os.path.dirname(base_path) if base_path else "."
        self.window_cache_path = os.path.join(cache_dir, f".valid_windows_{key_hash}.npz")

        if os.path.exists(self.window_cache_path):
            try:
                cache = np.load(self.window_cache_path, allow_pickle=False)
                idx = cache["windows"].astype(np.int32)
                if idx.ndim == 2 and idx.shape[1] == 2 and idx.shape[0] > 0:
                    self.valid_windows = [tuple(x) for x in idx.tolist()]
                    logger.info(
                        f"Loaded valid windows cache: {self.window_cache_path} "
                        f"({len(self.valid_windows)} windows)"
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed loading valid windows cache, rebuilding. err={e}")

        logger.info("Precomputing valid windows for Autoregressive training...")
        num_sequences = len(self.source_dataset)

        # Fast-path: bypass heavy __getitem__ preprocessing during window discovery.
        _fast_get = None
        if hasattr(self.source_dataset, "_get_record_by_index"):
            _fast_get = self.source_dataset._get_record_by_index

        # Decide whether we can use multiprocessing.
        _can_parallel = (
            _fast_get is not None
            and hasattr(self.source_dataset, "data_path")
            and hasattr(self.source_dataset, "index_cache_path")
            and os.path.exists(self.source_dataset.index_cache_path)
            and num_sequences > 50000
        )

        if _can_parallel:
            n_workers = window_workers or min(32, mp.cpu_count())
            n_workers = max(1, int(n_workers))
            chunk_size = max(1, num_sequences // n_workers)
            chunks = []
            for i in range(0, num_sequences, chunk_size):
                end = min(i + chunk_size, num_sequences)
                chunks.append(
                    (
                        self.source_dataset.data_path,
                        self.source_dataset.index_cache_path,
                        i,
                        end,
                        self.raw_window_size,
                        self.min_valid_history,
                        self.stride,
                    )
                )

            logger.info(
                f"Parallel window discovery: {n_workers} workers, "
                f"{len(chunks)} chunks, ~{chunk_size} seqs/chunk"
            )
            with mp.Pool(n_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_build_window_chunk, chunks),
                        total=len(chunks),
                        desc="Finding Valid AR Windows (parallel)",
                    )
                )
            for r in results:
                self.valid_windows.extend(r)
        else:
            iterator = (
                tqdm(range(num_sequences), total=num_sequences, desc="Finding Valid AR Windows")
                if "tqdm" in globals()
                else range(num_sequences)
            )
            for seq_idx in iterator:
                try:
                    if _fast_get is not None:
                        raw = _fast_get(seq_idx)
                        mask = raw.get("mask", None)
                        if mask is None:
                            seq = raw.get("sequence", [])
                            seq_len = len(seq) if hasattr(seq, "__len__") else 0
                            time_mask_1d = np.ones(seq_len, dtype=np.int8)
                        else:
                            mask_arr = np.asarray(mask, dtype=np.int8)
                            if mask_arr.ndim > 1:
                                time_mask_1d = (mask_arr == 1).any(axis=-1).astype(np.int8)
                            else:
                                time_mask_1d = mask_arr
                    else:
                        item = self.source_dataset[seq_idx]
                        mask = item.get("mask", None)
                        if mask is None:
                            mask = np.ones_like(item["values"], dtype=np.int8)
                        if mask.ndim > 1:
                            time_mask_1d = (mask == 1).any(axis=-1).astype(np.int8)
                        else:
                            time_mask_1d = mask

                    seq_len = len(time_mask_1d)
                    if seq_len < self.raw_window_size:
                        continue

                    if self.stride == 1:
                        view = sliding_window_view(time_mask_1d, self.raw_window_size)
                        valid_starts = np.where(view.sum(axis=1) >= self.min_valid_history)[0]
                    else:
                        view = sliding_window_view(time_mask_1d, self.raw_window_size)
                        valid_mask = view.sum(axis=1) >= self.min_valid_history
                        max_start = seq_len - self.raw_window_size
                        if max_start < 0:
                            continue
                        candidate_starts = np.arange(0, max_start + 1, self.stride)
                        valid_starts = candidate_starts[valid_mask[candidate_starts]]

                    self.valid_windows.extend(
                        zip(
                            np.full(len(valid_starts), seq_idx, dtype=np.int32),
                            valid_starts.astype(np.int32),
                        )
                    )
                except Exception as e:
                    logger.exception(f"seq_idx {seq_idx}: {e}")

        if not self.valid_windows:
            raise ValueError(
                f"No valid AR windows found with context_length={context_length}, "
                f"and min_valid_history={self.min_valid_history}. "
                f"Check sequence lengths ({num_sequences} sequences processed) and data validity (masks)."
            )
        logger.info(f"Found {len(self.valid_windows)} potential valid AR windows.")
        try:
            arr = np.asarray(self.valid_windows, dtype=np.int32)
            os.makedirs(os.path.dirname(self.window_cache_path) or ".", exist_ok=True)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".npz", dir=os.path.dirname(self.window_cache_path) or "."
            ) as tf:
                tmp = tf.name
            try:
                np.savez_compressed(tmp, windows=arr)
                os.replace(tmp, self.window_cache_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            logger.info(f"Saved valid windows cache: {self.window_cache_path}")
        except Exception as e:
            logger.warning(f"Failed saving valid windows cache: {e}")

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_windows[idx]

        try:
            item = self.source_dataset[seq_idx]
            if not isinstance(item, dict) or not all(k in item for k in ["sequence", "time", "mask"]):
                if isinstance(item, (np.ndarray, list)):
                    raw_sequence_full = np.array(item, dtype=np.float32)
                    raw_time_full = np.arange(len(raw_sequence_full), dtype=np.float32)
                    raw_mask_full = np.ones(len(raw_sequence_full), dtype=np.int32)
                    end_idx = start_idx + self.raw_window_size
                    if end_idx > len(raw_sequence_full):
                        raise IndexError(
                            f"Calculated end_idx {end_idx} exceeds generated sequence length {len(raw_sequence_full)}"
                        )
                    raw_sequence_window = raw_sequence_full[start_idx:end_idx]
                    raw_time_window = raw_time_full[start_idx:end_idx]
                    raw_mask_window = raw_mask_full[start_idx:end_idx]
                else:
                    raise TypeError(
                        f"Item from source_dataset at index {seq_idx} is not a dict or sequence array."
                    )
            else:
                end_idx = start_idx + self.raw_window_size
                raw_sequence_window = item["sequence"][start_idx:end_idx]
                raw_time_window = item["time"][start_idx:end_idx]
                raw_mask_window = item["mask"][start_idx:end_idx]

        except IndexError as e:
            logger.error(
                f"IndexError accessing data for window idx {idx} (seq_idx={seq_idx}, start_idx={start_idx}): {e}"
            )
            raise e
        except Exception as e:
            logger.error(
                f"Unexpected error accessing data for window idx {idx} (seq_idx={seq_idx}, start_idx={start_idx}): {e}"
            )
            raise e

        time_step_mask = (
            (raw_mask_window == 1).any(axis=-1)
            if raw_mask_window.ndim > 1
            else (raw_mask_window == 1)
        )
        valid_sequence = raw_sequence_window[time_step_mask]
        valid_time_abs = raw_time_window[time_step_mask]

        if len(valid_sequence) < 2:
            logger.debug(
                f"Skipping window {idx} (seq={seq_idx}, start={start_idx}) due to insufficient valid points ({len(valid_sequence)} < 2)."
            )
            return None

        valid_time_norm = valid_time_abs
        if self.time_normalizer is not None:
            try:
                if len(valid_time_abs) > 0:
                    valid_time_norm = (
                        self.time_normalizer.transform(valid_time_abs.reshape(-1, 1)).flatten()
                    )
            except Exception as e:
                logger.error(
                    f"Error applying time normalizer for window idx {idx}: {e}. Using original times."
                )
                valid_time_norm = valid_time_abs

        input_ids = valid_sequence[:-1].astype(np.float32)
        time_values = valid_time_norm[:-1].astype(np.float32)
        labels = np.array([valid_sequence[-1].astype(np.float32)])

        current_length = len(input_ids)
        attention_mask = np.ones(current_length, dtype=np.int32)
        loss_mask = np.ones(len(labels), dtype=np.int32)

        next_target_time_value = valid_time_norm[-1]
        next_target_time_value = np.float32(next_target_time_value)

        # Filter out windows with abnormal prediction time gaps (sensor disconnection artifacts).
        if current_length >= 2:
            diffs = np.diff(time_values)
            median_dt = np.median(diffs) if len(diffs) > 0 else 1.0
            median_dt = max(float(median_dt), 1e-8)
            delta_t = float(next_target_time_value) - float(time_values[-1])
            if delta_t > 5.0 * median_dt:
                return None

        return {
            "input_ids": input_ids,
            "time_values": time_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_masks": loss_mask,
            "next_target_time_values": next_target_time_value,
        }


# Code copied from time_moe.datasets.time_moe_window_dataset
class MIRAWindowDataset:
    """
    A dataset that generates windows of time series data.
    """

    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, **kwrags):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            if n_points < 2:
                continue
            for offset_idx in range(0, n_points, self.window_size):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.dataset[seq_i][offset_i : offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), "constant", constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), "constant", constant_values=0)

        return {"input_ids": seq[:-1], "labels": seq[1:], "loss_masks": loss_mask}


# Code copied from time_moe.datasets.time_moe_window_dataset
class UniversalMIRAWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq : start_idx_in_seq + offset]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {"input_ids": seq[:-1], "labels": seq[1:]}
