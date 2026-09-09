#!/usr/bin/env python
"""
Multi-dataset SOTER pretraining:
- One batch contains samples from exactly one dataset.
- 50% mask-style task (activates CDE path via observation_mask).
- 50% future-style task (no observation_mask, degenerates to ODE path).

This script is independent from main.py/train.sh and keeps the existing pipeline unchanged.
"""

import argparse
import contextlib
import json
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.preprocessing import MinMaxScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from soter.datasets.mira_window_dataset import TimeAwareWindowDataset
from soter.datasets.time_utils import time_aware_collate_fn
from soter.datasets.timeawared_dataset import TimeAwareJSONLDataset
from soter.models.modeling_soter import SoterConfig, SoterForPrediction


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0_print(msg: str) -> None:
    if not is_dist() or dist.get_rank() == 0:
        print(msg, flush=True)


def _print_model_param_dtype_summary(model: SoterForPrediction, prefix: str) -> None:
    """Summarize nn.Parameter dtypes (rank0 only)."""
    if is_dist() and dist.get_rank() != 0:
        return
    counts: Dict[str, int] = {}
    example: Dict[str, str] = {}
    for name, p in model.named_parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + 1
        if key not in example:
            example[key] = name
    parts = [f"{dt}: n={counts[dt]} (e.g. {example[dt]})" for dt in sorted(counts.keys())]
    rank0_print(f"{prefix} parameter dtypes: " + " | ".join(parts))


def _format_seconds(sec: float) -> str:
    sec_i = max(0, int(round(sec)))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _configure_warning_filters() -> None:
    """
    Silence known non-functional deprecation spam from transformers attention-mask utils.
    Keep other warnings visible.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
        category=FutureWarning,
        module=r"transformers\.modeling_attn_mask_utils",
    )


def setup_dist() -> Tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not is_dist():
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank


def set_seed(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _load_disk_weight_dict(save_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load raw tensors exactly as stored under save_dir (safetensors single/shard or pytorch bin).
    """
    from pathlib import Path

    save_path = Path(save_dir)
    single = save_path / "model.safetensors"
    if single.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(single)))
    index_path = save_path / "model.safetensors.index.json"
    if index_path.is_file():
        from safetensors.torch import load_file

        meta = json.loads(index_path.read_text(encoding="utf-8"))
        wm = meta.get("weight_map", {})
        merged: Dict[str, torch.Tensor] = {}
        for shard_file in sorted(set(wm.values())):
            shard = load_file(str(save_path / shard_file))
            merged.update(shard)
        return merged
    bin_path = save_path / "pytorch_model.bin"
    if bin_path.is_file():
        blob = torch.load(bin_path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "state_dict" in blob:
            return blob["state_dict"]
        return blob
    raise FileNotFoundError(
        f"[checkpoint verify] No model.safetensors / model.safetensors.index.json / pytorch_model.bin under {save_dir}"
    )


def _tensor_equal_to_disk(mem: torch.Tensor, disk: torch.Tensor) -> Tuple[bool, str]:
    """True iff `mem` matches `disk` as written (disk dtype is canonical for saved floats)."""
    m = mem.detach().cpu().contiguous()
    d = disk.detach().cpu().contiguous()
    if m.shape != d.shape:
        return False, f"shape {tuple(m.shape)} vs {tuple(d.shape)}"
    if m.dtype == d.dtype:
        return bool(torch.equal(m, d)), "same dtype but values differ"
    try:
        mq = m.to(d.dtype)
    except Exception as e:
        return False, f"cannot cast mem {m.dtype} to disk {d.dtype}: {e}"
    if torch.equal(mq, d):
        return True, ""
    if torch.is_floating_point(m) and torch.is_floating_point(d):
        max_abs = float((mq.float() - d.float()).abs().max().item())
        return False, f"after mem->disk_dtype max_abs={max_abs:.6e}"
    return False, "values differ after dtype cast"


def _verify_checkpoint_save_load(model: SoterForPrediction, save_dir: str) -> None:
    """
    Final verification logic triggered after a successful save_pretrained:
    1) Read the raw safetensors/bin weights from disk and verify they match the weights currently held in training memory.
    2) Using the safest native PyTorch architecture-reconstruction method, perform a genuine Loading pass to ensure the model can be 100% fully restored with lossless values.
    """
    # 1. Load the raw weight tensor dict from disk (the authoritative on-disk data source at this point)
    disk_sd = _load_disk_weight_dict(save_dir)
    ref_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    dk, rk = set(disk_sd.keys()), set(ref_sd.keys())
    if dk != rk:
        raise RuntimeError(
            f"[checkpoint verify] key mismatch for {save_dir}: "
            f"only_in_model={sorted(rk - dk)[:24]} only_on_disk={sorted(dk - rk)[:24]}"
        )

    # [Step 1] Verify: active in-memory weights vs raw on-disk file
    bad_save: List[Tuple[str, str]] = []
    for name in sorted(dk):
        ok, msg = _tensor_equal_to_disk(ref_sd[name], disk_sd[name])
        if not ok:
            bad_save.append((name, msg))

    if bad_save:
        lines = "\n".join(f"  {n}: {r}" for n, r in bad_save[:20])
        raise RuntimeError(
            f"[checkpoint verify] in-memory vs DISK mismatch ({len(bad_save)} tensors) for {save_dir}:\n{lines}"
        )

    # [Step 2] Real Loading process: fully bypasses any prefix/architecture-inference modifications that HF from_pretrained might apply
    if not is_dist() or dist.get_rank() == 0:
        print(f"\n[checkpoint verify] 🚀 Running real Loading verification: instantiating a fresh model skeleton and strictly loading the on-disk weights...", flush=True)

    # a. Load the saved config cleanly
    loaded_config = SoterConfig.from_pretrained(save_dir, local_files_only=True)
    # b. Randomly initialize an identical brand-new model (without any old weights)
    loaded = SoterForPrediction(loaded_config)
    
    # c. Load the on-disk tensors in native PyTorch strict mode (strict=True)
    # If any key on disk is missing, extra, or mismatched in the slightest, an exception is raised right here — silent failure is rejected
    loaded.load_state_dict(disk_sd, strict=True)
    
    try:
        # Get the model weight state recovered by the real Loading
        got_sd = {k: v.detach().cpu() for k, v in loaded.state_dict().items()}
    finally:
        # Release memory and GPU memory promptly
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(got_sd.keys()) != dk:
        raise RuntimeError(
            f"[checkpoint verify] Loaded model keys != disk keys for {save_dir}: "
            f"symdiff={sorted(set(got_sd.keys()) ^ dk)[:24]}"
        )

    # [Step 3] Verify: full tensor values recovered by the real Loading vs on-disk file values
    bad_load: List[Tuple[str, str]] = []
    for name in sorted(dk):
        ok, msg = _tensor_equal_to_disk(got_sd[name], disk_sd[name])
        if not ok:
            bad_load.append((name, msg))

    if bad_load:
        lines = "\n".join(f"  {n}: {r}" for n, r in bad_load[:20])
        raise RuntimeError(
            f"[checkpoint verify] CRITICAL: Real loaded model vs DISK mismatch ({len(bad_load)} tensors) for {save_dir}:\n{lines}"
        )

    rank0_print(
        f"[checkpoint verify] ✅ Perfect pass! Both full checks — memory->disk and real reload (Strict Loading)->disk — succeeded ({len(dk)} tensors match perfectly): {save_dir}"
    )


# def _verify_checkpoint_save_load(model: SoterForPrediction, save_dir: str) -> None:
#     """
#     After save_pretrained:
#     1) On-disk tensors (safetensors) must match in-memory state_dict when memory is cast to each disk dtype.
#        (HF often stores bf16 while training keeps fp32 master weights — strict mem-vs-load fp32 compare is a false alarm.)
#     2) from_pretrained must reproduce the same on-disk tensors (same cast rule).
#     """
#     disk_sd = _load_disk_weight_dict(save_dir)
#     ref_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

#     dk, rk = set(disk_sd.keys()), set(ref_sd.keys())
#     if dk != rk:
#         raise RuntimeError(
#             f"[checkpoint verify] key mismatch for {save_dir}: "
#             f"only_in_model={sorted(rk - dk)[:24]} only_on_disk={sorted(dk - rk)[:24]}"
#         )

#     bad_save: List[Tuple[str, str]] = []
#     for name in sorted(dk):
#         ok, msg = _tensor_equal_to_disk(ref_sd[name], disk_sd[name])
#         if not ok:
#             bad_save.append((name, msg))

#     if bad_save:
#         lines = "\n".join(f"  {n}: {r}" for n, r in bad_save[:20])
#         raise RuntimeError(
#             f"[checkpoint verify] in-memory vs DISK mismatch ({len(bad_save)} tensors) for {save_dir}:\n{lines}"
#         )

#     # loaded = SoterForPrediction.from_pretrained(save_dir, local_files_only=True)
#     loaded_config = SoterConfig.from_pretrained(save_dir, local_files_only=True)
#     loaded = SoterForPrediction(loaded_config)
#     loaded.load_state_dict(disk_sd, strict=True)
#     try:
#         got_sd = {k: v.detach().cpu() for k, v in loaded.state_dict().items()}
#     finally:
#         del loaded
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     if set(got_sd.keys()) != dk:
#         raise RuntimeError(
#             f"[checkpoint verify] loaded keys != disk keys for {save_dir}: "
#             f"symdiff={sorted(set(got_sd.keys()) ^ dk)[:24]}"
#         )

#     bad_load: List[Tuple[str, str]] = []
#     for name in sorted(dk):
#         ok, msg = _tensor_equal_to_disk(got_sd[name], disk_sd[name])
#         if not ok:
#             bad_load.append((name, msg))

#     if bad_load:
#         lines = "\n".join(f"  {n}: {r}" for n, r in bad_load[:20])
#         raise RuntimeError(
#             f"[checkpoint verify] from_pretrained vs DISK mismatch ({len(bad_load)} tensors) for {save_dir}:\n{lines}"
#         )

#     rank0_print(
#         f"[checkpoint verify] OK memory→disk and from_pretrained→disk ({len(dk)} tensors, disk is source of truth): {save_dir}"
#     )


def parse_dataset_specs(spec: str) -> OrderedDict:
    """
    spec format:
      SleepEDF=/path/train_set.jsonl,WESAD=/path/train_set.jsonl,PTB_XL=/path/train_set.jsonl,Chapman_ECG=/path/train_set.jsonl
    """
    out = OrderedDict()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid dataset spec: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise ValueError("No datasets parsed from --train_jsonls")
    return out


def create_loaders(
    dataset_paths: OrderedDict,
    batch_size: int,
    num_workers: int,
    max_length: int,
    world_size: int,
    rank: int,
    stride: int = 1,
    window_workers: int = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    loaders: Dict[str, DataLoader] = {}
    lengths: Dict[str, int] = {}
    for name, path in tqdm(dataset_paths.items(), desc="Build dataset loaders", unit="dataset", disable=(rank != 0)):
        ds = TimeAwareJSONLDataset(data_path=path, time_normalization="none")
        win_ds = TimeAwareWindowDataset(
            dataset=ds,
            context_length=max_length,
            prediction_length=0,
            time_normalizer=ds.get_time_normalizer(),
            min_valid_history=1,
            stride=stride,
            window_workers=window_workers,
        )
        lengths[name] = len(win_ds)
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(win_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(
            win_ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=time_aware_collate_fn,
            drop_last=True,
        )
        loaders[name] = loader
    return loaders, lengths


def cycle_loader(loader: DataLoader) -> Iterator[dict]:
    sampler = getattr(loader, "sampler", None)
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            epoch += 1
        for batch in loader:
            if batch:
                yield batch


def _load_jsonl_list(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_jsonl_path_arg(path_arg: Optional[str]) -> Optional[str]:
    """
    Accept both:
      - /abs/path/train_set.jsonl
      - NAME=/abs/path/train_set.jsonl
    and normalize to the real filesystem path.
    """
    if path_arg is None:
        return None
    s = str(path_arg).strip().strip('"').strip("'")
    if "=" in s:
        _, rhs = s.split("=", 1)
        s = rhs.strip().strip('"').strip("'")
    return s


def _fit_minmax_per_channel_from_train_jsonl(train_jsonl: str) -> List[MinMaxScaler]:
    records = _load_jsonl_list(train_jsonl)
    if not records:
        raise ValueError(f"Empty MIT-BIH train jsonl: {train_jsonl}")

    n_ch = None
    for rec in records:
        seq = np.asarray(rec.get("sequence"), dtype=np.float32)
        if seq.ndim == 1 and seq.size:
            n_ch = 1
            break
        if seq.ndim == 2 and seq.shape[1] > 0:
            n_ch = int(seq.shape[1])
            break
    if n_ch is None:
        raise ValueError("Cannot infer channel count from MIT-BIH train jsonl.")

    vals_by_ch: List[List[np.ndarray]] = [[] for _ in range(n_ch)]
    for rec in records:
        seq = np.asarray(rec.get("sequence"), dtype=np.float32)
        mask = rec.get("mask", None)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        if mask is None:
            m = np.ones_like(seq, dtype=np.int32)
        else:
            m = np.asarray(mask, dtype=np.int32)
            if m.ndim == 1:
                m = np.repeat(m.reshape(-1, 1), seq.shape[1], axis=1)
        if seq.shape[1] != n_ch:
            continue
        for c in range(n_ch):
            valid = m[:, c] == 1
            if np.any(valid):
                vals_by_ch[c].append(seq[:, c][valid].astype(np.float64).reshape(-1, 1))

    scalers: List[MinMaxScaler] = []
    for c in range(n_ch):
        sc = MinMaxScaler()
        if vals_by_ch[c]:
            sc.fit(np.vstack(vals_by_ch[c]))
        else:
            sc.fit(np.asarray([[0.0], [1.0]], dtype=np.float64))
        scalers.append(sc)
    return scalers


def _ensure_strictly_increasing_np(times: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    t = times.astype(np.float32).copy()
    for i in range(1, t.shape[0]):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


@torch.inference_mode()
def run_mitbih_quick_eval(
    model: SoterForPrediction,
    mitbih_test_jsonl: str,
    scalers: List[MinMaxScaler],
    n_sequences: int,
    seed: int,
    ctx: int,
    pred: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float, int]:
    records = _load_jsonl_list(mitbih_test_jsonl)
    if not records:
        return float("nan"), float("nan"), 0

    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_sequences, len(idxs))]

    _print_model_param_dtype_summary(model, prefix=f"[MITBIH quick eval] seed={seed}")

    per_ch_sq: Optional[List[List[float]]] = None
    per_ch_abs: Optional[List[List[float]]] = None
    n_pts = 0
    _fwd_dtype_logged = False

    eval_iterator = idxs
    if not is_dist() or dist.get_rank() == 0:
        eval_iterator = tqdm(idxs, desc="MITBIH quick eval", unit="seq", leave=False)
    for i in eval_iterator:
        rec = records[i]
        seq = np.asarray(rec["sequence"], dtype=np.float32)
        t = np.asarray(rec["time"], dtype=np.float32)
        m = np.asarray(rec.get("mask", np.ones_like(seq)), dtype=np.int32)

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        if m.ndim == 1:
            m = np.repeat(m.reshape(-1, 1), seq.shape[1], axis=1)
        if seq.shape[0] < ctx + pred:
            continue

        seq = seq[: ctx + pred]
        t = _ensure_strictly_increasing_np(t[: ctx + pred])
        m = m[: ctx + pred]
        n_ch = int(seq.shape[1])
        if per_ch_sq is None:
            per_ch_sq = [[] for _ in range(n_ch)]
            per_ch_abs = [[] for _ in range(n_ch)]

        for c in range(n_ch):
            sc = scalers[c] if c < len(scalers) else scalers[0]
            seq_norm = sc.transform(seq[:, c].astype(np.float64).reshape(-1, 1)).reshape(-1).astype(np.float32)
            preds = np.zeros((pred,), dtype=np.float32)
            for step in range(pred):
                hist_vals = torch.from_numpy(seq_norm[: ctx + step]).float().view(1, -1, 1).to(device)
                hist_times = torch.from_numpy(t[: ctx + step]).float().view(1, -1).to(device)
                next_t_val = float(t[ctx + step])
                last_t_val = float(hist_times[0, -1].item()) if hist_times.shape[1] > 0 else next_t_val
                if next_t_val <= last_t_val:
                    next_t_val = last_t_val + 1e-4
                next_t = torch.tensor([next_t_val], device=device, dtype=torch.float32)
                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
                    if amp_enabled
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    if not _fwd_dtype_logged:
                        if not is_dist() or dist.get_rank() == 0:
                            rank0_print(
                                "[MITBIH forward dtypes] first forward — "
                                f"torch.is_autocast_enabled('cuda')={torch.is_autocast_enabled(device_type='cuda')} "
                                f"autocast target dtype={amp_dtype} | "
                                f"input_ids={hist_vals.dtype} time_values={hist_times.dtype} "
                                f"next_target_time_values={next_t.dtype} "
                                f"attention_mask_dtype=int64 (long)"
                            )
                    out = model(
                        input_ids=hist_vals,
                        time_values=hist_times,
                        next_target_time_values=next_t,
                        attention_mask=torch.ones(hist_vals.shape[0], hist_vals.shape[1], dtype=torch.long, device=device),
                        return_dict=True,
                    )
                    if not _fwd_dtype_logged:
                        if not is_dist() or dist.get_rank() == 0:
                            rank0_print(
                                "[MITBIH forward dtypes] first forward — "
                                f"out.logits dtype={out.logits.dtype} device={out.logits.device}"
                            )
                        _fwd_dtype_logged = True
                preds[step] = float(out.logits[0, -1, 0].detach().cpu())

            gt = seq_norm[ctx : ctx + pred]
            valid = (m[ctx : ctx + pred, c] == 1)
            if not np.any(valid):
                continue
            err = preds[valid] - gt[valid]
            per_ch_sq[c].extend((err * err).tolist())
            per_ch_abs[c].extend(np.abs(err).tolist())
            n_pts += int(np.sum(valid))

    if per_ch_sq is None or n_pts == 0:
        return float("nan"), float("nan"), 0

    ch_rmse = [float(math.sqrt(float(np.mean(sq)))) for sq in per_ch_sq if sq]
    ch_mae = [float(np.mean(ab)) for ab in per_ch_abs if ab]
    rmse_macro = float(np.mean(ch_rmse)) if ch_rmse else float("nan")
    mae_macro = float(np.mean(ch_mae)) if ch_mae else float("nan")
    return rmse_macro, mae_macro, n_pts


def dataset_choice(
    names: List[str],
    lengths: Dict[str, int],
    step: int,
    mode: str,
    seed: int,
) -> str:
    rnd = random.Random(seed + step)
    if mode == "uniform":
        return names[rnd.randrange(len(names))]
    if mode == "proportional":
        total = sum(max(1, lengths[n]) for n in names)
        r = rnd.random() * total
        cur = 0.0
        for n in names:
            cur += max(1, lengths[n])
            if r <= cur:
                return n
        return names[-1]
    raise ValueError(f"Unknown dataset_sampling={mode}")


def build_observation_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Build observation mask [B, L, C] for mask-style task.
    1 = observed, 0 = missing.
    """
    b, l, c = input_ids.shape
    base = attention_mask.unsqueeze(-1).repeat(1, 1, c).float()  # valid padded steps -> 1
    if mask_ratio <= 0:
        return base
    rand = torch.rand_like(base)
    drop = (rand < mask_ratio).float() * base
    obs = base - drop
    # keep at least one observed step per (B,C) to avoid empty spline
    for bi in range(b):
        for ci in range(c):
            if torch.sum(obs[bi, :, ci]) < 1:
                valid_idx = torch.where(base[bi, :, ci] > 0)[0]
                if len(valid_idx) > 0:
                    obs[bi, valid_idx[0], ci] = 1.0
    return obs


def main():
    _configure_warning_filters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonls", type=str, required=True, help="name=path,name=path,...")
    parser.add_argument("--model_path", type=str, default="Maple728/TimeMoE-50M")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_sampling", type=str, choices=["uniform", "proportional"], default="proportional")
    parser.add_argument("--mask_task_prob", type=float, default=0.5, help="target proportion of mask-style task")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="mask ratio inside mask-style task")
    parser.add_argument("--cde_control_dim", type=int, default=1, help=">0 to enable CDE control path")
    parser.add_argument("--attn_implementation", type=str, choices=["eager", "flash_attention_2", "auto"], default="eager")
    parser.add_argument("--dft_router_impl", type=str, choices=["for_loop", "prefix_parallel"], default="for_loop")
    parser.add_argument("--time_aware_rotary", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ode_solver_method", type=str, choices=["dopri5", "rk4"], default="dopri5")
    parser.add_argument("--ode_solver_atol", type=float, default=1e-6)
    parser.add_argument("--ode_solver_rtol", type=float, default=1e-6)
    parser.add_argument("--mitbih_eval_jsonl", type=str, default=None, help="MIT-BIH OOD test jsonl for periodic quick eval.")
    parser.add_argument("--mitbih_train_jsonl", type=str, default=None, help="MIT-BIH OOD train jsonl to fit per-channel MinMax.")
    parser.add_argument("--mitbih_eval_every", type=int, default=1000, help="Run MIT-BIH quick eval every N steps.")
    parser.add_argument("--mitbih_eval_n", type=int, default=32, help="Number of sampled MIT-BIH sequences per quick eval.")
    parser.add_argument("--mitbih_eval_seed", type=int, default=123, help="Random seed for MIT-BIH quick eval sampling.")
    parser.add_argument("--mitbih_ctx", type=int, default=128, help="Context length for MIT-BIH quick eval.")
    parser.add_argument("--mitbih_pred", type=int, default=64, help="Prediction length for MIT-BIH quick eval.")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable tqdm progress bars on rank0.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for training window generation.")
    parser.add_argument("--window_workers", type=int, default=None, help="Parallel workers for window discovery (default: auto).")
    parser.add_argument(
        "--skip_checkpoint_verify",
        action="store_true",
        help="Skip save→from_pretrained→state_dict equality check after each checkpoint (saves time/IO).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    world_size, rank, local_rank = setup_dist()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)

    dataset_paths = parse_dataset_specs(args.train_jsonls)
    rank0_print("Datasets:")
    for k, v in dataset_paths.items():
        rank0_print(f"  - {k}: {v}")

    loaders, lengths = create_loaders(
        dataset_paths=dataset_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        world_size=world_size,
        rank=rank,
        stride=args.stride,
        window_workers=args.window_workers,
    )
    # names = list(loaders.keys())
    # iters = {k: cycle_loader(v) for k, v in loaders.items()}
    # rank0_print(f"Window counts: {lengths}")
    
    names = list(loaders.keys())
    rank0_print(f"Window counts: {lengths}")
    rank0_print("\n[*] Global Training Epoch Building.")
    iters = {}
    for name, loader in loaders.items():
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(0)
        iters[name] = iter(loader)
    
    global_schedule = []
    for name, loader in loaders.items():
        global_schedule.extend([name] * len(loader))
    
    total_steps_in_epoch = len(global_schedule)
    rank0_print(f"[*] Total: {total_steps_in_epoch} Steps.")

    rng = random.Random(args.seed)
    rng.shuffle(global_schedule)

    attn = args.attn_implementation
    if attn == "auto":
        attn = "eager"

    if args.from_scratch:
        cfg = SoterConfig.from_pretrained(
            args.model_path,
            _attn_implementation=attn,
            apply_aux_loss=False,
            cde_control_dim=args.cde_control_dim,
            time_aware_rotary=args.time_aware_rotary,
            dft_router_impl=args.dft_router_impl,
            ode_solver_method=args.ode_solver_method,
            ode_solver_atol=args.ode_solver_atol,
            ode_solver_rtol=args.ode_solver_rtol,
        )
        model = SoterForPrediction(cfg)
    else:
        model = SoterForPrediction.from_pretrained(
            args.model_path,
            _attn_implementation=attn,
            apply_aux_loss=False,
            dft_router_impl=args.dft_router_impl,
            ode_solver_method=args.ode_solver_method,
            ode_solver_atol=args.ode_solver_atol,
            ode_solver_rtol=args.ode_solver_rtol,
        )
        model.config.model_type = "soter"
        model.config.cde_control_dim = args.cde_control_dim
        model.config.time_aware_rotary = args.time_aware_rotary
        model.config.dft_router_impl = args.dft_router_impl
        model.config.ode_solver_method = args.ode_solver_method
        model.config.ode_solver_atol = args.ode_solver_atol
        model.config.ode_solver_rtol = args.ode_solver_rtol

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps_in_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16"))

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    model.train()
    train_start_ts = time.time()
    mitbih_scalers: Optional[List[MinMaxScaler]] = None
    if args.mitbih_eval_jsonl:
        if not args.mitbih_train_jsonl:
            raise ValueError("--mitbih_train_jsonl is required when --mitbih_eval_jsonl is set.")
        args.mitbih_train_jsonl = _normalize_jsonl_path_arg(args.mitbih_train_jsonl)
        args.mitbih_eval_jsonl = _normalize_jsonl_path_arg(args.mitbih_eval_jsonl)
        mitbih_scalers = _fit_minmax_per_channel_from_train_jsonl(args.mitbih_train_jsonl)
    # step_iter = range(1, args.max_steps + 1)
    # if args.enable_tqdm and (not is_dist() or dist.get_rank() == 0):
    #     step_iter = tqdm(step_iter, total=args.max_steps, desc="Training steps", unit="step")
    if args.enable_tqdm and (not is_dist() or dist.get_rank() == 0):
        step_iter = tqdm(enumerate(global_schedule, 1), total=total_steps_in_epoch, desc="Epoch 1 Training", unit="step")
    else:
        step_iter = enumerate(global_schedule, 1)

    # for step in step_iter:
    #     ds_name = dataset_choice(names, lengths, step, args.dataset_sampling, args.seed)
    #     batch = next(iters[ds_name])
    for step, ds_name in step_iter:
        batch = next(iters[ds_name])
        input_ids = batch["input_ids"].to(device)
        time_values = batch["time_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss_masks = batch["loss_masks"].to(device)
        next_t = batch["next_target_time_values"].to(device)

        is_mask_task = (random.Random(args.seed * 100000 + step).random() < args.mask_task_prob)
        observation_mask = None
        task_name = "future"
        if is_mask_task:
            task_name = "mask"
            observation_mask = build_observation_mask(input_ids, attention_mask, args.mask_ratio).to(device)
            input_ids = input_ids * observation_mask

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.precision in ["bf16", "fp16"]), dtype=amp_dtype):
            out = model(
                input_ids=input_ids,
                time_values=time_values,
                attention_mask=attention_mask,
                labels=labels,
                loss_masks=loss_masks,
                next_target_time_values=next_t,
                observation_mask=observation_mask,
                return_dict=True,
            )
            loss = out.loss

        if args.precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        if step % args.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            now_ts = time.time()
            elapsed_s = max(0.0, now_ts - train_start_ts)
            avg_step_s = elapsed_s / float(max(1, step))
            # remain_steps = max(0, args.max_steps - step)
            # eta_s = remain_steps * avg_step_s
            # eta_done_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts + eta_s))
            # rank0_print(
            #     f"step={step}/{args.max_steps} | ds={ds_name} | task={task_name} | "
            remain_steps = max(0, total_steps_in_epoch - step)
            eta_s = remain_steps * avg_step_s
            eta_done_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts + eta_s))
            rank0_print(
                f"step={step}/{total_steps_in_epoch} | ds={ds_name} | task={task_name} | "
                f"loss={float(loss.detach().cpu()):.6f} | lr={lr:.6e} | "
                f"elapsed={_format_seconds(elapsed_s)} | eta={_format_seconds(eta_s)} | "
                f"avg_step={avg_step_s:.3f}s | eta_at={eta_done_str}"
            )
            if args.enable_tqdm and (not is_dist() or dist.get_rank() == 0):
                try:
                    step_iter.set_postfix({"ds": ds_name, "task": task_name, "loss": f"{float(loss.detach().cpu()):.4f}"})
                except Exception:
                    pass

        should_run_mitbih_eval = (
            args.mitbih_eval_jsonl
            and args.mitbih_eval_every > 0
            and step % args.mitbih_eval_every == 0
            and mitbih_scalers is not None
        )
        if should_run_mitbih_eval:
            # IMPORTANT:
            # Run quick eval on all ranks (same workload), but only rank0 prints.
            # This avoids rank0-only long eval causing NCCL watchdog timeouts on other ranks.
            model_to_eval = model.module if isinstance(model, DDP) else model
            model_to_eval.eval()
            rmse_macro, mae_macro, n_pts = run_mitbih_quick_eval(
                model=model_to_eval,
                mitbih_test_jsonl=args.mitbih_eval_jsonl,
                scalers=mitbih_scalers,
                n_sequences=args.mitbih_eval_n,
                seed=args.mitbih_eval_seed + step,
                ctx=args.mitbih_ctx,
                pred=args.mitbih_pred,
                device=device,
                amp_enabled=(device.type == "cuda" and args.precision in ["bf16", "fp16"]),
                amp_dtype=amp_dtype,
            )
            if not is_dist() or dist.get_rank() == 0:
                rank0_print(
                    f"[MITBIH quick eval @step={step}] C={args.mitbih_ctx} P={args.mitbih_pred} n_pts={n_pts} "
                    f"RMSE_norm_macro={rmse_macro:.6f} MAE_norm_macro={mae_macro:.6f}"
                )
            model_to_eval.train()
        # if step % args.save_steps == 0 or step == args.max_steps:
        if step % args.save_steps == 0 or step == total_steps_in_epoch:
            save_dir = os.path.join(args.output_path, f"checkpoint-{step}")
            if not is_dist() or dist.get_rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
                model_to_save = model.module if isinstance(model, DDP) else model
                model_to_save.save_pretrained(save_dir)
                rank0_print(f"Saved checkpoint: {save_dir}")
            # All ranks wait for rank-0 checkpoint IO before verify or next step.
            if is_dist():
                dist.barrier()
            if not args.skip_checkpoint_verify:
                if not is_dist() or dist.get_rank() == 0:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    _verify_checkpoint_save_load(model_to_save, save_dir)
                if is_dist():
                    dist.barrier()

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
    rank0_print("Training finished.")


if __name__ == "__main__":
    main()

