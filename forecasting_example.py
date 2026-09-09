#!/usr/bin/env python
"""
Zero-shot forecasting example for SOTER.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_state_dict(ckpt_dir: str) -> dict:
    """Read raw weights from a checkpoint directory (safetensors or pytorch bin)."""
    from safetensors.torch import load_file

    path = Path(ckpt_dir)
    single = path / "model.safetensors"
    if single.is_file():
        return dict(load_file(str(single)))
    index = path / "model.safetensors.index.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8"))
        merged = {}
        for shard in sorted(set(meta["weight_map"].values())):
            merged.update(load_file(str(path / shard)))
        return merged
    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        blob = torch.load(bin_path, map_location="cpu", weights_only=True)
        return blob.get("state_dict", blob)
    raise FileNotFoundError(f"No model weights found in {ckpt_dir}")


def load_model(model_path: str) -> Tuple[torch.nn.Module, torch.device]:
    from soter.models.modeling_soter import SoterConfig, SoterForPrediction

    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors*"])

    config = SoterConfig.from_pretrained(model_path)
    model = SoterForPrediction(config)
    model.load_state_dict(_load_state_dict(model_path), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] inference device: {device}")
    model = model.to(device)
    model.eval()
    return model, device



def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_strictly_increasing(times: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    t = times.astype(np.float32).copy()
    for i in range(1, t.shape[0]):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def as_multichannel(rec: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = np.asarray(rec["sequence"], dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    t = np.asarray(rec["time"], dtype=np.float32)
    m = rec.get("mask")
    if m is None:
        m = np.ones_like(seq, dtype=np.int64)
    else:
        m = np.asarray(m, dtype=np.int64)
        if m.ndim == 1:
            m = np.repeat(m.reshape(-1, 1), seq.shape[1], axis=1)
    return seq, t, m


def infer_max_channel_count(records: List[dict]) -> int:
    n_ch = 0
    for rec in records:
        seq = np.asarray(rec.get("sequence"), dtype=np.float32)
        if seq.ndim == 1 and seq.size:
            n_ch = max(n_ch, 1)
        elif seq.ndim == 2 and seq.shape[1] > 0:
            n_ch = max(n_ch, int(seq.shape[1]))
    if n_ch <= 0:
        raise ValueError("Cannot infer channel count from JSONL records.")
    return n_ch


def fit_minmax_scalers(records: List[dict]):
    """Fit one MinMaxScaler per channel on the training split"""
    from sklearn.preprocessing import MinMaxScaler

    num_channels = infer_max_channel_count(records)
    vals: List[List[np.ndarray]] = [[] for _ in range(num_channels)]
    for rec in records:
        seq, _, m = as_multichannel(rec)
        if seq.shape[1] != num_channels:
            continue
        for c in range(num_channels):
            valid = m[:, c] == 1
            if np.any(valid):
                vals[c].append(seq[valid, c].astype(np.float64).reshape(-1, 1))

    scalers = []
    for c in range(num_channels):
        sc = MinMaxScaler()
        sc.fit(np.vstack(vals[c]) if vals[c] else np.asarray([[0.0], [1.0]]))
        scalers.append(sc)
    return scalers



# Autoregressive forecasting

@torch.inference_mode()
def forecast_channel(
    model: torch.nn.Module,
    history: np.ndarray,
    times: np.ndarray,
    context: int,
    horizon: int,
    device: torch.device,
    amp_dtype: torch.dtype = None,
) -> np.ndarray:
    """
    Autoregressive rollout over the horizon.
    history: [context + horizon] normalized values (only the first `context`
             values are shown to the model; its own predictions are fed back)
    times:   [context + horizon] raw timestamps (strictly increasing)
    returns: [horizon] predictions in normalized space
    """
    preds = np.zeros((horizon,), dtype=np.float32)

    cur_vals = torch.from_numpy(history[:context]).float().view(1, -1, 1).to(device)  # [1, L, 1]
    cur_times = torch.from_numpy(times[:context]).float().view(1, -1).to(device)      # [1, L]

    for step in range(horizon):
        # Target timestamp for this forecasting step (must exceed the last one).
        next_t_val = float(times[context + step])
        last_t_val = float(cur_times[0, -1].item())
        if next_t_val <= last_t_val:
            next_t_val = last_t_val + 1e-4
        next_t = torch.tensor([next_t_val], device=device, dtype=torch.float32)

        with torch.amp.autocast(device_type="cuda", enabled=amp_dtype is not None,
                                dtype=amp_dtype if amp_dtype is not None else torch.bfloat16):
            out = model(
                input_ids=cur_vals,
                time_values=cur_times,
                next_target_time_values=next_t,  # ODE integrates from t_last to this time
                attention_mask=torch.ones(cur_vals.shape[0], cur_vals.shape[1], dtype=torch.long, device=device),
                return_dict=True,
            )
        next_val = out.logits[0, -1, 0].detach().float()  # stays on `device`
        preds[step] = float(next_val.cpu())

        # Feed the prediction back for the next step (autoregressive rollout).
        cur_vals = torch.cat([cur_vals, next_val.view(1, 1, 1).to(cur_vals.dtype)], dim=1)
        cur_times = torch.cat(
            [cur_times, torch.tensor([[next_t_val]], device=device)], dim=1
        )
    return preds


def run_evaluation(args, model, device, amp_dtype) -> None:
    from tqdm import tqdm

    records = load_jsonl(args.data)

    if args.train_jsonl:
        print(f"[info] fitting per-channel MinMaxScaler on {args.train_jsonl}")
        scalers = fit_minmax_scalers(load_jsonl(args.train_jsonl))
        n_ch_expected = len(scalers)
    else:
        print("[info] no --train_jsonl given; falling back to per-window z-score normalization")
        scalers = None
        n_ch_expected = infer_max_channel_count(records)
    print(f"[info] channel count: {n_ch_expected}")

    idxs = list(range(len(records)))
    if str(args.num_eval).lower() == "all":
        pass  # full-file inference, in file order
    else:
        rng = random.Random(args.seed)
        rng.shuffle(idxs)
        idxs = idxs[: min(int(args.num_eval), len(idxs))]
    print(f"[info] evaluating {len(idxs)} sequence(s) from {args.data}")

    ctx, horizon = args.context, args.horizon
    sample_rmses: List[float] = []
    sample_maes: List[float] = []

    for i in tqdm(idxs, desc="evaluating", unit="seq"):
        seq, t, m = as_multichannel(records[i])
        if seq.shape[0] < ctx + horizon:
            continue
        seq, t, m = seq[: ctx + horizon], ensure_strictly_increasing(t[: ctx + horizon]), m[: ctx + horizon]
        if seq.shape[1] != n_ch_expected:
            continue

        ch_rmse_i: List[float] = []
        ch_mae_i: List[float] = []
        for c in range(n_ch_expected):
            if scalers is not None:
                seq_norm = (
                    scalers[c]
                    .transform(seq[:, c].astype(np.float64).reshape(-1, 1))
                    .reshape(-1)
                    .astype(np.float32)
                )
            else:
                mu, sd = float(seq[:ctx, c].mean()), float(seq[:ctx, c].std()) + 1e-6
                seq_norm = ((seq[:, c] - mu) / sd).astype(np.float32)

            preds = forecast_channel(model, seq_norm, t, ctx, horizon, device, amp_dtype)
            gt = seq_norm[ctx : ctx + horizon]
            valid = m[ctx : ctx + horizon, c] == 1
            if np.any(valid):
                err = preds[valid] - gt[valid]
                ch_rmse_i.append(math.sqrt(float(np.mean(err * err))))  # this channel's RMSE
                ch_mae_i.append(float(np.mean(np.abs(err))))            # this channel's MAE

        if ch_rmse_i:
            sample_rmses.append(float(np.mean(ch_rmse_i)))  # macro over channels
            sample_maes.append(float(np.mean(ch_mae_i)))

    # Average the per-sample scores over all evaluated sequences.
    rmse = float(np.mean(sample_rmses)) if sample_rmses else float("nan")
    mae = float(np.mean(sample_maes)) if sample_maes else float("nan")
    print(f"[result] context={ctx} horizon={horizon} | RMSE={rmse:.6f} | MAE={mae:.6f} (n={len(sample_rmses)})")


def run_demo(args, model, device, amp_dtype) -> None:
    """Forecast a synthetic noisy sinusoid with irregular timestamps."""
    rng = np.random.default_rng(args.seed)
    ctx, horizon = args.context, args.horizon

    t = np.sort(rng.uniform(0, 60.0, size=ctx + horizon)).astype(np.float32)
    t = ensure_strictly_increasing(t)
    signal = np.sin(2 * np.pi * t / 8.0) + 0.3 * np.sin(2 * np.pi * t / 3.0)
    signal += rng.normal(0, 0.05, size=signal.shape).astype(np.float32)

    mu, sd = float(signal[:ctx].mean()), float(signal[:ctx].std()) + 1e-6
    seq_norm = ((signal - mu) / sd).astype(np.float32)

    preds = forecast_channel(model, seq_norm, t, ctx, horizon, device, amp_dtype)
    gt = seq_norm[ctx : ctx + horizon]
    rmse = float(np.sqrt(np.mean((preds - gt) ** 2)))
    print(f"[demo] synthetic series | RMSE={rmse:.4f} (normalized space)")
    print("[demo] first 8 predictions:", np.round(preds[:8], 3).tolist())
    print("[demo] first 8 ground truth:", np.round(gt[:8], 3).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTER zero-shot forecasting example")
    parser.add_argument("--model", type=str, required=True,
                        help="Local checkpoint directory (e.g. ./checkpoint-1000000) or HF hub id")
    parser.add_argument("--data", type=str, default=None, help="Evaluation JSONL file")
    parser.add_argument("--train_jsonl", type=str, default=None,
                        help="Training JSONL used to fit per-channel MinMax scalers (paper protocol)")
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic series instead of real data")
    parser.add_argument("--context", type=int, default=128, help="Context window length")
    parser.add_argument("--horizon", type=int, default=64, help="Forecast horizon")
    parser.add_argument("--num_eval", type=str, default="32",
                        help="Number of sequences to evaluate, or 'all' for the whole file")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16",
                        help="Autocast precision for inference on GPU (paper evaluation used bf16)")
    parser.add_argument("--seed", type=int, default=223)
    args = parser.parse_args()

    model, device = load_model(args.model)

    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    elif args.precision != "fp32":
        print(f"[info] precision={args.precision} ignored on CPU; running fp32")

    if args.demo or not args.data:
        run_demo(args, model, device, amp_dtype)
    else:
        run_evaluation(args, model, device, amp_dtype)


if __name__ == "__main__":
    main()
