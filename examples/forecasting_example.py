#!/usr/bin/env python
"""
Zero-shot forecasting example for SOTER.

SOTER forecasts autoregressively: given a context window of (value, timestamp)
pairs it predicts the value at the next *target timestamp* by integrating its
terminal ODE block from the last observed time to the target time, then feeds
the prediction back and repeats.

Usage
-----
1) Synthetic demo (no data needed):

    python examples/forecasting_example.py --demo \
        --model /path/to/checkpoint-1000000

2) Forecast your own JSONL data (see README.md "Input data format"):

    python examples/forecasting_example.py \
        --model /path/to/checkpoint-1000000 \
        --data ./data/processed_jsonl/MIT_BIH_OOD/test_set.jsonl \
        --train_jsonl ./data/processed_jsonl/MIT_BIH_OOD/train_set.jsonl \
        --context 128 --horizon 64 --num_eval 32 \
        --plot forecast.png

Normalization protocol
----------------------
The released checkpoint was trained with per-channel MinMax scaling fitted on
the *training* split. Pass `--train_jsonl` to reproduce that protocol. If no
training file is given, the example falls back to per-window z-score
normalization, which also works reasonably in practice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


def load_model(model_path: str, precision: str = "fp32") -> torch.nn.Module:
    """Load SOTER with a strict state-dict load (the protocol used in the paper).

    Accepts a local checkpoint directory or a Hugging Face hub id (the weights
    are downloaded as a local snapshot first, then loaded strictly).
    """
    from soter.models.modeling_soter import SoterConfig, SoterForPrediction

    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors*"])

    config = SoterConfig.from_pretrained(model_path)
    model = SoterForPrediction(config)
    model.load_state_dict(_load_state_dict(model_path), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] inference device: {device}")
    if precision == "bf16" and device.type == "cuda":
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data utilities (JSONL: one record per line, see README.md)
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_strictly_increasing(times: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """The ODE solver requires strictly increasing timestamps."""
    t = times.astype(np.float32).copy()
    for i in range(1, t.shape[0]):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def as_multichannel(rec: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sequence [L, C], time [L], mask [L, C]) from one JSONL record."""
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


def fit_minmax_scalers(records: List[dict], num_channels: int):
    """Fit one MinMaxScaler per channel on the training split (paper protocol)."""
    from sklearn.preprocessing import MinMaxScaler

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


# ---------------------------------------------------------------------------
# Autoregressive forecasting
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forecast_channel(
    model: torch.nn.Module,
    history: np.ndarray,
    times: np.ndarray,
    context: int,
    horizon: int,
) -> np.ndarray:
    """
    Forecast one (already normalized) univariate series.

    history: [context + horizon] normalized values
    times:   [context + horizon] raw timestamps (strictly increasing)
    returns: [horizon] predictions in normalized space
    """
    device = next(model.parameters()).device
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

        out = model(
            input_ids=cur_vals,
            time_values=cur_times,
            next_target_time_values=next_t,  # ODE integrates from t_last to this time
            attention_mask=torch.ones(cur_vals.shape[0], cur_vals.shape[1], dtype=torch.long, device=device),
            return_dict=True,
        )
        next_val = out.logits[0, -1, 0].float().cpu()
        preds[step] = float(next_val)

        # Feed the prediction back for the next step (autoregressive rollout).
        cur_vals = torch.cat([cur_vals, next_val.view(1, 1, 1)], dim=1)
        cur_times = torch.cat(
            [cur_times, torch.tensor([[next_t_val]], device=device)], dim=1
        )
    return preds


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args, model) -> None:
    records = load_jsonl(args.data)
    rng = np.random.default_rng(args.seed)
    idxs = rng.permutation(len(records))[: args.num_eval]

    ctx, horizon = args.context, args.horizon
    num_channels = int(as_multichannel(records[int(idxs[0])])[0].shape[1])

    if args.train_jsonl:
        print(f"[info] fitting per-channel MinMaxScaler on {args.train_jsonl}")
        scalers = fit_minmax_scalers(load_jsonl(args.train_jsonl), num_channels)
    else:
        print("[info] no --train_jsonl given; falling back to per-window z-score normalization")
        scalers = None

    per_ch_sq: List[List[float]] = [[] for _ in range(num_channels)]
    per_ch_abs: List[List[float]] = [[] for _ in range(num_channels)]
    plotted = False

    for i in idxs:
        seq, t, m = as_multichannel(records[int(i)])
        if seq.shape[0] < ctx + horizon or seq.shape[1] != num_channels:
            continue
        seq, t, m = seq[: ctx + horizon], ensure_strictly_increasing(t[: ctx + horizon]), m[: ctx + horizon]

        gt_plot = pred_plot = ctx_plot = None
        for c in range(num_channels):
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

            preds = forecast_channel(model, seq_norm, t, ctx, horizon)
            gt = seq_norm[ctx : ctx + horizon]
            valid = m[ctx : ctx + horizon, c] == 1
            if np.any(valid):
                err = preds[valid] - gt[valid]
                per_ch_sq[c].extend((err * err).tolist())
                per_ch_abs[c].extend(np.abs(err).tolist())

            if c == 0 and not plotted:
                ctx_plot, gt_plot, pred_plot = seq_norm[:ctx], gt, preds

        if args.plot and not plotted and gt_plot is not None:
            save_plot(ctx_plot, gt_plot, pred_plot, args.plot)
            plotted = True

    ch_rmse = [math.sqrt(np.mean(sq)) if sq else float("nan") for sq in per_ch_sq]
    ch_mae = [float(np.mean(ab)) if ab else float("nan") for ab in per_ch_abs]
    rmse = float(np.nanmean(ch_rmse))
    mae = float(np.nanmean(ch_mae))
    print(f"[result] context={ctx} horizon={horizon} | RMSE(macro over channels)={rmse:.4f} | MAE={mae:.4f}")


def save_plot(context_vals: np.ndarray, gt: np.ndarray, preds: np.ndarray, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = len(context_vals)
    plt.figure(figsize=(10, 4))
    plt.plot(range(ctx), context_vals, color="black", label="Context (observed)")
    plt.plot(range(ctx, ctx + len(gt)), gt, "b--", label="Future (ground truth)")
    plt.plot(range(ctx, ctx + len(preds)), preds, "r", label="Future (SOTER prediction)")
    plt.axvline(x=ctx - 1, color="gray", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[info] saved forecast plot to {out_path}")


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def run_demo(args, model) -> None:
    """Forecast a synthetic noisy sinusoid with irregular timestamps."""
    rng = np.random.default_rng(args.seed)
    ctx, horizon = args.context, args.horizon

    t = np.sort(rng.uniform(0, 60.0, size=ctx + horizon)).astype(np.float32)
    t = ensure_strictly_increasing(t)
    signal = np.sin(2 * np.pi * t / 8.0) + 0.3 * np.sin(2 * np.pi * t / 3.0)
    signal += rng.normal(0, 0.05, size=signal.shape).astype(np.float32)

    mu, sd = float(signal[:ctx].mean()), float(signal[:ctx].std()) + 1e-6
    seq_norm = ((signal - mu) / sd).astype(np.float32)

    preds = forecast_channel(model, seq_norm, t, ctx, horizon)
    gt = seq_norm[ctx : ctx + horizon]
    rmse = float(np.sqrt(np.mean((preds - gt) ** 2)))
    print(f"[demo] synthetic series | RMSE={rmse:.4f} (normalized space)")
    print("[demo] first 8 predictions:", np.round(preds[:8], 3).tolist())
    print("[demo] first 8 ground truth:", np.round(gt[:8], 3).tolist())

    if args.plot:
        save_plot(seq_norm[:ctx], gt, preds, args.plot)


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
    parser.add_argument("--num_eval", type=int, default=32, help="Number of sequences to evaluate")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--plot", type=str, default=None, help="Optional path to save a forecast plot (png)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.model, args.precision)

    if args.demo or not args.data:
        run_demo(args, model)
    else:
        run_evaluation(args, model)


if __name__ == "__main__":
    main()
