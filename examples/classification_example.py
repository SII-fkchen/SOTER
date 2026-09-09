#!/usr/bin/env python
"""
Frozen linear-probe classification example for SOTER.

Protocol (same as the paper):
  1. Fit a per-channel MinMaxScaler on the *training* split.
  2. Extract fixed embeddings from the *frozen* SOTER backbone
     (masked mean-pooling over time, then mean-pooling over channels).
  3. Train a linear probe (StandardScaler + multinomial LogisticRegression)
     on the training embeddings.
  4. Report Accuracy and Macro-F1 on the test split.

Usage
-----
1) Synthetic demo (no data needed):

    python examples/classification_example.py --demo \
        --model /path/to/checkpoint-1000000

2) Your own labeled JSONL data (see README.md "Input data format"):

    python examples/classification_example.py \
        --model /path/to/checkpoint-1000000 \
        --train_jsonl ./data/MMASH/train_set.jsonl \
        --test_jsonl  ./data/MMASH/test_set.jsonl \
        --label_key label
"""

from __future__ import annotations

import argparse
import json
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


def load_backbone(model_path: str) -> torch.nn.Module:
    """Load SOTER with a strict state-dict load (the protocol used in the paper)
    and return its frozen backbone (without the forecasting head).

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

    backbone = model.model  # SoterModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] inference device: {device}")
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def get_label(rec: dict, label_key: str) -> Optional[str]:
    """Read a class label from the record (top level or inside the `extra` dict)."""
    val = rec.get(label_key)
    if val is None and isinstance(rec.get("extra"), dict):
        val = rec["extra"].get(label_key)
    if val is None or str(val).strip() == "":
        return None
    return str(val).strip()


def fit_minmax_scalers(records: List[dict], num_channels: int):
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


def normalize_record(rec: dict, scalers, max_length: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Scale one record and pad/truncate it to `max_length`.

    Returns (values [max_length, C], times [max_length], attention [max_length]).
    """
    seq, t, m = as_multichannel(rec)
    seq = seq[:max_length]
    t = t[:max_length]
    m = m[:max_length]
    if seq.shape[0] < 2:
        return None

    out = np.zeros((max_length, seq.shape[1]), dtype=np.float32)
    attn = np.zeros((max_length,), dtype=np.int64)
    for c in range(seq.shape[1]):
        out[: seq.shape[0], c] = (
            scalers[c]
            .transform(seq[:, c].astype(np.float64).reshape(-1, 1))
            .reshape(-1)
            .astype(np.float32)
        )
    times = np.zeros((max_length,), dtype=np.float32)
    times[: seq.shape[0]] = t
    attn[: seq.shape[0]] = 1
    row_valid = (m.max(axis=1) > 0).astype(np.int64)  # a timestep counts if any channel is valid
    attn[: seq.shape[0]] = row_valid
    return out, times, attn


# ---------------------------------------------------------------------------
# Frozen embedding extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_embeddings(
    backbone: torch.nn.Module,
    records: List[dict],
    scalers,
    max_length: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[int]]:
    """Return ([N, hidden] embeddings, indices of the records used)."""
    device = next(backbone.parameters()).device
    feats, used = [], []
    batch_x, batch_t, batch_a, batch_idx = [], [], [], []

    def flush():
        if not batch_x:
            return
        x = torch.from_numpy(np.stack(batch_x)).to(device)          # [B, L, C]
        tt = torch.from_numpy(np.stack(batch_t)).to(device)         # [B, L]
        aa = torch.from_numpy(np.stack(batch_a)).long().to(device)  # [B, L]
        out = backbone(input_ids=x, time_values=tt, attention_mask=aa, return_dict=True)
        hidden = out.last_hidden_state                              # [B, L, C, H]
        mask = aa.to(hidden.dtype).view(aa.shape[0], aa.shape[1], 1, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, C, H]
        emb = pooled.mean(dim=1)                                    # [B, H]
        feats.append(emb.float().cpu().numpy())
        used.extend(batch_idx)
        batch_x.clear(), batch_t.clear(), batch_a.clear(), batch_idx.clear()

    for i, rec in enumerate(records):
        prepared = normalize_record(rec, scalers, max_length)
        if prepared is None:
            continue
        x, tt, aa = prepared
        batch_x.append(x)
        batch_t.append(tt)
        batch_a.append(aa)
        batch_idx.append(i)
        if len(batch_x) >= batch_size:
            flush()
    flush()
    return np.concatenate(feats, axis=0), used


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_probe(args, backbone, train_records, test_records) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    label_key = args.label_key
    train_pairs = [(r, get_label(r, label_key)) for r in train_records]
    test_pairs = [(r, get_label(r, label_key)) for r in test_records]
    train_pairs = [(r, y) for r, y in train_pairs if y is not None]
    test_pairs = [(r, y) for r, y in test_pairs if y is not None]
    if not train_pairs or not test_pairs:
        raise RuntimeError(f"No records carry a `{label_key}` label field.")

    num_channels = int(as_multichannel(train_pairs[0][0])[0].shape[1])
    scalers = fit_minmax_scalers([r for r, _ in train_pairs], num_channels)

    print(f"[info] extracting embeddings: {len(train_pairs)} train / {len(test_pairs)} test records")
    x_train, used_train = extract_embeddings(
        backbone, [r for r, _ in train_pairs], scalers, args.max_length, args.batch_size
    )
    x_test, used_test = extract_embeddings(
        backbone, [r for r, _ in test_pairs], scalers, args.max_length, args.batch_size
    )
    y_train = np.asarray([train_pairs[i][1] for i in used_train])
    y_test = np.asarray([test_pairs[i][1] for i in used_test])

    probe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    probe.fit(x_train, y_train)
    y_pred = probe.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[result] classes={sorted(set(y_train))}")
    print(f"[result] Accuracy={acc:.4f} | Macro-F1={f1:.4f} (n_test={len(y_test)})")


# ---------------------------------------------------------------------------
# Synthetic demo: 3 classes of 1-channel signals
# ---------------------------------------------------------------------------

def make_demo_data(seed: int = 0) -> Tuple[List[dict], List[dict]]:
    rng = np.random.default_rng(seed)
    classes = {"slow": 0.5, "fast": 2.0, "chirp": None}

    def one(cls_name: str) -> dict:
        freq = classes[cls_name]
        t = np.sort(rng.uniform(0, 10.0, size=256)).astype(np.float32)
        for i in range(1, len(t)):  # strictly increasing
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + 1e-4
        phase = rng.uniform(0, 2 * np.pi)
        if cls_name == "chirp":
            sig = np.sin(2 * np.pi * (0.2 * t + 0.15 * t**2) + phase)
        else:
            sig = np.sin(2 * np.pi * freq * t + phase)
        sig = sig + rng.normal(0, 0.1, size=sig.shape).astype(np.float32)
        return {
            "sequence": sig.astype(np.float32).tolist(),
            "time": t.tolist(),
            "mask": [1] * len(t),
            "label": cls_name,
        }

    train = [one(c) for c in classes for _ in range(12)]
    test = [one(c) for c in classes for _ in range(5)]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTER frozen linear-probe classification example")
    parser.add_argument("--model", type=str, required=True,
                        help="Local checkpoint directory (e.g. ./checkpoint-1000000) or HF hub id")
    parser.add_argument("--train_jsonl", type=str, default=None, help="Labeled training JSONL")
    parser.add_argument("--test_jsonl", type=str, default=None, help="Labeled test JSONL")
    parser.add_argument("--label_key", type=str, default="label",
                        help="JSON field holding the class label (also searched inside `extra`)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Windows are truncated / zero-padded to this length")
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic 3-class data")
    args = parser.parse_args()

    backbone = load_backbone(args.model)

    if args.demo or not (args.train_jsonl and args.test_jsonl):
        train_records, test_records = make_demo_data()
        print(f"[demo] synthetic data: {len(train_records)} train / {len(test_records)} test, 3 classes")
    else:
        train_records = load_jsonl(args.train_jsonl)
        test_records = load_jsonl(args.test_jsonl)

    run_probe(args, backbone, train_records, test_records)


if __name__ == "__main__":
    main()
