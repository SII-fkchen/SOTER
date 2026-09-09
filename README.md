<div align="center">
  <h2><b>SOTER: Medical Time-Series Foundation Model</b></h2>
</div>

<div align="center">

**[<a href="https://arxiv.org/abs/XXXX.XXXXX">Paper Page</a>]** <!-- TODO: replace with the SOTER arXiv link -->
**[<a href="https://huggingface.co/YOUR_HF_USERNAME/SOTER">Model Weights</a>]** <!-- TODO: replace with your Hugging Face repo -->
**[<a href="https://github.com/YOUR_GITHUB_USERNAME/SOTER">GitHub</a>]** <!-- TODO: replace with your GitHub repo -->

</div>

## Overview

SOTER is a foundation model for medical time series, built upon
[MIRA](https://github.com/microsoft/MIRA) (NeurIPS '25) and designed for
heterogeneous real-world health data: irregularly sampled vitals, multi-channel
physiological waveforms, and wearable signals. SOTER learns unified
representations across datasets with very different sampling rates and channel
sets, and supports **zero-shot forecasting** at arbitrary future timestamps as
well as **frozen-embedding classification** with a simple linear probe.

**Key features**

- **T-shaped CI/CD architecture.** Low-level *channel-independent* (CI) layers
  model each channel's temporal dynamics separately with continuous-time
  rotary positional encoding (CT-RoPE); high-level *channel-dependence* (CD)
  layers attend across channels at the same physical time step, adding
  cross-channel fusion only where it is cheapest and most useful.
- **Causal PSD-guided MoE routing.** Instead of a learned router, each FFN is a
  Mixture-of-Experts whose routing logits are computed from the *causal prefix
  power spectral density* of the hidden states: the DFT spectrum up to the
  current token is split into frequency bands, one per expert, so experts
  specialize in distinct temporal frequency regimes. Routing is deterministic
  and requires **no auxiliary load-balancing loss**.
- **Terminal Neural ODE / CDE extrapolation.** A Neural ODE block integrates
  the last hidden state *continuously* from the last observed timestamp to any
  target future timestamp (`next_target_time_values`), enabling forecasting at
  arbitrary, irregular horizons. During masked-value (imputation-style)
  training, a cubic-spline control path turns the block into a Neural CDE.

**Released checkpoint** (`checkpoint-1000000`, ~113 M parameters, fp32):

| Field | Value |
|---|---|
| hidden size / heads / layers | 384 / 12 / 12 |
| intermediate size | 1536 |
| experts (routed + shared, top-k) | 8 + 1, k = 2 |
| CI layers / CD layers | 0–4, 6–11 / 5 |
| max sequence length | 4096 |
| ODE solver | `dopri5` (atol/rtol 1e-5) |

Pretraining data: SleepEDF, WESAD, PTB-XL, Chapman ECG and MIMIC-III Waveform
(JSONL; see [Data preparation](#data-preparation)).

---

## Installation

Requires Python 3.10+:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/SOTER.git
cd SOTER
pip install -r requirements.txt
```

> **Note:** the terminal ODE block requires
> [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq), already listed in
> `requirements.txt`.

## Quickstart

Download the weights from
[Hugging Face](https://huggingface.co/YOUR_HF_USERNAME/SOTER) and load them
with a **strict state-dict load** (this is the protocol used for all results
in the paper — see `examples/` for the full helper):

```python
import torch
from safetensors.torch import load_file
from soter.models.modeling_soter import SoterConfig, SoterForPrediction

ckpt = "/path/to/checkpoint-1000000"          # or the downloaded HF snapshot
config = SoterConfig.from_pretrained(ckpt)
model = SoterForPrediction(config)
model.load_state_dict(load_file(f"{ckpt}/model.safetensors"), strict=True)
model.eval().cuda()
```

> The Hugging Face repo also ships its modeling code, so
> `AutoModelForCausalLM.from_pretrained("YOUR_HF_USERNAME/SOTER", trust_remote_code=True)`
> works as a convenience path; the strict load above is the recommended,
> framework-version-independent way.

### Forecasting (zero-shot, autoregressive)

```python
# continuing from the snippet above, with `model` already loaded

# One univariate window: context of C points, predict the next P points.
# `values` are normalized (see README -> Input data format);
# `times` are the RAW physical timestamps (e.g. seconds), strictly increasing.
C, P = 128, 64
values = ...  # np.ndarray [C + P], float32
times  = ...  # np.ndarray [C + P], float32

cur_vals  = torch.tensor(values[:C], dtype=torch.float32).view(1, -1, 1).cuda()
cur_times = torch.tensor(times[:C],  dtype=torch.float32).view(1, -1).cuda()

preds = []
with torch.inference_mode():
    for i in range(P):
        next_t = torch.tensor([times[C + i]], device="cuda")
        out = model(
            input_ids=cur_vals,
            time_values=cur_times,
            next_target_time_values=next_t,   # ODE integrates t_last -> next_t
            attention_mask=torch.ones_like(cur_times, dtype=torch.long),
            return_dict=True,
        )
        nxt = out.logits[:, -1, :]            # [1, 1]
        preds.append(nxt)
        cur_vals  = torch.cat([cur_vals, nxt.unsqueeze(-1)], dim=1)
        cur_times = torch.cat([cur_times, next_t.view(1, 1)], dim=1)

preds = torch.stack(preds, dim=1).squeeze(0).cpu().numpy()  # [P]
```

A complete, runnable script (metrics + plotting, JSONL input, synthetic demo):

```bash
python examples/forecasting_example.py --model /path/to/checkpoint-1000000 --demo --plot forecast.png
```

### Classification (frozen embeddings + linear probe)

```python
# continuing from the loading snippet above
import torch

backbone = model.model                       # SoterModel without forecasting head

x  = torch.randn(4, 512, 3).cuda()           # [batch, length, channels]
t  = torch.linspace(0, 5, 512).expand(4, -1).cuda()  # raw timestamps
m  = torch.ones(4, 512, dtype=torch.long).cuda()

with torch.inference_mode():
    out = backbone(input_ids=x, time_values=t, attention_mask=m, return_dict=True)
hidden = out.last_hidden_state               # [batch, length, channels, hidden]
emb = (hidden * m.view(4, -1, 1, 1)).sum(1) / m.sum(1).view(-1, 1, 1)
emb = emb.mean(dim=1)                        # [batch, hidden] -> feed to any classifier
```

A complete frozen linear-probe pipeline (scaler fitting, batched embedding
extraction, LogisticRegression probe, Accuracy/Macro-F1, synthetic demo):

```bash
python examples/classification_example.py --model /path/to/checkpoint-1000000 --demo
```

---

## Input data format

All SOTER pipelines (training, forecasting, classification) use **JSONL**: one
JSON object per line, one record = one (windowed) multi-channel series.

| Field | Type | Shape | Required | Description |
|---|---|---|---|---|
| `sequence` | list[float] or list[list[float]] | `[L]` or `[L, C]` | **yes** | Signal values. 1-D = single channel; 2-D = `C` channels. |
| `time` | list[float] | `[L]` | **yes** | Raw physical timestamps (e.g. seconds). Should be non-decreasing; strictly increasing is enforced internally where required. |
| `mask` | list[int] or list[list[int]] | `[L]` or `[L, C]` | recommended | `1` = observed/valid, `0` = missing. Defaults to all-ones. |
| `label` | str / int | scalar | classification only | Class label for the linear probe (also searched inside `extra`). |
| `channel_names` | list[str] | `[C]` | optional | Channel names (metadata/auditing). |
| `sfreq` | float | scalar | optional | Sampling frequency in Hz (metadata). |
| `subject_id` / `dataset` | str | scalar | optional | Metadata used for subject-wise splits. |
| `extra` | dict | — | optional | Free-form metadata (labels are also looked up here). |

Example (multi-channel):

```json
{"sequence": [[1.0, 0.3], [1.2, 0.4], [0.8, 0.2]], "time": [0.12, 0.22, 0.41], "mask": [[1, 1], [1, 0], [1, 1]], "channel_names": ["PPG", "EDA"], "sfreq": 100.0, "subject_id": "S01", "dataset": "WESAD"}
```

Notes:

- **Normalization.** The released checkpoint was trained with per-channel
  MinMax scaling fitted on the *training* split
  (`examples/forecasting_example.py --train_jsonl` reproduces this).
- **Timestamps are raw.** Do **not** pre-normalize time: the model normalizes
  timestamps internally for CT-RoPE, and the ODE block needs physical time
  differences.
- **Forecasting targets.** At inference, pass the timestamp you want a
  prediction for via `next_target_time_values` — the horizon may be irregular.

## Data preparation

Prepare your own data in the JSONL format described above — converting raw
datasets into JSONL is up to you. Note that public datasets (e.g. PhysioNet
recordings) are governed by their own data-use agreements; apply for access
via the official providers.

## Training

Single dataset (auto-resumes from the latest `checkpoint-*` in `--output_path`):

```bash
python torch_dist_run.py main.py \
  -d ./data/processed_jsonl/SleepEDF/train_set.jsonl \
  -m YOUR_HF_USERNAME/SOTER \
  --output_path ./logs/soter \
  --from_scratch \
  --save_steps 10000 --save_strategy steps --save_total_limit 10 --save_only_model \
  --precision bf16 \
  --time_aware_dataset --time_aware_rotary
```

- `-m/--model_path` points to a checkpoint **or** just a config source: with
  `--from_scratch`, only its `config.json` is used to initialize a new model.
- Multi-GPU / multi-node: standard PyTorch elastic variables
  (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`); the launcher
  `torch_dist_run.py` wraps `torch.distributed`.
- `main.py --help` lists every option.

Multi-dataset mixed pretraining (50% future-forecasting / 50% masked-value
reconstruction, one dataset per batch), as used for the released checkpoint:

```bash
python torch_dist_run.py train_multidataset_soter.py \
  --train_jsonls ./data/processed_jsonl/SleepEDF/train_set.jsonl,./data/processed_jsonl/WESAD/train_set.jsonl,... \
  --output_path ./logs/soter_multi --precision bf16
```

Run `python train_multidataset_soter.py --help` for the full argument list.

## Repository layout

```
soter/                  # the SOTER package
  models/               # SoterConfig / SoterModel / SoterForPrediction, CT-RoPE,
                        # causal-PSD MoE, terminal ODE/CDE block
  datasets/             # JSONL datasets & sliding-window samplers
  trainer/              # Hugging Face Trainer wrapper
  runner.py             # training pipeline (used by main.py)
examples/
  forecasting_example.py      # zero-shot autoregressive forecasting + plot
  classification_example.py   # frozen embeddings + linear probe
main.py                 # single-dataset training entry
train_multidataset_soter.py   # multi-dataset mixed pretraining entry
torch_dist_run.py       # lightweight distributed launcher
```

## Citation

If you find SOTER useful, please cite our paper:

```bibtex
@article{soter2026,
  title   = {SOTER: TODO},
  author  = {TODO},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

and the MIRA paper this work builds upon:

```bibtex
@article{li2025mira,
  title   = {MIRA: Medical Time Series Foundation Model for Real-World Health Data},
  author  = {Li, Hao and Deng, Bowen and Xu, Chang and others},
  journal = {arXiv preprint arXiv:2506.07584},
  year    = {2025}
}
```

## Acknowledgments

SOTER is built on top of [MIRA](https://github.com/microsoft/MIRA) and reuses
parts of the [Time-MoE](https://github.com/Time-MoE/Time-MoE) training
pipeline. See `NOTICE.md`.

## License

This project is released under the [MIT License](LICENSE).
