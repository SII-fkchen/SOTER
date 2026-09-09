import torch
import numpy as np

def time_aware_collate_fn(batch, pad_value=0.0):
    batch = [b for b in batch if b is not None and b.get("input_ids") is not None]
    if len(batch) == 0:
        return {}

    max_len = max(len(b["input_ids"]) for b in batch)

    padded_batch = {
        "input_ids": [], "time_values": [], "attention_mask": [], 
        "labels": [], "loss_masks": [], "next_target_time_values": []
    }

    for item in batch:
        L = len(item["input_ids"])
        pad_len = max_len - L
        pad_time_val = float(item["time_values"][-1]) if len(item["time_values"]) > 0 else 0.0

        # ==========================================================
        # Core fix 1: dynamically support padding for both single-channel and multi-channel sequences
        # ==========================================================
        input_ids = item["input_ids"]
        if input_ids.ndim == 1:
            pad_width = (0, pad_len) # single channel [SeqLen]
        else:
            # multi-channel [SeqLen, Channel]: pad only the sequence dimension (dim 0), not the channel dimension (dim 1)
            pad_width = ((0, pad_len), (0, 0)) 
            
        padded_batch["input_ids"].append(np.pad(input_ids, pad_width, constant_values=pad_value))
        
        # time_values and attention_mask are 1-D [SeqLen]; use (0, pad_len) directly
        padded_batch["time_values"].append(np.pad(item["time_values"], (0, pad_len), constant_values=pad_time_val))
        padded_batch["attention_mask"].append(np.pad(item["attention_mask"], (0, pad_len), constant_values=0))
        
        # ==========================================================
        # Core fix 2: for the single-task Future Prediction, labels and loss_masks must never be padded
        # ==========================================================
        padded_batch["labels"].append(item["labels"])
        padded_batch["loss_masks"].append(item.get("loss_masks", np.ones_like(item["labels"])))

        nxt_time = item.get("next_target_time_values")
        if nxt_time is None:
            nxt_time = item["time_values"][-1] + 1.0
        padded_batch["next_target_time_values"].append(nxt_time)

    return {
        "input_ids": torch.tensor(np.stack(padded_batch["input_ids"]), dtype=torch.float32),
        "time_values": torch.tensor(np.stack(padded_batch["time_values"]), dtype=torch.float32),
        "attention_mask": torch.tensor(np.stack(padded_batch["attention_mask"]), dtype=torch.long),
        "labels": torch.tensor(np.stack(padded_batch["labels"]), dtype=torch.float32),
        "loss_masks": torch.tensor(np.stack(padded_batch["loss_masks"]), dtype=torch.float32), 
        "next_target_time_values": torch.tensor(padded_batch["next_target_time_values"], dtype=torch.float32),
    }