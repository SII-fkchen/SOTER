"""
Cubic spline utilities for SOTER CDE: fit observed (time, value) to get control path for mask prediction.
Used only in mask prediction / imputation; future prediction does not use spline (degenerates to ODE).
"""
from typing import Optional, Callable
import torch
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def build_spline_control_callable(
    time_values: torch.Tensor,
    observation_values: torch.Tensor,
    observation_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """
    Fit cubic spline to observed (time, value) per batch and per channel.
    Returns a callable z(t_eval) for CDE control path.

    Args:
        time_values: [B, L] time stamps
        observation_values: [B, L, C] or [B, L] observed values (masked positions can be any)
        observation_mask: [B, L] 1 = observed, 0 = missing
        device, dtype: output device and dtype

    Returns:
        callable: (t_eval) -> [B*C, control_dim]
            t_eval: [B] physical times at which to evaluate (one per batch item).
            For CDE we evaluate at same t for all channels of a batch item,
            so output is (B, C, 1) flattened to (B*C, 1) to match h_N_flat.
        Returns None if spline fitting is not possible (e.g. too few points).
    """
    if not HAS_SCIPY:
        return None

    time_np = time_values.detach().cpu().float().numpy()
    obs_np = observation_values.detach().cpu().float().numpy()
    mask_np = observation_mask.detach().cpu().numpy()

    if obs_np.ndim == 2:
        obs_np = obs_np[:, :, None]  # [B, L, 1]
    B, L, C = obs_np.shape

    # Per (batch, channel) store spline or None
    splines = []
    for b in range(B):
        row = []
        for c in range(C):
            if mask_np.ndim == 3:
                valid = mask_np[b, :, c] > 0
            else:
                valid = mask_np[b] > 0
            t_bc = time_np[b, valid]
            v_bc = obs_np[b, valid, c]
        # for c in range(C):
        #     valid = mask_np[b] > 0
        #     t_bc = time_np[b, valid]
        #     v_bc = obs_np[b, valid, c]
            if len(t_bc) < 2:
                row.append(None)
                continue
            # sort by time
            order = np.argsort(t_bc)
            t_bc = t_bc[order]
            v_bc = v_bc[order]
            try:
                sp = CubicSpline(t_bc, v_bc, bc_type='natural')
            except Exception:
                sp = None
            row.append(sp)
        splines.append(row)

    def control_at_t(t_eval: torch.Tensor) -> torch.Tensor:
        # t_eval: [B] or [B*C]
        t_eval = t_eval.detach().cpu().float().numpy()
        if t_eval.ndim == 0:
            t_eval = np.full(B, float(t_eval))
        elif t_eval.size == 1:
            t_eval = np.full(B, float(t_eval.flat[0]))
        out = np.zeros((B, C), dtype=np.float32)
        for b in range(B):
            t_b = float(t_eval[b]) if t_eval.ndim > 0 else float(t_eval)
            for c in range(C):
                if splines[b][c] is not None:
                    try:
                        out[b, c] = np.clip(splines[b][c](t_b), -1e6, 1e6)
                    except Exception:
                        out[b, c] = 0.0
                else:
                    out[b, c] = 0.0
        # [B, C] -> [B*C, 1] to match h_N layout
        out = torch.from_numpy(out).to(device=device, dtype=dtype).reshape(B * C, 1)
        return out

    return control_at_t
