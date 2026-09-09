'''
Part of code from MIRA.models.modeling_mira
https://github.com/microsoft/MIRA
'''

import math
from typing import Optional, Tuple, List, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from .configuration_soter import SoterConfig
from .ts_generation_mixin import MIRAGenerationMixin
from .utils_time_normalization import normalize_time_for_ctrope
from .spline_utils import build_spline_control_callable
from typing import Dict, Any, Optional, List, Union, Callable

logger = logging.get_logger(__name__)

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass

try:
    # Import torchdiffeq for the ODE solver
    from torchdiffeq import odeint_adjoint as odeint
    is_torchdiffeq_available = True
except ImportError:
    logger.info("torchdiffeq not found. Terminal ODE module will not be usable. Please install it: pip install torchdiffeq")
    # Define a dummy odeint to prevent NameError, but raise NotImplementedError if called
    odeint = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("torchdiffeq is not installed."))
    is_torchdiffeq_available = False


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MIRAInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    """

    def __init__(self, config: SoterConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default 1
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


class ContinuousTimeRotaryEmbedding(nn.Module):
    """Continuous-Time Rotary Positional Encoding (CT-RoPE) based on paper description."""
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)

        # Angular frequencies omega_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False) # [dim / 2]

        # formula theta_i(t) = omega_i * t.
        # Caching for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._last_t_hash = None # Cache based on time_values tensor content

    def _compute_cos_sin(self, t: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """Compute cos/sin embeddings based on time values t."""
        # Simple hash: use sum and shape
        current_hash = (t.sum().item(), t.shape, device, dtype)
        if current_hash == self._last_t_hash and self._cos_cached is not None:
            # Ensure cache is on correct device/dtype
            if self._cos_cached.device == device and self._cos_cached.dtype == dtype:
                 return self._cos_cached, self._sin_cached

        # Compute angles theta_i(t) = omega_i * t
        # t shape: [batch, seq_len]
        # inv_freq shape: [dim/2]
        # freqs shape: [batch, seq_len, dim/2]
        freqs = torch.einsum('b s, d -> b s d', t.to(device=device, dtype=torch.float32), self.inv_freq)

        # Concatenate for full dimension embedding
        emb = torch.cat((freqs, freqs), dim=-1) # Shape [batch, seq_len, dim]

        # Add head dimension for broadcasting: [batch, 1, seq_len, dim]
        emb = emb.unsqueeze(1)

        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        # Update cache
        self._cos_cached = cos
        self._sin_cached = sin
        self._last_t_hash = current_hash
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, time_values: torch.Tensor):
        """Apply CT-RoPE using absolute time_values."""
        # q, k shape: [batch, num_heads, seq_len, head_dim]
        # time_values shape: [batch, seq_len]
        batch_size, num_heads, seq_len, head_dim = q.shape

        if head_dim != self.dim:
             raise ValueError(f"Input head dimension {head_dim} must match RoPE dimension {self.dim}")
        if time_values.shape != (batch_size, seq_len):
             # Handle potential mismatch during generation with KV cache
             if q.shape[2] == 1 and time_values.shape[1] == 1: # Common case for single new token
                 pass # time_values is correctly shaped
             else:
                raise ValueError(f"time_values shape mismatch. Expected BxL = ({batch_size}, {seq_len}), got {time_values.shape}")


        cos, sin = self._compute_cos_sin(time_values, q.device, q.dtype) # Shape [B, 1, L, D]

        # Apply rotation
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed, cos, sin


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding 
class MIRARotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm 
class MIRARMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MIRATemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class MIRAMLP(MIRATemporalBlock):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


# Copied from time_moe.models.modeling_time_moe
class MIRASparseExpertsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MIRATemporalBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = MIRATemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class SoterSparseExpertsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        # Keep one shared expert as global residual branch.
        self.num_shared_experts = 1
        self.fft_size = 512
        self.dft_router_impl = getattr(config, "dft_router_impl", "for_loop")
        
        moe_intermediate_size = self.config.intermediate_size // self.top_k
        self.general_experts = nn.ModuleList(
            [MIRATemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )
        self.shared_expert = MIRATemporalBlock(
            hidden_size=self.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )

    def _causal_psd_router_logits_for_loop(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Build strictly causal PSD routing:
        router logits at token t are computed from prefix hidden_states[:, :t+1, :].
        This implementation is mathematically equivalent to per-step prefix rFFT,
        but uses incremental DFT state updates to avoid recomputing FFT from scratch.
        """
        b_c, seq_len, _ = hidden_states.shape
        hidden_float = hidden_states.float()
        hidden_complex = hidden_float.to(torch.complex64)
        freq_bins = self.fft_size // 2 + 1
        device = hidden_states.device

        # phase[t, k] = exp(-j * 2pi * k * t / N)
        t_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        k_idx = torch.arange(freq_bins, device=device, dtype=torch.float32)
        angles = (-2.0 * math.pi / float(self.fft_size)) * torch.outer(t_idx, k_idx)
        phase = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)

        # Running DFT state X_t(k,h) for each (batch*channel, freq, hidden)
        dft_state = torch.zeros(
            (b_c, freq_bins, hidden_states.shape[-1]),
            dtype=torch.complex64,
            device=device,
        )
        router_logits_seq = []
        for t in range(seq_len):
            x_t = hidden_complex[:, t, :].unsqueeze(1)  # [B*C, 1, H]
            phase_t = phase[t].unsqueeze(0).unsqueeze(-1)  # [1, F, 1]
            dft_state = dft_state + x_t * phase_t
            fft_mag = torch.abs(dft_state)
            psd = fft_mag.mean(dim=-1)
            bands = torch.tensor_split(psd, self.num_experts, dim=1)
            logits_t = torch.stack([band.sum(dim=1) for band in bands], dim=1)
            router_logits_seq.append(logits_t)
        return torch.stack(router_logits_seq, dim=1).reshape(b_c * seq_len, self.num_experts)

    def _causal_psd_router_logits_prefix_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Tensorized causal PSD routing via prefix cumulative sum:
          X_t(k,h) = sum_{tau<=t} x_tau(h) * exp(-j*2*pi*k*tau/N)
        This removes Python for-loop over sequence length and runs as batched kernels.
        """
        b_c, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        freq_bins = self.fft_size // 2 + 1

        hidden_complex = hidden_states.float().to(torch.complex64)  # [B*C, T, H]
        t_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        k_idx = torch.arange(freq_bins, device=device, dtype=torch.float32)
        angles = (-2.0 * math.pi / float(self.fft_size)) * torch.outer(t_idx, k_idx)  # [T, F]
        phase = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)  # [T, F]

        # [B*C, T, F, H] then prefix-sum over T to enforce strict causality.
        contributions = hidden_complex.unsqueeze(2) * phase.unsqueeze(0).unsqueeze(-1)
        dft_prefix = torch.cumsum(contributions, dim=1)  # [B*C, T, F, H]
        psd = torch.abs(dft_prefix).mean(dim=-1)  # [B*C, T, F]

        bands = torch.tensor_split(psd, self.num_experts, dim=2)
        router_logits_seq = torch.stack([band.sum(dim=2) for band in bands], dim=2)  # [B*C, T, E]
        return router_logits_seq.reshape(b_c * seq_len, self.num_experts)

    def _causal_psd_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            impl = str(self.dft_router_impl).lower()
            if impl == "prefix_parallel":
                return self._causal_psd_router_logits_prefix_parallel(hidden_states)
            return self._causal_psd_router_logits_for_loop(hidden_states)
        
    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [Batch*Channel, SeqLen, Hidden]
        B_C, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self._causal_psd_router_logits(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.general_experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.numel() > 0:
                current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_flat.dtype))

        # Shared expert only contributes global residual information.
        global_hidden = hidden_states.mean(dim=1, keepdim=True).reshape(-1, hidden_dim)
        shared_output = self.shared_expert(global_hidden).view(B_C, 1, hidden_dim)
        shared_output = shared_output.expand(-1, seq_len, -1).reshape(-1, hidden_dim)

        final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = final_hidden_states.reshape(B_C, seq_len, hidden_dim)
        
        return final_hidden_states, router_logits
        
class MIRAAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: SoterConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = ContinuousTimeRotaryEmbedding(
            self.head_dim,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            time_values: Optional[torch.Tensor] = None,
            skip_rope: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        self._time_aware_rotary_flag = True
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        past_len = past_key_value.get_usable_length(None, self.layer_idx) if past_key_value is not None else 0
        kv_seq_len = past_len + key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            
        if not skip_rope:
            if isinstance(self.rotary_emb, ContinuousTimeRotaryEmbedding):
                if time_values is None:
                    raise ValueError("SOTER Error: time_values cannot be None when using ContinuousTimeRotaryEmbedding in CI layers.")
                # CT-RoPE: forward(q, k, time_values) returns already-rotated q_embed, k_embed and cos/sin for cache
                query_states, key_states, cos, sin = self.rotary_emb(query_states, key_states, time_values)
            elif isinstance(self.rotary_emb, MIRARotaryEmbedding):
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            cos, sin = None, None

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply provided attention mask (additive 4D causal mask)
        if attention_mask is not None:
            # Ensure mask covers the full KV sequence length after caching
            expected_mask_shape_part = (bsz, 1, q_len, kv_seq_len)
            if attention_mask.shape != expected_mask_shape_part:
                # Try slicing mask if it's longer (e.g., full context length mask passed)
                if attention_mask.shape[-1] >= kv_seq_len:
                    attention_mask_sliced = attention_mask[:, :, -q_len:, :kv_seq_len]
                    if attention_mask_sliced.shape == expected_mask_shape_part:
                        attention_mask = attention_mask_sliced
                    else:
                        raise ValueError(f"Attention mask shape error. Expected sliceable to {expected_mask_shape_part}, got {attention_mask.shape}")
                else:
                    raise ValueError(f"Attention mask shape error. Expected {expected_mask_shape_part}, got {attention_mask.shape}")
            
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MIRAFlashAttention2(MIRAAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            skip_rope: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:

            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        origin_dtype = query_states.dtype
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            query_states = query_states.to(dtype=torch.bfloat16)
            key_states = key_states.to(dtype=torch.bfloat16)
            value_states = value_states.to(dtype=torch.bfloat16)

        # without attention mask to faster speed
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal
        )
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            return attn_output.to(origin_dtype)
        else:
            return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


TIME_MOE_ATTENTION_CLASSES = {
    "eager": MIRAAttention,
    'flash_attention_2': MIRAFlashAttention2,
}


class MIRADecoderLayer(nn.Module):
    def __init__(self, config: SoterConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = TIME_MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        if self.config.use_dense:
            self.ffn_layer = MIRAMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = MIRASparseExpertsLayer(config)
        self.input_layernorm = MIRARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MIRARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            time_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            time_values=time_values,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value, router_logits

class SoterDecoderLayer(nn.Module):
    def __init__(self, config: SoterConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = TIME_MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        if self.config.use_dense:
            self.ffn_layer = MIRAMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = SoterSparseExpertsLayer(config)
        self.input_layernorm = MIRARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MIRARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            time_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            num_channels: int = 1,
            **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        is_cd_layer = (self.layer_idx in self.config.cd_layers) and (num_channels > 1)
        
        if is_cd_layer:
            B_C, L, H = hidden_states.shape
            B = B_C // num_channels
            C = num_channels
            
            hidden_states = hidden_states.view(B, C, L, H).permute(0, 2, 1, 3).reshape(B * L, C, H)
            
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None, 
                output_attentions=output_attentions,
                skip_rope=True,
                past_key_value=None,
                use_cache=False
            )
            
            hidden_states = hidden_states.view(B, L, C, H).permute(0, 2, 1, 3).reshape(B * C, L, H)
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                time_values=time_values,
                output_attentions=output_attentions,
                skip_rope=False,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, present_key_value, router_logits


class MIRAPreTrainedModel(PreTrainedModel):
    config_class = SoterConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MIRADecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _initialize_weights(self, module, is_remote_code: bool = False):
        # Never re-initialize parameters that were just loaded from a checkpoint
        # (they carry the `_is_hf_initialized` flag). The base implementation only
        # applies this param-level check for remote-code models; without it,
        # `from_pretrained` would silently re-randomize every nn.Linear/Embedding
        # after loading when this package is used locally.
        if getattr(module, "_is_hf_initialized", False):
            return
        direct_params = list(module.parameters(recurse=False))
        direct_buffers = [b for b in module.buffers(recurse=False) if b is not None]
        if direct_params and all(getattr(p, "_is_hf_initialized", False) for p in direct_params) and all(
            getattr(b, "_is_hf_initialized", False) for b in direct_buffers
        ):
            module._is_hf_initialized = True
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])


# =========================================================================
# Replacement 4: modify the backbone model to implement multi-channel folding and safe alignment in CI mode
# =========================================================================
class SoterModel(MIRAPreTrainedModel):
    """
    SOTER backbone: a T-shaped architecture that stacks channel-independent (CI)
    layers for per-channel temporal modeling with CT-RoPE, followed by
    channel-dependence (CD) layers (``config.cd_layers``) that attend across
    channels at the same time step.
    """

    def __init__(self, config: SoterConfig):
        super().__init__(config)
        # The embedding only maps a single channel (input_size=1); channels are
        # folded into the batch dimension below (channel independence).
        self.embed_layer = MIRAInputEmbedding(config)

        self.layers = nn.ModuleList(
            [SoterDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self._attn_implementation = config._attn_implementation
        if config.time_aware_rotary and self._attn_implementation == "flash_attention_2":
            logger.warning("CT-RoPE requires specific adaptations for Flash Attention. Falling back to 'eager'.")
            self._attn_implementation = "eager" 

        self.norm = MIRARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_layer.emb_layer

    def set_input_embeddings(self, value):
        self.embed_layer.emb_layer = value

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            time_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # =================================================================
        # 1. Core innovation: CI channel-folding mechanism (Channel Independence)
        # =================================================================
        num_channels = 1
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            if len(input_ids.shape) == 2:
                input_ids.unsqueeze_(dim=-1)
            
            # Get the true batch, seq_len and number of channels
            batch_size, seq_length, num_channels = input_ids.shape
            
            # [Batch, SeqLen, Channel] -> [Batch, Channel, SeqLen] -> [Batch * Channel, SeqLen, 1]
            # Force multi-lead features to be split into independent batch samples, reducing the input dimension to 1
            input_ids = input_ids.transpose(1, 2).reshape(batch_size * num_channels, seq_length, 1)
        elif inputs_embeds is not None:
            # Fallback logic for when embeds are passed directly during inference
            batch_size_mul_channels, seq_length, _ = inputs_embeds.shape
            batch_size = batch_size_mul_channels
            num_channels = 1
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # =================================================================
        # 2. Dimension-safe cloning of auxiliary variables (aligned to B * C)
        # =================================================================
        effective_batch_size = batch_size * num_channels

        # Align time_values: [Batch, SeqLen] -> [Batch * Channel, SeqLen]
        if time_values is not None:
            time_values = time_values.repeat_interleave(num_channels, dim=0)

        # Align attention_mask: [Batch, SeqLen] -> [Batch * Channel, SeqLen]
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_channels, dim=0)

        # Generate position_ids matching effective_batch_size
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache and past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is not None:
                past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.view(-1, seq_length)

        # CT-RoPE time normalization (based on time_values already expanded to B*C)
        if time_values is not None and getattr(self.config, "time_aware_rotary", False):
            alpha = getattr(self.config, "time_scale", 1.0)
            time_values, t_min, t_max = normalize_time_for_ctrope(
                time_values=time_values,
                attention_mask=attention_mask,
                seq_length=seq_length,
                alpha=alpha,
            )

        if inputs_embeds is None:
            # At this point input_ids has shape [B*C, L, 1]; after embedding it becomes [B*C, L, H]
            inputs_embeds = self.embed_layer(input_ids)

        # Construct the 4D causal mask (using effective_batch_size)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (effective_batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = ()
        next_decoder_cache = None

        # =================================================================
        # 3. Iterate through the T-shaped network (pass num_channels for CD-layer reshaping)
        # =================================================================
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    time_values,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    num_channels,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    time_values=time_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    num_channels=num_channels,
                )

            hidden_states = layer_outputs[0]
            all_router_logits += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if use_cache:
                next_decoder_cache = layer_outputs[2]

        # First apply global normalization at the [B*C, L, H] scale
        hidden_states = self.norm(hidden_states)

        # =================================================================
        # 4. Feature unfolding and spatial restoration
        # =================================================================
        # Restoration logic: [Batch * Channel, SeqLen, Hidden] 
        # -> view: [Batch, Channel, SeqLen, Hidden] 
        # -> transpose: [Batch, SeqLen, Channel, Hidden]
        hidden_states = hidden_states.view(batch_size, num_channels, seq_length, -1).transpose(1, 2).contiguous()

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if getattr(self, "use_legacy_cache", False) else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
            
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits
        )
    

class ODEFunc(nn.Module):
    """
    Time-augmented dynamics: dh/ds = f(s, h[, z]).
    - Future prediction: z=None -> pure Neural ODE (time s only).
    - Mask prediction: z = spline(t) -> CDE driven by cubic-spline control path.
    """
    def __init__(self, config: SoterConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.use_time = config.ode_func_use_time
        self.activation = ACT2FN[config.ode_func_activation]
        cde_enabled = getattr(config, "cde_control_dim", 0) > 0
        self.control_dim = config.input_size if cde_enabled else 0

        layers = []
        input_dim = self.hidden_size
        if self.use_time:
            input_dim += 1
        if self.control_dim > 0:
            input_dim += self.control_dim
        current_dim = input_dim

        for hidden_dim in config.ode_func_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, s, h, z: Optional[torch.Tensor] = None):
        # s: scalar integration time; h: [B, hidden_size]; z: [B, control_dim] or None
        parts = [h]
        if self.use_time:
            s_vec = torch.full((h.shape[0], 1), s.item(), device=h.device, dtype=h.dtype)
            parts.insert(0, s_vec)
        if self.control_dim > 0:
            if z is None:
                z = torch.zeros(h.shape[0], self.control_dim, device=h.device, dtype=h.dtype)
            parts.append(z)
        input_ode = torch.cat(parts, dim=-1)
        return self.net(input_ode)


class TerminalODEBlock(nn.Module):
    def __init__(self, config: SoterConfig):
        super().__init__()
        self.config = config
        if not is_torchdiffeq_available:
             self.ode_func = None
             logger.error("TerminalODEBlock created, but torchdiffeq is not available.")
        else:
             self.ode_func = ODEFunc(config)
        self.ode_method = config.ode_solver_method
        self.atol = config.ode_solver_atol
        self.rtol = config.ode_solver_rtol

    # def forward(self, h_N: torch.Tensor, t_N: torch.Tensor, t_Nplus1: torch.Tensor):
    #     if self.ode_func is None:
    #         raise RuntimeError("torchdiffeq is not installed, cannot use TerminalODEBlock.")

    #     if h_N.dim() != 2: raise ValueError(f"Expected h_N [B, D], got {h_N.shape}")
    #     if t_N.dim() > 1 or t_Nplus1.dim() > 1: warnings.warn("ODE time inputs have >1 dim, ensure compatibility.")

    #     # ODE solver expects time points as a 1D tensor.
    #     delta_t = t_Nplus1 - t_N
    #     # Handle cases where delta_t might be zero or negative per batch item
    #     # We consider delat_t smaller than 1 as regular time interval as pos_ids: 0,1,2...
    #     if torch.any(delta_t <= 1):
    #          warnings.warn("ODE integration interval delta_t <= 0 detected for some batch items. Returning initial state h_N for those items.")
    #          # Identify indices where delta_t > 0
    #          valid_indices = delta_t > 0
    #          if not torch.any(valid_indices): return h_N # All invalid, return original
             
    #          # Prepare inputs only for valid indices
    #          h_N_valid = h_N[valid_indices]
    #          delta_t_valid = delta_t[valid_indices]
    #          max_delta_t = torch.max(delta_t_valid)
    #          t_eval = torch.tensor([0.0, max_delta_t.item()], device=h_N.device)

    #          # Solve ODE only for valid batch items
    #          solution_valid = odeint(
    #              self.ode_func, h_N_valid, t_eval,
    #              method=self.ode_method, atol=self.atol, rtol=self.rtol,
    #              adjoint_params=tuple(self.parameters()) # Pass ODEFunc params for adjoint
    #          )
    #          h_extrapolated_valid = solution_valid[-1] # State at max_delta_t [N_valid, D]

    #          if torch.any(delta_t <= 1):
    #              warnings.warn("delta_t negative detected, returning input state for those items.")
    #              # For now, just solve for the whole batch using first item's delta_t for t_eval shape
    #              # This requires user to ensure valid delta_t or handle results carefully.
    #              t_eval = torch.tensor([0.0, delta_t[0].item()], device=h_N.device)

    #     else:
    #         # All delta_t are positive, use the first one to define t_eval interval points
    #         t_eval = torch.tensor([0.0, delta_t[0].item()], device=h_N.device)


    #     # --- Solve ODE (relative time integral) ---
    #     solution = odeint(
    #         self.ode_func,
    #         h_N, # Initial condition y0 [B, D]
    #         t_eval, # Evaluate at relative time 0 and delta_t [2]
    #         method=self.ode_method,
    #         atol=self.atol,
    #         rtol=self.rtol,
    #         adjoint_params=tuple(p for p in self.ode_func.parameters() if p.requires_grad) # Adjoint needs params
    #     )

    #     # Extract the solution at the end of the interval (relative time delta_t)
    #     h_extrapolated = solution[-1] # Shape [B, D]

    #     # If some intervals were invalid, restore original h_N for those
    #     if torch.any(delta_t <= 0):
    #          h_extrapolated[delta_t <= 0] = h_N[delta_t <= 0]

    #     return h_extrapolated
    def forward(
        self,
        h_N: torch.Tensor,
        t_N: torch.Tensor,
        t_Nplus1: torch.Tensor,
        control_path_callable: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_channels: int = 1,
    ):
        """
        Future prediction: control_path_callable=None -> Neural ODE (degenerate).
        Mask prediction: control_path_callable(t_phys) returns z [B*C, control_dim] -> time-augmented CDE.
        """
        if self.ode_func is None:
            raise RuntimeError("torchdiffeq is not installed, cannot use TerminalODEBlock.")

        if h_N.dim() != 2:
            raise ValueError(f"Expected h_N [B, D], got {h_N.shape}")

        delta_t = t_Nplus1 - t_N
        delta_t = torch.clamp(delta_t, min=1e-5)
        B_C = h_N.shape[0]
        B = B_C // num_channels

        use_cde = control_path_callable is not None and getattr(self.ode_func, "control_dim", 0) > 0

        def solve(h_N_in, t_N_in, t_Nplus1_in, delta_t_in):
            t_eval = torch.tensor([0.0, 1.0], dtype=h_N_in.dtype, device=h_N_in.device)

            if use_cde:
                # CDE: z(t) from cubic spline; t_phys per batch item
                t_N_b = t_N_in.view(B, num_channels)[:, 0]
                t_Nplus1_b = t_Nplus1_in.view(B, num_channels)[:, 0]
                delta_t_b = t_Nplus1_b - t_N_b

                def scaled_ode_func(s, h):
                    t_phys = t_N_b + s * delta_t_b
                    z = control_path_callable(t_phys)
                    dh = self.ode_func(s, h, z)
                    return dh * delta_t_in.unsqueeze(1)
            else:
                # ODE (future prediction): no control
                def scaled_ode_func(s, h):
                    dh = self.ode_func(s, h, None)
                    return dh * delta_t_in.unsqueeze(1)

            solution = odeint(
                scaled_ode_func,
                h_N_in,
                t_eval,
                method=self.ode_method,
                atol=self.atol,
                rtol=self.rtol,
                adjoint_params=tuple(p for p in self.ode_func.parameters() if p.requires_grad),
            )
            return solution[-1]

        if torch.is_inference_mode_enabled():
            # torchdiffeq's adaptive solvers can underflow the step size when
            # called inside torch.inference_mode (inference tensors lack version
            # counters). Evaluation needs no gradients, so solve in a plain
            # no-grad region with cloned, non-inference tensors instead.
            with torch.inference_mode(False), torch.no_grad():
                return solve(h_N.clone(), t_N.clone(), t_Nplus1.clone(), delta_t.clone())

        return solve(h_N, t_N, t_Nplus1, delta_t)


class MIRAOutputLayer(nn.Module):

    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()

        self.out_layer = nn.Linear(
            hidden_size,
            input_size * horizon_length,
            bias=False,
        )

    def forward(self, x):
        """

        Args:
            x (torch.FloatTensor): with shape [B, seq_len, hidden_size]

        Returns:
    `       torch.FloatTensor: final prediction with shape [B, seq_len, input_size]
        """
        return self.out_layer(x)


class SoterForPrediction(MIRAPreTrainedModel, MIRAGenerationMixin):
    def __init__(self, config: SoterConfig):
        config.horizon_lengths = [1]
        super().__init__(config)
        self.config = config

        # Auxiliary MoE balancing losses are disabled: SOTER uses deterministic
        # PSD-based routing (see SoterConfig).
        self.apply_aux_loss = getattr(config, "apply_aux_loss", False)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = getattr(config, "router_aux_loss_factor", 0.0)

        # CI/CD backbone
        self.model = SoterModel(config)

        # Continuous-dynamics extrapolation module (Neural ODE / CDE engine)
        self.use_terminal_ode = getattr(config, "use_terminal_ode", True)
        if self.use_terminal_ode:
            if not is_torchdiffeq_available:
                logger.error("use_terminal_ode=True but torchdiffeq is not installed. ODE block disabled.")
                self.ode_extrapolation_block = None
                self.use_terminal_ode = False
            else:
                self.ode_extrapolation_block = TerminalODEBlock(config)
        else:
            self.ode_extrapolation_block = None

        # Linear decoding head(s), one per horizon length
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.horizon_lengths):
            lm_head_list.append(
                MIRAOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,  # single-channel output
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)
        
        # Regression loss for time-series forecasting.
        # NOTE: avoid using attribute name `loss_function` because recent
        # transformers versions define model.loss_function for LM losses.
        self.regression_loss_fn = torch.nn.HuberLoss(reduction='none', delta=2.0)
        self.post_init()

    def get_input_embeddings(self): 
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value): 
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self): 
        return self.lm_heads[0].out_layer # Proxy

    def set_output_embeddings(self, new_layer): 
        self.lm_heads[0].out_layer = new_layer # Only first head

    def _tie_weights(self): 
        pass

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            time_values: Optional[torch.FloatTensor] = None,
            next_target_time_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None,
            observation_mask: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Backbone model output, with shape [Batch, SeqLen, Channel, Hidden]
        outputs = self.model(
            input_ids=input_ids,
            time_values=time_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_all = outputs.last_hidden_state if return_dict else outputs[0]
        batch_size, seq_length, num_channels, hidden_dim = hidden_states_all.shape
        
        # Take the last time step for extrapolation: [Batch, Channel, Hidden]
        hidden_states_last = hidden_states_all[:, -1, :, :] 

        time_values_last = None
        if time_values is not None:
            time_values_last = time_values[:, -1] # [Batch]
        
        hidden_states_for_head = hidden_states_last  # passed through directly by default when ODE is disabled

        # =========================================================================
        # Core: Terminal ODE / CDE multi-channel integration
        # - Future prediction: no control -> degenerates to a Neural ODE
        # - Mask prediction: cubic spline fitted to the observations (time, value) -> time-augmented CDE, control_path_callable provides z(t)
        # =========================================================================
        if (
            self.use_terminal_ode
            and self.ode_extrapolation_block is not None
            and next_target_time_values is not None
            and time_values_last is not None
        ):
            h_N_flat = hidden_states_last.reshape(batch_size * num_channels, hidden_dim)

            if time_values_last.dim() == 0:
                time_values_last = time_values_last.expand(batch_size)
            if next_target_time_values.dim() == 0:
                next_target_time_values = next_target_time_values.expand(batch_size)
            if time_values_last.dim() == 2:
                time_values_last = time_values_last.squeeze(-1)
            if next_target_time_values.dim() == 2:
                next_target_time_values = next_target_time_values.squeeze(-1)

            t_N_flat = time_values_last.repeat_interleave(num_channels, dim=0)
            t_Nplus1_flat = next_target_time_values.repeat_interleave(num_channels, dim=0)

            control_path_callable = None
            if getattr(self.config, "cde_control_dim", 0) > 0 and observation_mask is not None and input_ids is not None and time_values is not None:
                obs_values = input_ids if input_ids.dim() == 3 else input_ids.unsqueeze(-1)
                if obs_values.shape[1] == time_values.shape[1] and observation_mask.shape[0] == time_values.shape[0]:
                    control_path_callable = build_spline_control_callable(
                        time_values,
                        obs_values,
                        observation_mask,
                        device=input_ids.device,
                        dtype=hidden_states_last.dtype,
                    )

            h_extrapolated_flat = self.ode_extrapolation_block(
                h_N=h_N_flat,
                t_N=t_N_flat,
                t_Nplus1=t_Nplus1_flat,
                control_path_callable=control_path_callable,
                num_channels=num_channels,
            )

            hidden_states_for_head = h_extrapolated_flat.view(batch_size, num_channels, hidden_dim)

        # Add the sequence dimension [Batch, 1, Channel, Hidden] to prepare for decoding
        hidden_states = hidden_states_for_head.unsqueeze(1) 

        predictions = None
        loss = None
        aux_loss = None

        if labels is not None:
            # Ensure labels and masks match the dimensions: [Batch, 1, Channel]
            if labels.dim() == 2: # [Batch, Seq]
                labels = labels.unsqueeze(-1)

            labels = labels[:, -1:, :]
            if loss_masks is not None:
                if loss_masks.dim() == 2:
                    loss_masks = loss_masks.unsqueeze(-1)
                loss_masks = loss_masks[:, -1:, :]
            
            ar_loss = 0.0
            for lm_head, horizon_length in zip(self.lm_heads, self.config.horizon_lengths):
                # Prediction: [Batch, 1, Channel, InputSize] -> squeeze -> [Batch, 1, Channel] (since InputSize=1)
                one_predictions = lm_head(hidden_states).squeeze(-1)
                
                # Strict masked regression error computation
                one_loss = self.calc_ar_loss(one_predictions, labels, loss_masks, horizon_length)
                ar_loss += one_loss
                if predictions is None:
                    predictions = one_predictions
            loss = ar_loss / len(self.config.horizon_lengths)
            
            # (Aux loss is already intercepted, so this if-branch is never entered in theory; purely a defensive design)
            if self.apply_aux_loss:
                pass 
                
        else:
            lm_head = self.lm_heads[0]
            predictions = lm_head(hidden_states).squeeze(-1)

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss, aux_loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        if len(labels.shape) == 2:
            labels.unsqueeze_(dim=-1)
            # enable model parallelism
            labels = labels.to(predictions.device)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks.unsqueeze_(dim=-1)
            # enable model parallelism
            loss_masks = loss_masks.to(predictions.device)

        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(batch_size, seq_len, horizon_length, -1)

            # pad to the same length with predictions
            # shape -> [B, input_size, seq_len + horizon_length -1]
            labels = F.pad(labels.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

            # shape -> [B, input_size, seq_len, horizon_length]
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)

            if loss_masks is not None:
                # pad to the same length with predictions
                loss_masks = F.pad(loss_masks.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

                loss_masks = loss_masks.unfold(dimension=-1, size=horizon_length, step=1)
                loss_masks = loss_masks.permute(0, 2, 3, 1)

        else:
            shift_predictions = predictions
            shift_labels = labels

        # Calculate loss with mask
        losses = self.regression_loss_fn(shift_predictions, shift_labels)
        if loss_masks is not None:
            losses = losses * loss_masks
            # loss = losses.sum() / loss_masks.sum()
            loss = losses.sum() / (loss_masks.sum() * shift_predictions.shape[-1])
        else:
            loss = torch.mean(losses)
        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.FloatTensor,
        time_values: Optional[torch.FloatTensor] = None,
        next_target_time_values: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):

        # Handle KV-cache slicing and alignment with time_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                keep_len = attention_mask.shape[1] - past_length
                input_ids = input_ids[:, -keep_len:]
                # Slice time_values consistently with input_ids
                if time_values is not None and time_values.shape[1] > keep_len:
                    time_values = time_values[:, -keep_len:]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                if time_values is not None and time_values.shape[1] > past_length:
                    time_values = time_values[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # Compute position_ids (for standard RoPE)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if getattr(self.config, "time_aware_rotary", False):
            if time_values is None:
                raise ValueError(
                    "`time_values` is required for CT-RoPE during generation."
                )

            # Ensure time_values and input_ids lengths match
            if time_values.shape[1] != input_ids.shape[1]:
                if time_values.shape[1] > input_ids.shape[1]:
                    time_values = time_values[:, -input_ids.shape[1]:]
                else:
                    raise ValueError(
                        f"time_values length ({time_values.shape[1]}) is shorter than "
                        f"input_ids length ({input_ids.shape[1]})."
                    )

        model_inputs["time_values"] = time_values


        if next_target_time_values is not None:
            if next_target_time_values.dim() == 1:
                next_target_time_values = next_target_time_values.unsqueeze(1)  # [B] -> [B, 1]
            elif next_target_time_values.dim() == 2 and next_target_time_values.shape[1] != 1:
                next_target_time_values = next_target_time_values[:, -1:]
        else:
            if time_values is not None and self.use_terminal_ode:
                time_step = kwargs.get("time_step", None)
                if time_step is None and time_values.shape[1] > 1:
                    time_diffs = time_values[:, 1:] - time_values[:, :-1]
                    time_step = time_diffs.mean(dim=1, keepdim=True)
                elif time_step is None:
                    time_step = torch.ones(time_values.shape[0], 1, device=time_values.device)

                next_target_time_values = time_values[:, -1:] + time_step

        model_inputs["next_target_time_values"] = next_target_time_values

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        if kwargs.get("observation_mask") is not None:
            model_inputs["observation_mask"] = kwargs["observation_mask"]

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
