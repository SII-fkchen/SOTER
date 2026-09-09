from typing import List, Optional
from transformers import PretrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

class SoterConfig(PretrainedConfig):
    model_type = "soter"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 256,
            intermediate_size: int = 512,
            num_hidden_layers: int = 6,
            num_attention_heads: int = 4,
            num_key_value_heads: int = None,
            hidden_act: str = "silu",

            # CI = Channel Independence (low-level): per-channel temporal modeling with CT-RoPE.
            # CD = Channel Dependence (high-level): cross-channel attention at the same time step.
            ci_layers: List[int] = [0, 1, 2, 3, 4],
            cd_layers: List[int] = [5],

            num_experts: int = 6,
            num_shared_experts: int = 1,
            num_experts_per_tok: int = 2,
            dft_router_impl: str = "for_loop",

            use_terminal_ode: bool = True,
            ode_func_hidden_dims: List[int] = [128, 128],
            ode_func_activation: str = "silu",
            ode_solver_method: str = "dopri5",
            ode_solver_atol: float = 1e-6,
            ode_solver_rtol: float = 1e-6,
            ode_func_use_time: bool = True,
            ode_activation_threshold: float = 1.0,
            # CDE: control path dim for mask prediction (cubic spline). 0 = pure ODE (future prediction).
            cde_control_dim: int = 0,

            time_aware_rotary: bool = True,
            rope_theta: int = 10000,
            time_scale: float = 1.0,
            max_position_embeddings: int = 32768,

            apply_aux_loss: bool = False,
            router_aux_loss_factor: float = 0.0,
            use_cache: bool = False,
            use_dense: bool = False,
            horizon_lengths: List[int] = 1,

            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            attention_dropout: float = 0.0,
            tie_word_embeddings: bool = False,
            gradient_checkpointing_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act

        self.ci_layers = ci_layers
        self.cd_layers = cd_layers
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.dft_router_impl = dft_router_impl

        self.rope_theta = rope_theta
        self.time_aware_rotary = time_aware_rotary
        self.time_scale = time_scale 
        self.use_terminal_ode = use_terminal_ode
        self.ode_func_hidden_dims = ode_func_hidden_dims
        self.ode_func_activation = ode_func_activation
        self.ode_solver_method = ode_solver_method
        self.ode_solver_atol = ode_solver_atol
        self.ode_solver_rtol = ode_solver_rtol
        self.ode_func_use_time = ode_func_use_time
        self.ode_activation_threshold = ode_activation_threshold
        self.cde_control_dim = cde_control_dim

        # disable aux loss
        if apply_aux_loss:
            logger.warning("Overriding apply_aux_loss=True -> False for SOTER deterministic PSD routing.")
        self.apply_aux_loss = False
        self.router_aux_loss_factor = 0.0
        self.use_cache = use_cache
        self.use_dense = use_dense
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs if gradient_checkpointing_kwargs is not None else {"use_reentrant": True}

        kwargs.pop('tie_word_embeddings', None)
        self._attn_implementation = kwargs.pop('_attn_implementation', 'eager')
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )