"""125M param config for publishable benchmarks (GPT-2 Small equivalent)."""


class LargeConfig:
    # Model — GPT-2 Small architecture
    n_layers: int = 12
    n_embd: int = 768
    n_head: int = 12
    ffn_hidden: int = 3072
    context_len: int = 512
    vocab_size: int = 50257

    # NativeBit (best settings from scaleup experiments)
    block_size: int = 128
    block_size_attn: int = 32
    n_codebook: int = 4       # 2-bit
    use_ema: bool = True
    ema_decay: float = 0.9974
    factored_init: bool = True
    requant_every: int = 10

    # Training
    batch_size: int = 16
    lr: float = 6e-4
    codebook_lr: float = 1e-3
    max_steps: int = 30000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 5000
    weight_decay: float = 0.05

    # Gradient checkpointing — essential for 8GB GPU
    grad_checkpoint: bool = True

    # Data
    dataset: str = "wikitext-103"
    seed: int = 42

    # Disabled features
    grad_accum_steps: int = 1
    n_codebooks: int = 1
    factored_codebook: bool = False
    learned_distance: bool = False
    entropy_lambda: float = 0.0
    entropy_temperature: float = 0.01
    progressive: bool = False
    merge_util_threshold: float = 0.02
    merge_dist_threshold = None
    merge_steps = None
    tau_start: float = 0.0
    tau_end: float = 0.01
    tau_anneal_steps: int = 3000
    quantize_mode: str = "ste"
    diversity_lambda: float = 0.0
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 10.0
    delay_quant_steps: int = 0
    distill_alpha: float = 0.0
    distill_temp: float = 2.0
    quant_warmup_steps: int = 0
    quant_dropout: float = 0.0


class LargeProgressiveConfig(LargeConfig):
    """Progressive merge: 16 entries -> 4 entries over 30k steps.

    Merge spacing increases for later merges (each removes a larger
    fraction of remaining entries, needs more recovery time).
    Early: 1000-step gaps (16->10, small perturbation)
    Mid:   1500-step gaps (10->6, moderate)
    Late:  2000-2500 step gaps (6->4, major disruption)
    """
    n_codebook: int = 16      # Start 4-bit
    merge_schedule: list = [3000, 4000, 5000, 6000, 7500, 9000,
                            10500, 12500, 15000, 17500, 20000, 23000]
    merge_min_active: int = 4


class LargeFloatConfig(LargeConfig):
    """Float baseline — no quantization."""
    use_ema: bool = False
    requant_every: int = 1
