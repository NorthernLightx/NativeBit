"""Default NativeBit training configuration."""


class DefaultConfig:
    # Model (LLaMA-style: RoPE, RMSNorm, SwiGLU)
    n_layers: int = 20
    n_embd: int = 192
    n_head: int = 4
    ffn_hidden: int = 768
    context_len: int = 256
    vocab_size: int = 50257  # tiktoken gpt2 vocab

    # NativeBit quantization
    block_size: int = 64     # weights per codebook block
    n_codebook: int = 8      # entries per codebook (8 = 3-bit)

    # Training
    batch_size: int = 8
    lr: float = 1.5e-3
    codebook_lr: float = 1.5e-4
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 25
    log_every: int = 50
    weight_decay: float = 0.01

    # Data
    dataset: str = "wikitext-2"

    # Reproducibility
    seed: int = 42
