"""Local-GPU (RTX 3070, 8 GB) configs for the scaling-curve runs.

Middle point between tpu-small (~48M total) and tpu-medium (~152M total).
Batch 8 + --grad-accum 8 gives the same 32K tokens/step as the TPU configs.
"""


class Local76MConfig:
    """~76M total params (embeddings included). Fits 8 GB at batch 8."""
    n_layers: int = 12
    n_embd: int = 512
    n_head: int = 8
    ffn_hidden: int = 2048
    context_len: int = 512
    vocab_size: int = 50257

    block_size: int = 64
    n_codebook: int = 8

    batch_size: int = 8
    lr: float = 8e-4          # between tpu-small (1e-3) and tpu-medium (6e-4)
    codebook_lr: float = 8e-5
    max_steps: int = 10000    # x 32K tok/step = 328M tokens, same as tpu-small
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    log_every: int = 50
    weight_decay: float = 0.01

    dataset: str = "wikitext-103"
    seed: int = 42
