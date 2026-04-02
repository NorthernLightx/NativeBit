"""TPU training configurations — scaled up for Cloud TPU hardware.

TPU v5e: 16GB HBM per chip, bf16 native, ~197 TFLOPS
TPU v6e: 32GB HBM per chip, bf16 native, ~918 TFLOPS
TPU v4:  32GB HBM per chip, bf16 native, ~275 TFLOPS
"""


class TPUSmallConfig:
    """~25M params — quick TPU smoke test (runs in minutes)."""
    n_layers: int = 12
    n_embd: int = 384
    n_head: int = 6
    ffn_hidden: int = 1536
    context_len: int = 512
    vocab_size: int = 50257

    block_size: int = 64
    n_codebook: int = 8

    batch_size: int = 64       # TPU can handle much larger batches
    lr: float = 1e-3
    codebook_lr: float = 1e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 25
    log_every: int = 50
    weight_decay: float = 0.01

    dataset: str = "wikitext-2"
    seed: int = 42


class TPUMediumConfig:
    """~125M params — single TPU chip (v5e/v6e). Paper-quality results."""
    n_layers: int = 12
    n_embd: int = 768
    n_head: int = 12
    ffn_hidden: int = 3072
    context_len: int = 1024
    vocab_size: int = 50257

    block_size: int = 128
    n_codebook: int = 8

    batch_size: int = 32       # 32 * 1024 = 32k tokens/step
    lr: float = 6e-4
    codebook_lr: float = 6e-5
    max_steps: int = 50000
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 50
    log_every: int = 100
    weight_decay: float = 0.1
    delay_quant_steps: int = 500  # Train as float for first 500 steps
    ema_decay: float = 0.999
    requantize_every: int = 10

    dataset: str = "wikitext-103"
    seed: int = 42


class TPULargeConfig:
    """~350M params — multi-chip (v6e slice). Full benchmark scale."""
    n_layers: int = 24
    n_embd: int = 1024
    n_head: int = 16
    ffn_hidden: int = 4096
    context_len: int = 1024
    vocab_size: int = 50257

    block_size: int = 128
    n_codebook: int = 8

    batch_size: int = 16       # with remat; increase if memory allows
    lr: float = 3e-4
    codebook_lr: float = 3e-5
    max_steps: int = 10000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 100
    log_every: int = 100
    weight_decay: float = 0.1
    delay_quant_steps: int = 500
    ema_decay: float = 0.999
    requantize_every: int = 10

    dataset: str = "wikitext-103"
    seed: int = 42


class TPUXLConfig:
    """~1.7B params — full TPU v6e slice (64 chips). Flagship experiments."""
    n_layers: int = 24
    n_embd: int = 2048
    n_head: int = 16
    ffn_hidden: int = 8192
    context_len: int = 2048
    vocab_size: int = 50257

    block_size: int = 256
    n_codebook: int = 8

    batch_size: int = 16       # per-device; total = 16 * n_devices
    lr: float = 2e-4
    codebook_lr: float = 2e-5
    max_steps: int = 200000
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 200
    log_every: int = 100
    weight_decay: float = 0.1

    dataset: str = "wikitext-103"
    seed: int = 42


class TPU2BConfig:
    """~2.2B params — matches BitNet b1.58-2B-4T scale. FSDP on v6e-8.

    Same hidden/FFN dims as BitNet (2560/6912). 26 layers instead of 30
    because we use MHA (BitNet uses GQA 20/5), matching total param count.
    """
    n_layers: int = 26
    n_embd: int = 2560
    n_head: int = 20
    ffn_hidden: int = 6912
    context_len: int = 2048
    vocab_size: int = 50257

    block_size: int = 128
    n_codebook: int = 8

    batch_size: int = 32       # 4 per chip on v6e-8 with FSDP
    lr: float = 1.5e-4
    codebook_lr: float = 1.5e-5
    max_steps: int = 20000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 500
    log_every: int = 100
    weight_decay: float = 0.1
    delay_quant_steps: int = 500
    ema_decay: float = 0.999
    requantize_every: int = 200
    checkpoint_every: int = 1000

    dataset: str = "wikitext-103"
    seed: int = 42
