"""Device abstraction — unified API for CUDA, TPU (XLA), and CPU.

All device-specific calls go through this module so the rest of the
codebase stays device-agnostic.
"""

import torch

# Lazy XLA imports — only loaded when actually on TPU
_xla_available = None


def _check_xla() -> bool:
    global _xla_available
    if _xla_available is None:
        try:
            import torch_xla  # noqa: F401
            _xla_available = True
        except ImportError:
            _xla_available = False
    return _xla_available


def get_device() -> torch.device:
    """Auto-detect best available device: TPU > CUDA > CPU."""
    if _check_xla():
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_tpu(device: torch.device | None = None) -> bool:
    """Check if device is a TPU/XLA device."""
    if device is not None:
        return device.type == "xla"
    return _check_xla()


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def mark_step():
    """XLA: trigger pending graph execution. No-op on CUDA/CPU."""
    if _check_xla():
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def optimizer_step(optimizer, barrier: bool = False):
    """Device-aware optimizer step.

    On XLA: uses xm.optimizer_step which includes mark_step().
    On CUDA/CPU: plain optimizer.step().
    """
    if _check_xla():
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer, barrier=barrier)
    else:
        optimizer.step()


def sync_device(device: torch.device):
    """Synchronize device (for timing). No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xla" and _check_xla():
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()


def amp_context(device: torch.device, dtype=None):
    """Device-agnostic autocast context manager.

    - CUDA: autocast to float16
    - TPU: no autocast — codebook quantization ops (distance, stochastic rounding)
           need fp32 precision. TPU is fast enough in fp32 and bf16 degrades
           codebook convergence.
    - CPU: no autocast (identity context)
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype or torch.float16)
    else:
        import contextlib
        return contextlib.nullcontext()


def amp_enabled(device: torch.device) -> bool:
    """Whether AMP should be used on this device."""
    return device.type in ("cuda", "xla")


def needs_grad_scaler(device: torch.device) -> bool:
    """GradScaler is only needed for CUDA fp16. TPU bf16 doesn't need it."""
    return device.type == "cuda"


def get_memory_info(device: torch.device) -> dict:
    """Get device memory stats (best-effort)."""
    if device.type == "cuda":
        return {
            "peak_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            "current_mb": torch.cuda.memory_allocated(device) / 1024 / 1024,
        }
    # TPU and CPU: no direct memory query API
    return {}


def device_name(device: torch.device) -> str:
    """Human-readable device name."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "xla":
        return "Cloud TPU (XLA)"
    return "CPU"
