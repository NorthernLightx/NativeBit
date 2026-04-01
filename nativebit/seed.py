"""Reproducibility utilities — fix all random seeds from day one."""

import os
import random

import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for torch, numpy, python random, and XLA."""
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # XLA / TPU
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except (ImportError, Exception):
        pass

    # Numpy (if available)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
