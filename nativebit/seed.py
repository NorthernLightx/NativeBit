"""Reproducibility utilities — fix all random seeds from day one."""

import os
import random

import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for torch, numpy, and python random."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed numpy if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
