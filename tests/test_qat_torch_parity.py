"""PyTorch QAT port must match the JAX reference bit-for-bit (within fp32 tolerance).

The QAT recipe (canonical VQ-VAE EMA + commitment loss) was validated on TPU
via the JAX backend. The local scaling-curve runs use the PyTorch port, so
these tests pin the port to the JAX semantics on identical inputs:

  1. canonical EMA update — codebook, ema_N, ema_s after K iterations
  2. commitment loss value and its gradient target (latent weights only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from nativebit.codebook_utils import ema_update_canonical
from nativebit.layers import NativeBitLinear, compute_quant_reg
from nativebit_jax import layers as nb_jax_layers
from nativebit_jax.layers import (
    requantize_params, init_canonical_ema_state,
    compute_quant_reg as compute_quant_reg_jax,
)

NUM_BLOCKS, BLOCK_SIZE, N_ENTRIES = 4, 64, 8
DECAY = 0.99


def _reset_jitted_caches():
    nb_jax_layers._requantize_all_jitted = None
    nb_jax_layers._requantize_all_jitted_canonical = None


def _random_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(NUM_BLOCKS, BLOCK_SIZE, generator=g)
    cb = torch.sort(torch.randn(NUM_BLOCKS, N_ENTRIES, generator=g), dim=1).values
    return w, cb


def _jax_params(w: torch.Tensor, cb: torch.Tensor) -> dict:
    """Minimal tree matching the layer.weight + layer.codebook pattern."""
    jw = jnp.array(w.numpy())
    jcb = jnp.array(cb.numpy())
    return {
        "params": {"layer": {"weight": jw, "codebook": jcb}},
        "cache": {"layer": {"qw_delta": jnp.zeros_like(jw, dtype=jnp.bfloat16)}},
    }


def test_canonical_ema_matches_jax_over_iterations():
    w, cb = _random_inputs(seed=1)

    # --- JAX reference: 5 canonical updates, weights perturbed each round ---
    _reset_jitted_caches()
    params = init_canonical_ema_state(_jax_params(w, cb))
    torch_cb = cb.clone()
    torch_N = torch.ones(NUM_BLOCKS, N_ENTRIES)
    torch_s = cb.clone().float()

    for i in range(5):
        params, _ = requantize_params(params, ema_decay=DECAY,
                                      use_canonical_ema=True)
        ema_update_canonical(torch_cb, torch_N, torch_s, w + 0.01 * i, DECAY)
        # keep JAX weights in sync with the same perturbation
        params["params"]["layer"]["weight"] = jnp.array((w + 0.01 * (i + 1)).numpy())

        jax_cb = torch.tensor(jax.device_get(
            params["params"]["layer"]["codebook"])).float()
        diff = (jax_cb - torch_cb).abs().max().item()
        assert diff < 1e-4, f"iter {i}: codebook diverged, max diff {diff:.2e}"

    jax_N = torch.tensor(jax.device_get(
        params["cache"]["layer"]["ema_N"])).float()
    jax_s = torch.tensor(jax.device_get(
        params["cache"]["layer"]["ema_s"])).float()
    assert (jax_N - torch_N).abs().max() < 1e-4, "ema_N diverged"
    assert (jax_s - torch_s).abs().max() < 1e-4, "ema_s diverged"


def test_commitment_loss_matches_jax():
    w, cb = _random_inputs(seed=2)

    jax_val = float(compute_quant_reg_jax(
        {"params": {"layer": {"weight": jnp.array(w.numpy()),
                              "codebook": jnp.array(cb.numpy())}}}))

    # Torch path goes through a real layer (same shapes, injected tensors)
    layer = NativeBitLinear(BLOCK_SIZE, NUM_BLOCKS, bias=False,
                            block_size=BLOCK_SIZE, n_entries=N_ENTRIES)
    layer.weight.data.copy_(w.view(NUM_BLOCKS, BLOCK_SIZE))
    layer.codebook.data.copy_(cb)
    torch_val = compute_quant_reg([layer]).item()

    assert abs(jax_val - torch_val) < 1e-4 * max(abs(jax_val), 1.0), \
        f"commitment mismatch: jax={jax_val}, torch={torch_val}"


def test_commitment_grad_flows_to_weight_not_codebook():
    w, cb = _random_inputs(seed=3)
    layer = NativeBitLinear(BLOCK_SIZE, NUM_BLOCKS, bias=False,
                            block_size=BLOCK_SIZE, n_entries=N_ENTRIES)
    layer.weight.data.copy_(w.view(NUM_BLOCKS, BLOCK_SIZE))
    layer.codebook.data.copy_(cb)

    compute_quant_reg([layer]).backward()
    assert layer.weight.grad is not None
    assert layer.codebook.grad is None
    # d/dw of min_j (w - sg(cb_j))^2 = 2 (w - cb_nearest) / n_layers
    w_blocks, _ = layer._weight_blocks()
    d = (w_blocks.float().unsqueeze(-1) - cb.float().unsqueeze(1)).square()
    nearest = cb.float()[torch.arange(NUM_BLOCKS).unsqueeze(1), d.argmin(dim=-1)]
    expected = 2 * (w_blocks.float() - nearest)
    assert torch.allclose(layer.weight.grad.view(NUM_BLOCKS, BLOCK_SIZE),
                          expected, atol=1e-5)
