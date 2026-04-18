"""Unit tests for VQ-VAE-style commitment loss on NativeBit weights.

Verifies:
  1. compute_quant_reg returns 0 when w == Q(w)
  2. compute_quant_reg returns known value when w is offset from Q(w)
  3. Gradient flows to w (pulls toward nearest entry) but NOT to codebook
  4. Multi-layer: averaged correctly across all NB weights
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from nativebit_jax.layers import compute_quant_reg


def _make_params(w, cb):
    """Wrap in params structure matching _extract_layer_arrays's search pattern."""
    return {"params": {"layer": {"weight": w, "codebook": cb}}}


def test_zero_when_weights_on_codebook():
    # Weights all on codebook entries → reg = 0
    # block_size = 2, 2 entries per block
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)  # 1 block, entries 0 and 1
    w = jnp.array([[0.0, 1.0]], dtype=jnp.float32)   # exactly on entries
    reg = compute_quant_reg(_make_params(w, cb))
    assert float(reg) < 1e-7, f"Expected 0, got {float(reg)}"
    print(f"  [PASS] Zero reg when weights == codebook entries (got {float(reg):.2e})")


def test_known_offset():
    # Each weight offset by 0.3 from nearest entry. sum = 0.09 + 0.09 = 0.18.
    # n_layers = 1, so reg = 0.18.
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    w = jnp.array([[0.3, 0.7]], dtype=jnp.float32)
    reg = compute_quant_reg(_make_params(w, cb))
    expected = 0.18
    assert abs(float(reg) - expected) < 1e-6, f"Expected {expected}, got {float(reg)}"
    print(f"  [PASS] Known offset: reg = {float(reg):.6f} (expected {expected})")


def test_gradient_pulls_w_toward_entry():
    # Single layer; grad per weight = 2(w_i - nearest_entry) / n_layers = 2*drift / 1
    # w_0 = 0.3 -> nearest = 0 -> grad = 2*0.3 = 0.6
    # w_1 = 0.7 -> nearest = 1 -> grad = 2*(0.7-1) = -0.6
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    w = jnp.array([[0.3, 0.7]], dtype=jnp.float32)

    def reg_fn(w):
        return compute_quant_reg(_make_params(w, cb))

    grad = jax.grad(reg_fn)(w)
    expected = jnp.array([[0.6, -0.6]], dtype=jnp.float32)
    diff = jnp.abs(grad - expected).max()
    assert float(diff) < 1e-6, f"grad={grad}, expected={expected}"
    print(f"  [PASS] Gradient pulls w toward nearest entry: grad = {grad.tolist()}")


def test_gradient_is_zero_on_codebook():
    # Codebook should not receive gradient (stop_gradient inside compute_quant_reg)
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    w = jnp.array([[0.3, 0.7]], dtype=jnp.float32)

    def reg_fn(cb):
        return compute_quant_reg(_make_params(w, cb))

    grad_cb = jax.grad(reg_fn)(cb)
    assert float(jnp.abs(grad_cb).max()) < 1e-7, (
        f"Codebook got non-zero grad: {grad_cb}")
    print(f"  [PASS] Codebook gradient is zero (max |grad| = "
          f"{float(jnp.abs(grad_cb).max()):.2e})")


def test_multi_layer_averaged():
    # Two layers. Sum of sq errors: 0.09 + 0.09 + 0.01 = 0.19.
    # n_layers = 2, so reg = 0.19 / 2 = 0.095.
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    w1 = jnp.array([[0.3, 0.3]], dtype=jnp.float32)  # 2 weights, err^2=0.09 each
    w2 = jnp.array([[0.1]], dtype=jnp.float32)       # 1 weight, err^2=0.01

    params = {"params": {
        "layer1": {"weight": w1, "codebook": cb},
        "layer2": {"weight": w2, "codebook": jnp.array([[0.0, 1.0]], dtype=jnp.float32)},
    }}
    reg = compute_quant_reg(params)
    expected = 0.19 / 2.0
    assert abs(float(reg) - expected) < 1e-6, f"Expected {expected}, got {float(reg)}"
    print(f"  [PASS] Multi-layer normalized by n_layers: reg = {float(reg):.6f} "
          f"(expected {expected:.6f})")


def test_no_nb_layers_returns_zero():
    # Params with only float layers (no codebook) → reg = 0
    params = {"params": {"embedding": jnp.ones((10, 5))}}
    reg = compute_quant_reg(params)
    assert float(reg) == 0.0, f"Expected 0, got {float(reg)}"
    print(f"  [PASS] No NB layers -> reg = 0")


if __name__ == "__main__":
    print("Testing compute_quant_reg:")
    test_zero_when_weights_on_codebook()
    test_known_offset()
    test_gradient_pulls_w_toward_entry()
    test_gradient_is_zero_on_codebook()
    test_multi_layer_averaged()
    test_no_nb_layers_returns_zero()
    print("\nAll tests passed.")
