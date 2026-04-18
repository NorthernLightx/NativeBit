"""Verify the JAX attention fix: cross-position, not cross-head.

Tests:
1. Transposed dot_product_attention matches manual einsum (the fix is correct)
2. Original dot_product_attention does NOT match (the bug exists)
3. Context sensitivity: full context vs single token must differ

Run: python tests/test_attention_fix.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np


def manual_attention(q, k, v):
    """Standard cross-position causal attention via einsum. Ground truth."""
    _, _, T, D = q.shape
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q * scale, k)
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    attn = jnp.where(causal[None, None], attn, -1e30)
    probs = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', probs, v)


def transposed_dpa(q, k, v):
    """Option A fix: transpose to (B,T,H,D) for dot_product_attention."""
    return jax.nn.dot_product_attention(
        q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3), is_causal=True,
    ).transpose(0, 2, 1, 3)


def broken_dpa(q, k, v):
    """Original broken call: (B,H,T,D) directly to dot_product_attention."""
    return jax.nn.dot_product_attention(q, k, v, is_causal=True)


def test_fix_matches_manual():
    """Transposed DPA should match manual einsum attention."""
    np.random.seed(42)
    q = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))
    k = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))
    v = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))

    out_manual = manual_attention(q, k, v)
    out_fixed = transposed_dpa(q, k, v)

    max_diff = float(jnp.abs(out_manual - out_fixed).max())
    assert max_diff < 1e-5, f"Fix doesn't match manual: max_diff={max_diff}"
    print(f"  PASS: fix matches manual (max_diff={max_diff:.2e})")


def test_broken_differs():
    """Original broken call should NOT match manual attention."""
    np.random.seed(42)
    q = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))
    k = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))
    v = jnp.array(np.random.randn(2, 4, 16, 32).astype(np.float32))

    out_manual = manual_attention(q, k, v)
    out_broken = broken_dpa(q, k, v)

    max_diff = float(jnp.abs(out_manual - out_broken).max())
    assert max_diff > 0.1, f"Broken call matches manual — bug may not exist: max_diff={max_diff}"
    print(f"  PASS: broken call differs from manual (max_diff={max_diff:.2f})")


def test_context_sensitivity():
    """With correct attention, logits must depend on context (previous tokens)."""
    from nativebit_jax.model import build_model
    from configs.tpu import TPUSmallConfig

    config = TPUSmallConfig()
    model = build_model(config, use_nativebit=False)

    rng = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(rng, dummy)

    # Full context: 16 tokens
    tokens = jax.random.randint(rng, (1, 16), 0, 1000)
    logits_full = model.apply(params, tokens)[0, -1]

    # Single token (last one only)
    logits_single = model.apply(params, tokens[:, -1:])[0, -1]

    # KL divergence
    p = jax.nn.softmax(logits_full)
    q = jax.nn.softmax(logits_single)
    kl = float(jnp.sum(p * (jnp.log(p + 1e-10) - jnp.log(q + 1e-10))))

    assert kl > 0.01, f"Context sensitivity too low: KL={kl:.6f} (model ignores context)"
    print(f"  PASS: context sensitivity KL={kl:.4f} (model uses context)")


def test_real_model_shapes():
    """Test with actual model dimensions (B=1, H=20, T=5, D=128)."""
    np.random.seed(123)
    q = jnp.array(np.random.randn(1, 20, 5, 128).astype(np.float32))
    k = jnp.array(np.random.randn(1, 20, 5, 128).astype(np.float32))
    v = jnp.array(np.random.randn(1, 20, 5, 128).astype(np.float32))

    out_manual = manual_attention(q, k, v)
    out_fixed = transposed_dpa(q, k, v)

    max_diff = float(jnp.abs(out_manual - out_fixed).max())
    assert max_diff < 1e-5, f"Real-size mismatch: max_diff={max_diff}"
    print(f"  PASS: real model shapes match (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    print("Test 1: Fix matches manual attention")
    test_fix_matches_manual()

    print("Test 2: Broken call differs from manual")
    test_broken_differs()

    print("Test 3: Context sensitivity (full model)")
    test_context_sensitivity()

    print("Test 4: Real model shapes (1, 20, 5, 128)")
    test_real_model_shapes()

    print("\nAll tests passed!")
