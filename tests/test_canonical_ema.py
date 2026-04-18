"""Verify canonical VQ-VAE EMA:
  N_new = a*N_old + (1-a)*batch_count
  s_new = a*s_old + (1-a)*batch_sum
  codebook = s_new / N_new

Compares to the legacy EMA-of-means formulation on a contrived scenario
with skewed batch counts to show the difference.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from nativebit_jax import layers as nb_layers
from nativebit_jax.layers import requantize_params, init_canonical_ema_state


def _reset_jitted_caches():
    """Module-level caches are keyed to the first-seen model shape;
    reset them between tests that use different shapes."""
    nb_layers._requantize_all_jitted = None
    nb_layers._requantize_all_jitted_canonical = None


def _make_params(w, cb):
    """Build a minimal params+cache tree with the 'layer.weight + layer.codebook'
    pattern that `_extract_layer_arrays` looks for."""
    return {
        "params": {"layer": {"weight": w, "codebook": cb}},
        "cache": {"layer": {
            "qw_delta": jnp.zeros_like(w, dtype=jnp.bfloat16),
        }},
    }


def test_canonical_preserves_init_codebook_on_first_call():
    # If weights already sit exactly on codebook entries, the first canonical
    # update should leave codebook unchanged (s/N = cb after init).
    cb = jnp.array([[-1.0, 0.0, 1.0]], dtype=jnp.float32)
    # 6 weights, exactly 2 per entry
    w = jnp.array([[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32)

    _reset_jitted_caches()
    params = _make_params(w, cb)
    params = init_canonical_ema_state(params)

    new_params, _ = requantize_params(params, ema_decay=0.9, use_canonical_ema=True)
    new_cb = np.array(new_params["params"]["layer"]["codebook"])
    assert np.allclose(new_cb, cb, atol=1e-5), f"cb drift: {new_cb} vs {cb}"
    print(f"  [PASS] First canonical update preserves on-entry codebook "
          f"(max drift={np.max(np.abs(new_cb - cb)):.2e})")


def test_canonical_converges_count_weighted():
    # Two batches, very different counts. After canonical update:
    #   s = 0.9*cb + 0.1*(sum_batch)
    #   N = 0.9*1 + 0.1*count_batch
    #   new_cb = s / N
    # Compare to EMA-of-means (legacy) which equally weights a 1-sample batch
    # and a 100-sample batch.
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    # Assign most weights to entry 0 (around 0.1) and one to entry 1 (near 1.1):
    # 100 samples at 0.1 -> assigned to entry 0; 1 sample at 0.9 -> entry 1.
    w_block = jnp.concatenate([jnp.full((100,), 0.1),
                               jnp.full((1,), 0.9)])
    w = w_block.reshape(1, -1)  # (out=1, in=101)
    # Block size = 101 so we get exactly one block.

    # Build params where num_blocks=1 is enforced by matching sizes
    total = w.size
    cb_fit = jnp.array([[0.0, 1.0]], dtype=jnp.float32)  # (num_blocks=1, K=2)
    _reset_jitted_caches()
    params = _make_params(w, cb_fit)
    params = init_canonical_ema_state(params)

    alpha = 0.9
    # Apply one canonical update
    new_params, _ = requantize_params(params, ema_decay=alpha, use_canonical_ema=True)
    new_cb = np.array(new_params["params"]["layer"]["codebook"])

    # Entry 0: 100 samples at 0.1. Expected:
    #   N_new = 0.9*1 + 0.1*100 = 10.9
    #   s_new = 0.9*0.0 + 0.1*10.0 = 1.0
    #   cb_new = 1.0 / 10.9 ~ 0.0917
    # Entry 1: 1 sample at 0.9. Expected:
    #   N_new = 0.9*1 + 0.1*1 = 1.0
    #   s_new = 0.9*1.0 + 0.1*0.9 = 0.99
    #   cb_new = 0.99 / 1.0 = 0.99
    expected = np.array([[1.0 / 10.9, 0.99]], dtype=np.float32)
    assert np.allclose(new_cb, expected, atol=1e-4), (
        f"got {new_cb}, expected {expected}")
    print(f"  [PASS] Canonical codebook after skewed batch: {new_cb.tolist()} "
          f"(expected {expected.tolist()})")


def test_legacy_gives_different_result_on_skewed_batch():
    # Legacy EMA-of-means should blend both batch means equally, ignoring count.
    # Entry 0: cb_new = 0.9*0 + 0.1*0.1 = 0.01
    # Entry 1: cb_new = 0.9*1.0 + 0.1*0.9 = 0.99  (count=1, same formula)
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    w_block = jnp.concatenate([jnp.full((100,), 0.1), jnp.full((1,), 0.9)])
    w = w_block.reshape(1, -1)
    _reset_jitted_caches()
    params = _make_params(w, cb)

    new_params, _ = requantize_params(params, ema_decay=0.9, use_canonical_ema=False)
    new_cb = np.array(new_params["params"]["layer"]["codebook"])

    expected = np.array([[0.01, 0.99]], dtype=np.float32)
    assert np.allclose(new_cb, expected, atol=1e-4)

    # Crucially, entry 0 moves MORE in legacy than canonical:
    # canonical: ~0.092 (count-weighted by 100 samples pulls harder)
    # legacy:    0.01  (1-sample batch and 100-sample batch treated equal)
    print(f"  [PASS] Legacy codebook: {new_cb.tolist()} "
          f"(expected {expected.tolist()})")
    print(f"          Legacy entry-0 step: 0.01  vs  canonical entry-0 step: "
          f"~0.092  (canonical pulls harder on well-populated entry)")


def test_dead_entry_handling():
    # If no weight gets assigned to an entry, N stays decayed, s stays decayed,
    # and (for N > threshold) derived codebook is s/N ~= old cb * (a/a) = old.
    cb = jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32)
    # All weights near entry 1 (0.5) - entries 0 and 2 get no assignments.
    w = jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float32)

    _reset_jitted_caches()
    params = _make_params(w, cb)
    params = init_canonical_ema_state(params)

    new_params, _ = requantize_params(params, ema_decay=0.9, use_canonical_ema=True)
    new_cb = np.array(new_params["params"]["layer"]["codebook"])

    # After one update with all 3 samples on entry 1:
    # Entry 0: no assignment. N=0.9*1=0.9. s=0.9*0=0. cb=0/0.9=0 (stays at 0).
    # Entry 1: 3 samples at 0.5. N=0.9*1+0.1*3=1.2. s=0.9*0.5+0.1*1.5=0.6. cb=0.5.
    # Entry 2: no assignment. N=0.9*1=0.9. s=0.9*1=0.9. cb=0.9/0.9=1.0 (unchanged).
    expected = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert np.allclose(new_cb, expected, atol=1e-5), (
        f"got {new_cb}, expected {expected}")
    print(f"  [PASS] Dead entries preserved by canonical EMA: "
          f"{new_cb.tolist()}")


def test_many_iterations_converge_to_batch_means():
    # With constant batch and stationary weights, both formulations converge
    # to the same limit: the cluster means.
    cb = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    # 8 weights: 5 near 0.2 (entry 0), 3 near 0.8 (entry 1).
    w_block = jnp.concatenate([jnp.full((5,), 0.2), jnp.full((3,), 0.8)])
    w = w_block.reshape(1, -1)

    for ema_name, canonical in [("canonical", True), ("legacy", False)]:
        _reset_jitted_caches()
        params = _make_params(w, cb)
        if canonical:
            params = init_canonical_ema_state(params)
        for _ in range(500):
            params, _ = requantize_params(params, ema_decay=0.99,
                                          use_canonical_ema=canonical)
        final_cb = np.array(params["params"]["layer"]["codebook"])
        # Should converge near batch cluster means: [0.2, 0.8]
        expected = np.array([[0.2, 0.8]], dtype=np.float32)
        diff = float(np.max(np.abs(final_cb - expected)))
        # Canonical converges slightly faster; both are near [0.2, 0.8].
        assert diff < 5e-3, f"{ema_name}: got {final_cb}, expected {expected}"
        print(f"  [PASS] {ema_name} converges to batch means after 500 iters "
              f"(max diff={diff:.2e})")


if __name__ == "__main__":
    print("Testing canonical VQ-VAE EMA:")
    test_canonical_preserves_init_codebook_on_first_call()
    test_canonical_converges_count_weighted()
    test_legacy_gives_different_result_on_skewed_batch()
    test_dead_entry_handling()
    test_many_iterations_converge_to_batch_means()
    print("\nAll tests passed.")
