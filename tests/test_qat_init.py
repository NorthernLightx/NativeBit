"""Verify --init-from correctly translates float (nn.Dense) checkpoint keys
into NativeBitDense layer params, with proper transpose."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

from nativebit_jax.model import build_model


def test_float_to_nb_key_translation():
    """Load a small float checkpoint into an NB model and verify most leaves get
    populated (not just a few)."""

    # Small config matching what DefaultConfig does — fast to init
    class TinyConfig:
        vocab_size = 50257
        n_layers = 2
        n_embd = 64
        n_head = 4
        ffn_hidden = 128
        context_len = 32
        block_size = 32
        n_codebook = 8
        seed = 42

    # Build a float model and get its param structure
    float_model = build_model(TinyConfig(), use_nativebit=False)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 32), dtype=jnp.int32)
    float_params = float_model.init(rng, dummy)

    # Flatten into the flat-key format that _set_leaf/np.savez uses
    flat = {}
    def _walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}/{k}" if path else k)
        else:
            flat[path] = np.asarray(node)
    _walk(float_params)

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez(tmp.name, **flat)

    # Peek: how many Dense/kernel keys are in the float checkpoint
    n_dense = sum(1 for k in flat if k.endswith("/kernel") and "Dense_" in k)
    print(f"  Float checkpoint: {n_dense} Dense/kernel entries")
    assert n_dense > 0, "Expected some Dense kernels in float checkpoint"

    # Now simulate the _load_leaf logic from train.py
    # (copy-pasted equivalent — if train.py changes, update here)
    def _candidate_keys(path_str):
        yield path_str, False
        if "/NativeBitDense_" in path_str and path_str.endswith("/weight"):
            translated = (path_str.replace("/NativeBitDense_", "/Dense_")
                          .replace("/weight", "/kernel"))
            yield translated, True

    # Build NB model, see if we can load the float checkpoint in
    nb_model = build_model(TinyConfig(), use_nativebit=True)
    nb_params = nb_model.init(rng, dummy)

    ckpt_data = np.load(tmp.name)
    ckpt_keys = set(ckpt_data.files)
    loaded = {"transposed": 0, "exact": 0, "missed_nb_weight": 0}
    total_nb_weights = 0

    def _check(path, leaf):
        nonlocal total_nb_weights
        key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        is_nb_weight = "/NativeBitDense_" in key and key.endswith("/weight")
        if is_nb_weight:
            total_nb_weights += 1
        for cand, needs_transpose in _candidate_keys(key):
            if cand in ckpt_keys:
                arr = np.asarray(ckpt_data[cand])
                if needs_transpose:
                    arr = arr.T
                if arr.shape == leaf.shape:
                    if needs_transpose:
                        loaded["transposed"] += 1
                    else:
                        loaded["exact"] += 1
                return
        if is_nb_weight:
            loaded["missed_nb_weight"] += 1

    jax.tree_util.tree_map_with_path(_check, nb_params)
    ckpt_data.close()

    print(f"  NB layers with /weight: {total_nb_weights}")
    print(f"  Loaded exact (embeddings, norms): {loaded['exact']}")
    print(f"  Loaded with transpose (NB<-Dense): {loaded['transposed']}")
    print(f"  Missed NB weights: {loaded['missed_nb_weight']}")

    assert loaded["transposed"] == total_nb_weights, (
        f"Expected all {total_nb_weights} NB weights to load with transpose, "
        f"got {loaded['transposed']}")
    assert loaded["exact"] > 0, "Expected embeddings / norms to load exact"
    assert loaded["missed_nb_weight"] == 0, "All NB weights should have loaded"
    print(f"  [PASS] All {total_nb_weights} NB layer weights correctly translated "
          f"from float Dense kernels")
    try:
        os.unlink(tmp.name)
    except Exception:
        pass  # Windows leaves the mmap open longer; harmless


if __name__ == "__main__":
    print("Testing QAT init-from float checkpoint:")
    test_float_to_nb_key_translation()
    print("\nAll tests passed.")
