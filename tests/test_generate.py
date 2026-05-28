"""Tests for text generation."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.model import NativeBitGPT
from nativebit.generate import generate


@pytest.fixture
def tiny_model():
    """Create a tiny model for generation tests."""
    model = NativeBitGPT(
        vocab_size=50257, n_layers=2, n_embd=32,
        n_head=2, ffn_hidden=64, context_len=64,
        block_size=8, n_entries=8, use_nativebit=True,
    )
    return model


class TestGenerate:
    def test_returns_string(self, tiny_model):
        text = generate(tiny_model, "Hello", max_tokens=10, temperature=0.8)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_prompt(self, tiny_model):
        text = generate(tiny_model, "Hello", max_tokens=10, temperature=0.8)
        assert text.startswith("Hello")

    def test_max_tokens_respected(self, tiny_model):
        # Count generated tokens exactly via the returned IDs. A BPE
        # decode/re-encode roundtrip is not a bijection (adjacent generated
        # tokens can merge), so counting re-encoded text under-counts and is
        # tiktoken-version-dependent. return_ids gives the exact generated count.
        for max_t in [5, 20, 50]:
            text, new_ids = generate(tiny_model, "The", max_tokens=max_t,
                                     temperature=0, stop_at_eos=False,
                                     return_ids=True)
            assert len(new_ids) == max_t, f"Expected {max_t}, got {len(new_ids)}"

    def test_stop_at_eos_false(self, tiny_model):
        # With stop_at_eos=False, generate runs the full loop and emits exactly
        # max_tokens new tokens. Check the generated IDs directly rather than a
        # BPE round-trip (see test_max_tokens_respected).
        text, new_ids = generate(tiny_model, "The", max_tokens=30,
                                 temperature=0, stop_at_eos=False,
                                 return_ids=True)
        assert len(new_ids) == 30, f"Expected 30 new tokens, got {len(new_ids)}"

    def test_greedy_deterministic(self, tiny_model):
        t1 = generate(tiny_model, "Once", max_tokens=20, temperature=0)
        t2 = generate(tiny_model, "Once", max_tokens=20, temperature=0)
        assert t1 == t2

    def test_float_model_works(self):
        model = NativeBitGPT(
            vocab_size=50257, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=64,
            use_nativebit=False,
        )
        text = generate(model, "The", max_tokens=10, temperature=0.8)
        assert isinstance(text, str)
        assert text.startswith("The")
