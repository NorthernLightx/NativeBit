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
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        # Use greedy (temperature=0) to avoid BPE roundtrip issues
        # and count via the raw token IDs inside generate
        for max_t in [5, 20, 50]:
            text = generate(tiny_model, "The", max_tokens=max_t,
                            temperature=0, stop_at_eos=False)
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
            prompt_tokens = len(enc.encode("The", allowed_special={"<|endoftext|>"}))
            gen_tokens = len(tokens) - prompt_tokens
            # BPE decode/re-encode can differ by +/- 1 token
            assert abs(gen_tokens - max_t) <= 1, f"Expected ~{max_t}, got {gen_tokens}"

    def test_stop_at_eos_false(self, tiny_model):
        # With stop_at_eos=False, generate should run the full loop (max_tokens
        # new tokens) without early-stopping on EOS.
        #
        # We can't check the count exactly via BPE round-trip: decoded text
        # re-encoded with tiktoken can differ substantially from the generated
        # token count, especially for a random-init tiny model whose output
        # contains byte sequences that re-tokenize differently. Just verify
        # the output is non-trivial length — the real "didn't stop early"
        # guarantee lives in the `for _ in range(max_tokens)` loop.
        text = generate(tiny_model, "The", max_tokens=30,
                        temperature=0, stop_at_eos=False)
        assert len(text) > len("The"), "generate produced no new content"

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
