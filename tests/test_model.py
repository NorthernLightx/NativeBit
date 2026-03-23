"""Tests for NativeBitGPT model."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.model import NativeBitGPT, build_model_from_config
from nativebit.layers import NativeBitLinear


class TestNativeBitGPT:
    def test_float_model_creation(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            use_nativebit=False,
        )
        # Should have no NativeBitLinear layers
        nb = model.get_nativebit_layers()
        assert len(nb) == 0

    def test_nativebit_model_creation(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )
        nb = model.get_nativebit_layers()
        assert len(nb) > 0

    def test_forward_float(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            use_nativebit=False,
        )
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_nativebit(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_backward(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert model.tok_emb.weight.grad is not None
        for layer in model.get_nativebit_layers():
            assert layer.weight.grad is not None

    def test_weight_tying(self):
        model = NativeBitGPT(vocab_size=256, n_layers=2, n_embd=32,
                             n_head=2, ffn_hidden=64, context_len=32)
        assert model.lm_head.weight is model.tok_emb.weight

    def test_context_len_enforced(self):
        model = NativeBitGPT(vocab_size=256, n_layers=2, n_embd=32,
                             n_head=2, ffn_hidden=64, context_len=32)
        x = torch.randint(0, 256, (1, 64))
        with pytest.raises(AssertionError):
            model(x)

    def test_count_parameters(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["embedding"] > 0
        assert counts["codebook"] > 0

    def test_set_mode_inference(self):
        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )
        model.set_mode_inference()
        assert not model.training
        for p in model.parameters():
            assert not p.requires_grad

    def test_build_model_from_config(self):
        class Config:
            vocab_size = 256
            n_layers = 2
            n_embd = 32
            n_head = 2
            ffn_hidden = 64
            context_len = 32
            block_size = 8
            n_codebook = 8

        model = build_model_from_config(Config(), use_nativebit=True)
        assert isinstance(model, NativeBitGPT)

        x = torch.randint(0, 256, (1, 16))
        logits = model(x)
        assert logits.shape == (1, 16, 256)
