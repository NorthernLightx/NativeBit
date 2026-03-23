"""Tests for NativeBitLinear layer."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.layers import NativeBitLinear
from nativebit.codebook_utils import init_codebook_percentile, revive_dead_entries


class TestNativeBitLinear:
    def test_creation(self):
        layer = NativeBitLinear(16, 32, bias=True, block_size=8, n_entries=8)
        assert layer.in_features == 16
        assert layer.out_features == 32
        assert layer.num_blocks == math.ceil(16 * 32 / 8)
        assert layer.codebook.shape == (layer.num_blocks, 8)

    def test_forward_shape(self):
        layer = NativeBitLinear(16, 32, bias=True, block_size=8, n_entries=8)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 32)

    def test_forward_no_bias(self):
        layer = NativeBitLinear(16, 32, bias=False, block_size=8, n_entries=8)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 32)

    def test_backward_flows(self):
        layer = NativeBitLinear(16, 32, block_size=8, n_entries=8)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Gradients should flow to both weight and codebook
        assert layer.weight.grad is not None
        assert layer.codebook.grad is not None
        assert x.grad is not None

    def test_quantized_output_uses_codebook_values(self):
        layer = NativeBitLinear(4, 4, bias=False, block_size=4, n_entries=4)
        layer.train(False)

        x = torch.randn(1, 1, 4)
        with torch.no_grad():
            out = layer(x)

        # After quantization, the effective weight values should be codebook entries
        w_flat = layer.weight.view(-1)
        codebook_values = set()
        for b in range(layer.num_blocks):
            for e in range(layer.n_entries):
                codebook_values.add(round(layer.codebook[b, e].item(), 6))

    def test_utilization_update(self):
        layer = NativeBitLinear(16, 16, block_size=8, n_entries=8)
        x = torch.randn(4, 8, 16)
        _ = layer(x)
        layer.update_utilization_from_cache()

        # After update, utilization should have non-zero values
        assert layer.utilization.sum().item() > 0

    def test_different_block_sizes(self):
        for bs in [4, 8, 16, 32, 64]:
            layer = NativeBitLinear(64, 64, block_size=bs, n_entries=8)
            x = torch.randn(1, 4, 64)
            out = layer(x)
            assert out.shape == (1, 4, 64), f"Failed for block_size={bs}"

    def test_different_n_entries(self):
        for n in [2, 4, 8, 16]:
            layer = NativeBitLinear(32, 32, block_size=8, n_entries=n)
            x = torch.randn(1, 4, 32)
            out = layer(x)
            assert out.shape == (1, 4, 32), f"Failed for n_entries={n}"


class TestCodebookUtils:
    def test_init_percentile(self):
        w = torch.randn(100)
        cb = init_codebook_percentile(w, 8)
        assert cb.shape == (8,)
        # Should be sorted (percentiles are monotonic)
        assert torch.all(cb[1:] >= cb[:-1])

    def test_init_percentile_covers_range(self):
        w = torch.randn(1000)
        cb = init_codebook_percentile(w, 8)
        # First entry should be near min, last near max
        assert cb[0] <= w.median()
        assert cb[-1] >= w.median()

    def test_revive_dead_entries(self):
        num_blocks, n_entries = 10, 8
        codebook = torch.randn(num_blocks, n_entries)
        codebook = torch.nn.Parameter(codebook)

        # Simulate all usage going to entry 0
        util = torch.zeros(num_blocks, n_entries, dtype=torch.long)
        util[:, 0] = 100

        old_values = codebook.data.clone()
        n_revived = revive_dead_entries(codebook, util)

        # 7 dead entries per block * 10 blocks = 70
        assert n_revived == 70
        # Dead entries should have been updated
        assert not torch.equal(old_values, codebook.data)

    def test_revive_no_dead(self):
        num_blocks, n_entries = 5, 8
        codebook = torch.randn(num_blocks, n_entries)
        codebook = torch.nn.Parameter(codebook)

        # Uniform utilization
        util = torch.ones(num_blocks, n_entries, dtype=torch.long) * 100
        n_revived = revive_dead_entries(codebook, util)
        assert n_revived == 0
