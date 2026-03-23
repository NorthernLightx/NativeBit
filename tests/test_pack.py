"""Tests for bit-packing, export, and loading."""

import math
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.pack import (
    pack_3bit, unpack_3bit,
    pack_4bit, unpack_4bit,
    pack_nbits, unpack_nbits,
    reconstruct_weight,
)


# ---------------------------------------------------------------------------
# 3-bit packing
# ---------------------------------------------------------------------------

class TestPack3Bit:
    def test_roundtrip_simple(self):
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
        packed = pack_3bit(indices)
        recovered = unpack_3bit(packed, 8)
        assert torch.equal(indices, recovered)

    def test_roundtrip_random(self):
        indices = torch.randint(0, 8, (1000,), dtype=torch.uint8)
        packed = pack_3bit(indices)
        recovered = unpack_3bit(packed, 1000)
        assert torch.equal(indices, recovered)

    def test_roundtrip_non_multiple_of_8(self):
        for n in [1, 3, 7, 10, 13, 100, 1001]:
            indices = torch.randint(0, 8, (n,), dtype=torch.uint8)
            packed = pack_3bit(indices)
            recovered = unpack_3bit(packed, n)
            assert torch.equal(indices, recovered), f"Failed for n={n}"

    def test_compression_ratio(self):
        # 8 indices * 3 bits = 24 bits = 3 bytes
        indices = torch.randint(0, 8, (8,), dtype=torch.uint8)
        packed = pack_3bit(indices)
        assert packed.numel() == 3

    def test_large_tensor(self):
        indices = torch.randint(0, 8, (100_000,), dtype=torch.uint8)
        packed = pack_3bit(indices)
        recovered = unpack_3bit(packed, 100_000)
        assert torch.equal(indices, recovered)

    def test_all_zeros(self):
        indices = torch.zeros(16, dtype=torch.uint8)
        packed = pack_3bit(indices)
        recovered = unpack_3bit(packed, 16)
        assert torch.equal(indices, recovered)

    def test_all_sevens(self):
        indices = torch.full((16,), 7, dtype=torch.uint8)
        packed = pack_3bit(indices)
        recovered = unpack_3bit(packed, 16)
        assert torch.equal(indices, recovered)


# ---------------------------------------------------------------------------
# 4-bit packing
# ---------------------------------------------------------------------------

class TestPack4Bit:
    def test_roundtrip(self):
        indices = torch.randint(0, 16, (100,), dtype=torch.uint8)
        packed = pack_4bit(indices)
        recovered = unpack_4bit(packed, 100)
        assert torch.equal(indices, recovered)

    def test_compression_ratio(self):
        indices = torch.randint(0, 16, (100,), dtype=torch.uint8)
        packed = pack_4bit(indices)
        assert packed.numel() == 50

    def test_odd_count(self):
        indices = torch.randint(0, 16, (101,), dtype=torch.uint8)
        packed = pack_4bit(indices)
        recovered = unpack_4bit(packed, 101)
        assert torch.equal(indices, recovered)


# ---------------------------------------------------------------------------
# Generic N-bit packing
# ---------------------------------------------------------------------------

class TestPackNBits:
    def test_2bit_roundtrip(self):
        indices = torch.randint(0, 4, (100,), dtype=torch.uint8)
        packed = pack_nbits(indices, 2)
        recovered = unpack_nbits(packed, 100, 2)
        assert torch.equal(indices, recovered)

    def test_5bit_roundtrip(self):
        indices = torch.randint(0, 32, (50,), dtype=torch.uint8)
        packed = pack_nbits(indices, 5)
        recovered = unpack_nbits(packed, 50, 5)
        assert torch.equal(indices, recovered)

    def test_3bit_delegates(self):
        indices = torch.randint(0, 8, (24,), dtype=torch.uint8)
        packed_generic = pack_nbits(indices, 3)
        packed_direct = pack_3bit(indices)
        assert torch.equal(packed_generic, packed_direct)


# ---------------------------------------------------------------------------
# Weight reconstruction
# ---------------------------------------------------------------------------

class TestReconstructWeight:
    def test_basic(self):
        num_blocks = 4
        block_size = 8
        n_entries = 8
        shape = (4, 8)  # 32 weights = 4 blocks * 8

        codebook = torch.randn(num_blocks, n_entries)
        indices = torch.randint(0, n_entries, (num_blocks * block_size,))

        weight = reconstruct_weight(indices, codebook, shape, block_size, 32)
        assert weight.shape == shape

        # Verify first block manually
        for i in range(block_size):
            expected = codebook[0, indices[i]]
            assert weight.view(-1)[i] == expected

    def test_with_padding(self):
        # 10 weights, block_size=8 -> 2 blocks, padded_len=16
        num_blocks = 2
        block_size = 8
        total_weights = 10
        shape = (2, 5)

        codebook = torch.randn(num_blocks, 8)
        indices = torch.randint(0, 8, (num_blocks * block_size,))

        weight = reconstruct_weight(indices, codebook, shape, block_size, total_weights)
        assert weight.shape == shape
        assert weight.numel() == total_weights


# ---------------------------------------------------------------------------
# Export + Load integration (requires trained checkpoint)
# ---------------------------------------------------------------------------

class TestExportLoad:
    """Integration tests using a tiny model trained on the fly."""

    @pytest.fixture
    def tiny_model_checkpoint(self, tmp_path):
        """Create a tiny NativeBit model and save a checkpoint."""
        from nativebit.model import NativeBitGPT

        model = NativeBitGPT(
            vocab_size=256, n_layers=2, n_embd=32,
            n_head=2, ffn_hidden=64, context_len=32,
            block_size=8, n_entries=8, use_nativebit=True,
        )

        # Run a forward pass to populate utilization
        x = torch.randint(0, 256, (1, 16))
        _ = model(x)
        model.update_all_utilization()

        ckpt_path = str(tmp_path / "test_ckpt.pt")
        config_dict = {
            "vocab_size": 256, "n_layers": 2, "n_embd": 32,
            "n_head": 2, "ffn_hidden": 64, "context_len": 32,
            "block_size": 8, "n_codebook": 8,
        }
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config_dict,
        }, ckpt_path)

        return ckpt_path, model, tmp_path

    def test_export_creates_file(self, tiny_model_checkpoint):
        from nativebit.pack import export_packed

        ckpt_path, _, tmp_path = tiny_model_checkpoint
        output_path = str(tmp_path / "test.nbpack")

        stats = export_packed(ckpt_path, output_path)
        assert os.path.exists(output_path)
        assert stats["file_size_bytes"] > 0
        assert stats["n_quantized_layers"] > 0
        assert stats["bits"] == 3

    def test_packed_is_smaller(self, tiny_model_checkpoint):
        from nativebit.pack import export_packed

        ckpt_path, _, tmp_path = tiny_model_checkpoint
        output_path = str(tmp_path / "test.nbpack")

        stats = export_packed(ckpt_path, output_path)
        assert stats["file_size_bytes"] < stats["original_size_bytes"]

    def test_load_packed_returns_model(self, tiny_model_checkpoint):
        from nativebit.pack import export_packed, load_packed

        ckpt_path, _, tmp_path = tiny_model_checkpoint
        output_path = str(tmp_path / "test.nbpack")

        export_packed(ckpt_path, output_path)
        model = load_packed(output_path, device="cpu")

        # Should be able to do forward pass
        x = torch.randint(0, 256, (1, 16))
        logits = model(x)
        assert logits.shape == (1, 16, 256)

    def test_packed_output_matches_original(self, tiny_model_checkpoint):
        from nativebit.pack import export_packed, verify_packed

        ckpt_path, _, tmp_path = tiny_model_checkpoint
        output_path = str(tmp_path / "test.nbpack")

        export_packed(ckpt_path, output_path)
        result = verify_packed(ckpt_path, output_path, device="cpu")

        assert result["passed"], f"Max diff: {result['max_diff']}"
        assert result["max_diff"] < 0.1
