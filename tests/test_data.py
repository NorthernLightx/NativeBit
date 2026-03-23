import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
from nativebit.data import build_token_byte_table


class TestBPB:
    def test_token_byte_table_shape(self):
        enc = tiktoken.get_encoding("gpt2")
        table = build_token_byte_table()
        assert table.shape == (enc.n_vocab,)
        assert table.dtype == torch.int32

    def test_token_byte_table_values(self):
        enc = tiktoken.get_encoding("gpt2")
        table = build_token_byte_table()
        hello_ids = enc.encode("Hello")
        for tid in hello_ids:
            decoded = enc.decode([tid])
            expected_bytes = len(decoded.encode("utf-8"))
            assert table[tid].item() == expected_bytes

    def test_token_byte_table_no_zeros(self):
        table = build_token_byte_table()
        assert (table > 0).sum().item() > 50000

    def test_compute_bpb_returns_float(self):
        from nativebit.model import build_model_from_config
        from nativebit.data import compute_bpb, get_dataloaders
        from configs.default import DefaultConfig

        config = DefaultConfig()
        model = build_model_from_config(config, use_nativebit=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        _, valid_loader, _ = get_dataloaders(config.context_len, config.batch_size, "data")
        bpb = compute_bpb(model, valid_loader, device)

        assert isinstance(bpb, float)
        assert bpb > 0
        assert bpb > 2.0  # untrained model = high BPB (random ~3-4 BPB on WikiText-2)
