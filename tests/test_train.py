"""Integration test: a few training steps run end-to-end without blowing up.

Replaces the earlier subprocess smoke test (`python train.py --max-steps 50`)
which downloaded WikiText-2 on first run and was unsuitable for CI. This
exercises the same mechanics — forward, loss, backward, optimizer — against
synthetic data instead.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.model import NativeBitGPT


def _build_tiny_model(use_nativebit: bool) -> NativeBitGPT:
    return NativeBitGPT(
        vocab_size=100, n_layers=2, n_embd=16, n_head=2,
        ffn_hidden=32, context_len=32, block_size=8, n_entries=8,
        use_nativebit=use_nativebit,
    )


def _run_steps(model: NativeBitGPT, n_steps: int, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(n_steps):
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 100), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class TestTrainingStep:
    def test_nativebit_steps_run_and_update_weights(self):
        torch.manual_seed(0)
        model = _build_tiny_model(use_nativebit=True)
        before = model.tok_emb.weight.detach().clone()
        losses = _run_steps(model, n_steps=3)

        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            f"Non-finite loss in {losses}"
        assert not torch.allclose(before, model.tok_emb.weight), \
            "Token embedding did not update after optimizer step"

    def test_float_steps_run_and_update_weights(self):
        torch.manual_seed(0)
        model = _build_tiny_model(use_nativebit=False)
        before = model.tok_emb.weight.detach().clone()
        losses = _run_steps(model, n_steps=3)

        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        assert not torch.allclose(before, model.tok_emb.weight)
