"""CUDA-graph decode for packed NativeBit models.

Eager decode launches ~130 kernels per token (26 layers x 5 matvecs plus
norms/rope/attention); at ~40-90us per matvec the launch+Python overhead is
a real fraction of every token. This module rewrites the single-token step
to be fully device-side — position as a device tensor, in-place KV writes,
greedy argmax fed back on-device — captures it once as a CUDA graph, and
replays the whole per-token sequence with a single launch.

Usage:
    python inference/graph_decode.py inference/2b_nb_fixed.nbpack.npz \
        --n-generate 256 --compare
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.generate_torch import (
    PackedGPT, load_packed_model, init_kv_cache,
)


class GraphDecoder:
    """Greedy single-token decode step, CUDA-graph capturable.

    All state lives in static device tensors:
        cur_token (1,1) long — the token being fed in, overwritten each step
        pos       ()    long — write position in the KV cache
        k_buf/v_buf per layer — preallocated (B, H, ctx, hd)
        tokens_out (n_max,) long — ring of generated tokens
        step_idx  ()    long — index into tokens_out
    """

    def __init__(self, model: PackedGPT, n_max: int, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        self.n_max = n_max
        ctx = model.context_len
        head_dim = model.n_embd // model.n_head

        self.cur_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros((), dtype=torch.long, device=device)
        self.step_idx = torch.zeros((), dtype=torch.long, device=device)
        self.tokens_out = torch.zeros(n_max, dtype=torch.long, device=device)
        self.k_pos = torch.arange(ctx, device=device)

        self.k_bufs = [torch.zeros(1, model.n_head, ctx, head_dim,
                                   device=device, dtype=torch.float32)
                       for _ in range(model.n_layers)]
        self.v_bufs = [torch.zeros_like(self.k_bufs[0])
                       for _ in range(model.n_layers)]

        # Per-position RoPE tables from the first block (shared shape)
        attn0 = model.blocks[0].attn
        self.cos = attn0.cos.to(device)   # (ctx, hd/2)
        self.sin = attn0.sin.to(device)

        self.graph = None

    def _rope_one(self, q, k):
        """RoPE at position self.pos for single-token q,k: (1, H, 1, hd)."""
        cos = self.cos.index_select(0, self.pos.view(1))  # (1, hd/2)
        sin = self.sin.index_select(0, self.pos.view(1))
        cos2 = cos.repeat_interleave(2, dim=-1)[None, None]  # (1,1,1,hd)
        sin2 = sin.repeat_interleave(2, dim=-1)[None, None]

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape(x.shape)

        q = q.float() * cos2 + rotate(q.float()) * sin2
        k = k.float() * cos2 + rotate(k.float()) * sin2
        return q, k

    def step(self):
        """One fully device-side greedy decode step."""
        m = self.model
        head_dim = m.n_embd // m.n_head
        x = m.embedding(self.cur_token)                      # (1,1,C)

        pos1 = self.pos.view(1)
        for i, block in enumerate(m.blocks):
            h = block.ln1(x)
            qkv = block.attn.qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(1, 1, m.n_head, head_dim).transpose(1, 2)
            k = k.reshape(1, 1, m.n_head, head_dim).transpose(1, 2)
            v = v.reshape(1, 1, m.n_head, head_dim).transpose(1, 2)
            q, k = self._rope_one(q, k)

            self.k_bufs[i].index_copy_(2, pos1, k)
            self.v_bufs[i].index_copy_(2, pos1, v.float())

            scale = head_dim ** -0.5
            attn = torch.einsum("bhqd,bhkd->bhqk",
                                q * scale, self.k_bufs[i])   # (1,H,1,ctx)
            mask = self.k_pos <= self.pos                    # (ctx,)
            attn = attn.masked_fill(~mask[None, None, None],
                                    torch.finfo(attn.dtype).min)
            out = torch.einsum("bhqk,bhkd->bhqd",
                               F.softmax(attn, dim=-1), self.v_bufs[i])
            out = out.transpose(1, 2).reshape(1, 1, m.n_embd).to(x.dtype)
            x = x + block.attn.out_proj(out)
            x = x + block.ffn(block.ln2(x))

        x = m.ln_f(x)
        logits = x.to(m.embedding.weight.dtype) @ m.embedding.weight.T
        logits = 30.0 * torch.tanh(logits.float() / 30.0)

        token = logits[0, -1].argmax().view(1)
        self.cur_token.copy_(token.view(1, 1))
        self.tokens_out.index_copy_(0, self.step_idx.view(1), token)
        self.step_idx.add_(1)
        self.pos.add_(1)

    def prefill(self, prompt_tokens):
        """Eager prefill into this decoder's KV buffers."""
        caches = [(self.k_bufs[i], self.v_bufs[i], 0)
                  for i in range(self.model.n_layers)]
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x, kv_caches=caches)
        self.pos.fill_(len(prompt_tokens))
        self.step_idx.fill_(0)
        first = logits[0, -1].argmax().view(1, 1)
        self.cur_token.copy_(first)
        # The prefill prediction is the first generated token; the decode
        # loop consumes it as input, so record it for text output.
        self.first_token = int(first)

    def prefill_graph(self, prompt_tokens):
        """Prefill by replaying the captured graph once per prompt token.

        The batched (T>1) PackedLinear path unpacks 3-bit indices in Python
        per layer per call (~778ms TTFT at 2.2B); one graph replay per token
        is ~8ms. Requires capture() to have run. The tokens ring accumulates
        prefill garbage, so step_idx is reset afterwards; the last replay's
        argmax is exactly the first generated token.
        """
        assert self.graph is not None, "call capture() first"
        self.pos.fill_(0)
        self.step_idx.fill_(0)
        for t in prompt_tokens:
            self.cur_token.fill_(t)
            self.graph.replay()
        torch.cuda.synchronize()
        self.first_token = int(self.cur_token)
        self.step_idx.fill_(0)

    def capture(self):
        """Warm up on a side stream, then capture one step as a CUDA graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    self.step()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.step()

    def decode(self, n_tokens, use_graph=True):
        """Generate n_tokens greedily. Returns (tokens list, seconds)."""
        torch.cuda.synchronize()
        t0 = time.time()
        if use_graph:
            for _ in range(n_tokens):
                self.graph.replay()
        else:
            with torch.no_grad():
                for _ in range(n_tokens):
                    self.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        n = int(self.step_idx)
        return self.tokens_out[:min(n, self.n_max)].tolist(), dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--n-generate", type=int, default=256)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--compare", action="store_true",
                        help="Bench eager step vs graph replay + check tokens match")
    args = parser.parse_args()

    from configs.tpu import TPU2BConfig
    config = TPU2BConfig()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = load_packed_model(args.checkpoint, config)
    dec = GraphDecoder(model, n_max=args.n_generate + 8)

    if args.compare:
        # Eager reference
        dec.prefill(prompt_tokens)
        eager_tokens, eager_t = dec.decode(args.n_generate, use_graph=False)
        eager_tps = args.n_generate / eager_t
        print(f"  eager: {eager_tps:6.1f} tok/s ({eager_t/args.n_generate*1e3:.1f} ms/tok)")

        # Graph
        dec.prefill(prompt_tokens)
        dec.capture()
        dec.prefill(prompt_tokens)  # reset state mutated by warmup+capture
        graph_tokens, graph_t = dec.decode(args.n_generate, use_graph=True)
        graph_tps = args.n_generate / graph_t
        print(f"  graph: {graph_tps:6.1f} tok/s ({graph_t/args.n_generate*1e3:.1f} ms/tok)")
        print(f"  speedup: {graph_tps/eager_tps:.2f}x")

        match = eager_tokens[:args.n_generate] == graph_tokens[:args.n_generate]
        print(f"  token streams match: {match}")
        if not match:
            for i, (a, b) in enumerate(zip(eager_tokens, graph_tokens)):
                if a != b:
                    print(f"  first divergence at step {i}: {a} vs {b}")
                    break
    else:
        dec.prefill(prompt_tokens)
        dec.capture()
        dec.prefill(prompt_tokens)
        tokens, dt = dec.decode(args.n_generate, use_graph=True)
        print(f"  {args.n_generate / dt:.1f} tok/s")
        print(enc.decode(prompt_tokens + [dec.first_token] + tokens))


if __name__ == "__main__":
    main()
