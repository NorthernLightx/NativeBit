"""Four-way 2.2B decode benchmark: float fp16 vs NB 3-bit, eager vs CUDA graph.

Produces the systems table for the paper: weights size, VRAM, prefill
latency (time-to-first-token), decode throughput, and generation samples
from both models. Writes markdown to logs/decode_bench_2b.md.

Usage:
    python inference/bench_decode_2b.py \
        --nb inference/2b_nb_fixed.nbpack.npz \
        --float-npz logs/gcs/2b_float_fixed_params.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.generate_torch import PackedGPT, load_packed_model
from inference.graph_decode import GraphDecoder


def load_float_model(npz_path, config, device="cuda"):
    """Build PackedGPT with fp16 nn.Linear from a JAX float .npz checkpoint."""
    print(f"  Loading float {npz_path}...", flush=True)
    t0 = time.time()
    z = np.load(npz_path)

    model = PackedGPT(
        config.vocab_size, config.n_layers, config.n_embd, config.n_head,
        config.ffn_hidden, config.context_len,
        lambda out_f, in_f: nn.Linear(in_f, out_f, bias=False),
    )

    def put(module, key, transpose=True):
        arr = z[key]
        t = torch.from_numpy(np.ascontiguousarray(arr.T if transpose else arr))
        module.weight.data = t.to(torch.float16).to(device)

    put(model.embedding, "params/embedding", transpose=False)
    for i in range(config.n_layers):
        b = model.blocks[i]
        pre = f"params/block_{i}"
        put(b.attn.qkv, f"{pre}/CausalSelfAttention_0/Dense_0/kernel")
        put(b.attn.out_proj, f"{pre}/CausalSelfAttention_0/Dense_1/kernel")
        put(b.ffn.gate, f"{pre}/SwiGLU_0/Dense_0/kernel")
        put(b.ffn.up, f"{pre}/SwiGLU_0/Dense_1/kernel")
        put(b.ffn.down, f"{pre}/SwiGLU_0/Dense_2/kernel")
        b.ln1.weight.data = torch.from_numpy(
            z[f"{pre}/RMSNorm_0/weight"]).to(torch.float16).to(device)
        b.ln2.weight.data = torch.from_numpy(
            z[f"{pre}/RMSNorm_1/weight"]).to(torch.float16).to(device)
    model.ln_f.weight.data = torch.from_numpy(
        z["params/ln_f/weight"]).to(torch.float16).to(device)

    model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    wbytes = sum(p.nbytes for p in model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s, weights {wbytes/1e9:.2f} GB", flush=True)
    return model


@torch.no_grad()
def sample_eager(model, dec, prompt_tokens, n=100, temp=0.8, top_k=40, seed=42):
    """Sampled generation via eager steps (temperature + top-k)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dec.prefill(prompt_tokens)
    tokens = []
    caches = [(dec.k_bufs[i], dec.v_bufs[i], len(prompt_tokens))
              for i in range(model.n_layers)]
    cur = torch.tensor([[dec.first_token]], dtype=torch.long, device="cuda")
    tokens.append(dec.first_token)
    for _ in range(n - 1):
        logits, caches = model(cur, kv_caches=caches)
        logits = logits[0, -1].float() / temp
        v, ix = torch.topk(logits, top_k)
        probs = F.softmax(v, dim=-1)
        nxt = ix[torch.multinomial(probs, 1, generator=g)]
        tokens.append(int(nxt))
        cur = nxt.view(1, 1)
    return tokens


def bench_model(name, model, prompt_tokens, n_generate=256):
    """Measure prefill latency, eager + graph decode throughput, VRAM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res = {"name": name}
    dec = GraphDecoder(model, n_max=n_generate + 8)
    dec.capture()  # capture on fresh state; prefill overwrites it

    # Prefill latency (time-to-first-token): eager batched vs per-token graph
    def _median(fn):
        lats = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            lats.append(time.time() - t0)
        return sorted(lats)[2] * 1e3

    dec.prefill_graph(prompt_tokens)      # clock ramp before any timing
    dec.decode(64, use_graph=True)
    res["ttft_eager_ms"] = _median(lambda: dec.prefill(prompt_tokens))
    res["ttft_graph_ms"] = _median(lambda: dec.prefill_graph(prompt_tokens))
    res["ttft_ms"] = min(res["ttft_eager_ms"], res["ttft_graph_ms"])

    # Eager decode (warm up first — same clock-ramp caveat as below)
    dec.prefill_graph(prompt_tokens)
    dec.decode(32, use_graph=False)
    dec.prefill_graph(prompt_tokens)
    _, t = dec.decode(128, use_graph=False)
    res["eager_tps"] = 128 / t

    # Graph decode. The first timed decode after an idle GPU pays clock
    # ramp-up (210 MHz idle -> ~1920 MHz sustained) and allocator growth —
    # worth 2-3x on this card. Warm up, then take the median of 3.
    dec.prefill_graph(prompt_tokens)
    dec.decode(n_generate, use_graph=True)
    runs = []
    for _ in range(3):
        dec.prefill_graph(prompt_tokens)
        graph_tokens, t = dec.decode(n_generate, use_graph=True)
        runs.append(t)
    t = sorted(runs)[1]
    res["graph_tps"] = n_generate / t
    res["graph_ms_tok"] = t / n_generate * 1e3

    # Greedy sample text (from the graph run)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    res["greedy_text"] = enc.decode(
        prompt_tokens + [dec.first_token] + graph_tokens[:60])

    # Sampled text
    sampled = sample_eager(model, dec, prompt_tokens, n=100)
    res["sampled_text"] = enc.decode(prompt_tokens + sampled)

    res["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["weights_gb"] = (sum(p.nbytes for p in model.parameters())
                         + sum(b.nbytes for b in model.buffers())) / 1e9
    del dec
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb", default="inference/2b_nb_fixed.nbpack.npz")
    parser.add_argument("--float-npz", default="logs/gcs/2b_float_fixed_params.npz")
    parser.add_argument("--prompt", default="The city of Paris announced today")
    parser.add_argument("--out", default="logs/decode_bench_2b.md")
    args = parser.parse_args()

    from configs.tpu import TPU2BConfig
    config = TPU2BConfig()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    model_nb = load_packed_model(args.nb, config)
    results.append(bench_model("NB 3-bit (packed)", model_nb, prompt_tokens))
    del model_nb
    torch.cuda.empty_cache()

    model_f = load_float_model(args.float_npz, config)
    results.append(bench_model("Float fp16", model_f, prompt_tokens))
    del model_f
    torch.cuda.empty_cache()

    # Report
    lines = [
        "# 2.2B Decode Benchmark — Float fp16 vs NativeBit 3-bit",
        "",
        f"GPU: {torch.cuda.get_device_name(0)}. CUDA-graph decode, greedy, "
        f"prompt: \"{args.prompt}\".",
        "",
        "| Model | Weights | Peak VRAM | TTFT | Eager tok/s | Graph tok/s | ms/tok (graph) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['weights_gb']:.2f} GB | {r['peak_vram_gb']:.2f} GB "
            f"| {r['ttft_ms']:.0f} ms | {r['eager_tps']:.1f} "
            f"| {r['graph_tps']:.1f} | {r['graph_ms_tok']:.1f} |")
    lines.append("")
    lines.append("TTFT = best of eager batched prefill vs per-token graph prefill: "
                 + "; ".join(f"{r['name']}: eager {r['ttft_eager_ms']:.0f} ms, "
                             f"graph {r['ttft_graph_ms']:.0f} ms" for r in results))
    lines.append("")
    for r in results:
        lines.append(f"## {r['name']} — greedy (60 tok)\n")
        lines.append(f"> {r['greedy_text']}\n")
        lines.append(f"## {r['name']} — sampled temp 0.8 top-k 40 seed 42 (100 tok)\n")
        lines.append(f"> {r['sampled_text']}\n")

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    for line in lines[:12]:
        print(line)


if __name__ == "__main__":
    main()
