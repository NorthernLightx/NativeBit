"""Post-hoc quantization baselines on a trained float checkpoint.

Loads a float checkpoint produced by train.py and evaluates post-hoc
3-bit methods on the SAME weights the QAT run fine-tunes from — the
apples-to-apples comparison for the local scaling curve:

    float ckpt --> eval                (upper bound)
    float ckpt --> RTN 3-bit --> eval  (post-hoc baseline)
    float ckpt --> k-means 3-bit -> eval (stronger post-hoc baseline)

Excludes lm_head (tied to tok_emb, never quantized) and ve_gate modules
(kept float in NativeBit models) so the quantized weight set matches
exactly what NativeBitLinear covers.

Usage:
    python benchmarks/benchmark_posthoc_ckpt.py --ckpt logs/local48m_float_final.pt \
        --config tpu-small --dataset wikitext-103 --block-size 64
"""

import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from nativebit.device import get_device, amp_context
from nativebit.baselines import quantize_uniform, quantize_kmeans


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss * y.numel()
        total_tokens += y.numel()
    return total_loss.item() / max(total_tokens, 1)


def excluded_modules(model) -> list:
    """Modules NativeBit never quantizes: lm_head + every ve_gate."""
    exclude = [model.lm_head]
    for name, module in model.named_modules():
        if name.endswith("ve_gate"):
            exclude.append(module)
    return exclude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="float checkpoint (.pt)")
    parser.add_argument("--config", default="tpu-small",
                        choices=["default", "local-76m", "tpu-small", "tpu-medium",
                                 "tpu-large"])
    parser.add_argument("--dataset", default="wikitext-103")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64,
                        help="Post-hoc block size; match the NB config's")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    if args.config == "default":
        from configs.default import DefaultConfig as Cfg
    elif args.config == "local-76m":
        from configs.local import Local76MConfig as Cfg
    else:
        from configs import tpu
        Cfg = {"tpu-small": tpu.TPUSmallConfig,
               "tpu-medium": tpu.TPUMediumConfig,
               "tpu-large": tpu.TPULargeConfig}[args.config]
    config = Cfg()
    config.dataset = args.dataset
    config.batch_size = args.batch_size

    model = build_model_from_config(config, use_nativebit=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys loading float ckpt: {missing[:5]}")
    model = model.to(device)

    _, valid_loader, test_loader = get_dataloaders(
        config.context_len, config.batch_size, args.data_dir,
        dataset=args.dataset)

    n_entries = 2 ** args.bits
    results = {}

    def report(name, m):
        val = run_eval(m, valid_loader, device)
        test = run_eval(m, test_loader, device)
        results[name] = (val, test)
        print(f"  {name:<22} val_ppl={math.exp(min(val, 20)):>8.2f}  "
              f"test_ppl={math.exp(min(test, 20)):>8.2f}", flush=True)

    print(f"Post-hoc baselines on {args.ckpt} "
          f"({args.bits}-bit, block {args.block_size}):")
    report("float (unquantized)", model)

    m_rtn = copy.deepcopy(model)
    quantize_uniform(m_rtn, bits=args.bits, block_size=args.block_size,
                     exclude_modules=excluded_modules(m_rtn))
    report(f"RTN {args.bits}-bit", m_rtn)
    del m_rtn
    torch.cuda.empty_cache()

    m_km = copy.deepcopy(model)
    quantize_kmeans(m_km, n_entries=n_entries, block_size=args.block_size,
                    exclude_modules=excluded_modules(m_km))
    report(f"k-means {args.bits}-bit", m_km)

    float_test = results["float (unquantized)"][1]
    print("\nvs float (test):")
    for name, (_, test) in results.items():
        delta = (math.exp(min(test, 20)) / math.exp(min(float_test, 20)) - 1) * 100
        print(f"  {name:<22} {delta:+.2f}%")


if __name__ == "__main__":
    main()
