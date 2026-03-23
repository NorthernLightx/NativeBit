"""Train NativeBit models -- quantization-aware training from birth."""

import argparse
import math
import os
import sys
import time

# torch.compile on Windows: auto-detect MSVC compiler + set cache paths
if sys.platform == "win32":
    if "CC" not in os.environ:
        _msvc_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        ]
        for _base in _msvc_paths:
            if os.path.isdir(_base):
                _versions = sorted(os.listdir(_base), reverse=True)
                if _versions:
                    _cl = os.path.join(_base, _versions[0], "bin", "Hostx64", "x64", "cl.exe")
                    if os.path.isfile(_cl):
                        os.environ["CC"] = _cl
                        break
    for _var, _path in [("TRITON_CACHE_DIR", r"C:\tmp\triton"),
                        ("TORCHINDUCTOR_CACHE_DIR", r"C:\tmp\inductor")]:
        if _var not in os.environ:
            os.environ[_var] = _path
            os.makedirs(_path, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.layers import NativeBitLinear
from nativebit.data import get_dataloaders, compute_bpb
from nativebit.logging import TrainingLogger, compute_gradient_info
from configs.default import DefaultConfig


def get_param_groups(model: nn.Module, lr: float, codebook_lr: float,
                     weight_decay: float = 0.01) -> list[dict]:
    """Separate parameters into weight-decay, no-decay, and codebook groups."""
    decay, no_decay, codebook = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "codebook" in name:
            codebook.append(param)
        elif "tok_emb" in name or "ln" in name or "norm" in name or (name.endswith(".weight") and param.dim() == 1):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]
    if codebook:
        groups.append({"params": codebook, "lr": codebook_lr, "weight_decay": 0.0})
    return groups


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int,
                        min_lr_ratio: float = 0.1):
    """Cosine LR schedule with linear warmup and minimum LR floor."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def run_evaluation(model: nn.Module, loader, device: torch.device) -> float:
    """Compute average cross-entropy loss on a dataloader."""
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


def clip_codebook_grads(model: nn.Module, max_norm: float) -> float:
    """Clip codebook parameter gradients separately."""
    params = [p for n, p in model.named_parameters() if "codebook" in n and p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm).item()


def train(model: nn.Module, config, device: torch.device,
          experiment_name: str = "nativebit", log_dir: str = "logs",
          data_dir: str = "data") -> dict:
    """Run the NativeBit training loop.

    Returns dict with val_loss, val_ppl, test_loss, test_ppl, val_bpb.
    """
    torch.set_float32_matmul_precision("high")
    model = model.to(device)
    model.train()

    # Data
    train_loader, valid_loader, test_loader = get_dataloaders(
        config.context_len, config.batch_size, data_dir,
        dataset=getattr(config, "dataset", "wikitext-2"),
    )

    # Optimizer + scheduler
    param_groups = get_param_groups(model, config.lr, config.codebook_lr,
                                   weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule(optimizer, config.warmup_steps, config.max_steps)

    # Logger
    logger = TrainingLogger(log_dir, experiment_name)

    # Model summary
    counts = model.count_parameters() if hasattr(model, "count_parameters") else {}
    print(f"\n=== {experiment_name} ===")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print(f"  Device: {device}")
    print(f"  Steps: {config.max_steps}")
    print()

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    step = 0
    train_iter = iter(train_loader)

    while True:
        optimizer.zero_grad()

        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward + loss
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward
        scaler.scale(loss).backward()
        loss_val = loss.item()

        # Fast fail
        if math.isnan(loss_val) or loss_val > 100:
            print("FAIL")
            sys.exit(1)

        # Gradient clipping + optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        clip_codebook_grads(model, config.codebook_grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update utilization counters
        if (step % config.log_every == 0 or
                (step > 0 and step % config.revive_every == 0)):
            if hasattr(model, "update_all_utilization"):
                model.update_all_utilization()

        # Revive dead codebook entries
        if (hasattr(model, "revive_all_dead_entries") and
                step > 0 and step % config.revive_every == 0):
            revived = model.revive_all_dead_entries()
            if revived > 0:
                print(f"  Step {step}: revived {revived} dead codebook entries")

        # Logging
        if step % config.log_every == 0:
            grad_info = compute_gradient_info(model) if hasattr(model, "get_nativebit_layers") else None
            current_lr = optimizer.param_groups[0]["lr"]
            record = logger.log_step(step, loss_val, current_lr, model, grad_info)

            ppl_str = f"{record['perplexity']:>10.2f}"
            dead_str = f"dead={record.get('dead_entries', 0)}" if "dead_entries" in record else ""
            grad_str = f"cb/w={record.get('grad_ratio_cb_w', 0):.3f}" if "grad_ratio_cb_w" in record else ""
            print(f"  step={step:>5d}  loss={record['loss']:.4f}  ppl={ppl_str}  lr={record['lr']:.2e}  {dead_str}  {grad_str}")

            if step % (config.log_every * 4) == 0:
                logger.save_codebook_snapshot(step, model)

        if step >= config.max_steps:
            break
        step += 1

    # Final metrics
    print(f"\nFinal metrics for {experiment_name}...")
    val_loss = run_evaluation(model, valid_loader, device)
    test_loss = run_evaluation(model, test_loader, device)
    val_ppl = math.exp(min(val_loss, 20))
    test_ppl = math.exp(min(test_loss, 20))
    val_bpb = compute_bpb(model, valid_loader, device)

    print(f"  Val loss: {val_loss:.4f}  Val PPL: {val_ppl:.2f}")
    print(f"  Test loss: {test_loss:.4f}  Test PPL: {test_ppl:.2f}")
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"num_steps:        {step}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    if device.type == "cuda":
        print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")

    # Save checkpoint
    ckpt_path = os.path.join(log_dir, f"{experiment_name}_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: getattr(config, k) for k in dir(config)
                   if not k.startswith("_") and not callable(getattr(config, k))},
        "val_loss": val_loss, "val_ppl": val_ppl,
        "test_loss": test_loss, "test_ppl": test_ppl, "val_bpb": val_bpb,
    }, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")
    logger.close()

    return {"train_loss": loss_val, "val_loss": val_loss, "val_ppl": val_ppl,
            "test_loss": test_loss, "test_ppl": test_ppl, "val_bpb": val_bpb}


def main():
    parser = argparse.ArgumentParser(description="Train NativeBit GPT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="nativebit")
    parser.add_argument("--no-nativebit", action="store_true",
                        help="Train float baseline (no quantization)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    config = DefaultConfig()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    config.seed = args.seed
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_nativebit = not args.no_nativebit

    model = build_model_from_config(config, use_nativebit=use_nativebit)

    # Re-init codebooks via k-means after model._init_weights rescales some layers
    if use_nativebit and hasattr(model, 'get_nativebit_layers'):
        from nativebit.codebook_utils import init_codebook_kmeans_batch
        for layer in model.get_nativebit_layers():
            w_flat = layer.weight.data.view(-1)
            if layer._padded_len > layer.total_weights:
                w_flat = F.pad(w_flat, (0, layer._padded_len - layer.total_weights))
            w_blocks = w_flat.view(layer.num_blocks, layer.block_size)
            layer.codebook.data.copy_(init_codebook_kmeans_batch(w_blocks, layer.n_entries))

    if device.type == "cuda":
        model = torch.compile(model)

    results = train(model, config, device, args.name, args.log_dir, args.data_dir)
    print(f"\n=== Final Results ({args.name}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
