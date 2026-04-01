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
from nativebit.device import (
    get_device, is_tpu, is_cuda, amp_context, needs_grad_scaler,
    mark_step, get_memory_info, device_name,
)
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
def run_evaluation(model: nn.Module, loader, device: torch.device,
                   max_batches: int = 0) -> float:
    """Compute average cross-entropy loss on a dataloader.

    Args:
        max_batches: limit eval to this many batches (0 = all). Use on TPU
                     to avoid slow XLA recompilation over large eval sets.
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss * y.numel()
        total_tokens += y.numel()
        mark_step()
    model.train()
    # Single .item() at end — avoids per-batch graph breaks on XLA
    return total_loss.item() / max(total_tokens, 1)


def clip_codebook_grads(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Clip codebook parameter gradients separately.

    Returns the total norm as a tensor (no .item() to avoid XLA graph breaks).
    """
    params = [p for n, p in model.named_parameters() if "codebook" in n and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def _abort(msg: str) -> None:
    """Print preflight/gate failure and exit."""
    print(f"\n{'='*60}")
    print(f"  ABORT: {msg}")
    print(f"{'='*60}\n")
    sys.exit(1)


# Minimum steps/s thresholds by device type and model size (n_embd).
# Below these values, something is fundamentally wrong (wrong config,
# XLA recompilation, etc.) and continuing wastes resources.
_MIN_THROUGHPUT = {
    # (device_type, min_n_embd): min_steps_per_sec
    ("xla", 768): 10,     # TPU + 125M model
    ("xla", 384): 20,     # TPU + 48M model
    ("xla", 192): 50,     # TPU + tiny model (should not be run on TPU)
    ("cuda", 192): 2,     # RTX 3070 + tiny model
    ("cuda", 384): 1,     # RTX 3070 + medium model
}


def _get_min_throughput(device: torch.device, n_embd: int) -> float:
    """Look up minimum acceptable steps/s for this device + model size."""
    dtype = device.type
    # Find the closest n_embd match for this device type
    candidates = [(k, v) for k, v in _MIN_THROUGHPUT.items() if k[0] == dtype]
    if not candidates:
        return 0.5  # conservative fallback: at least 0.5 steps/s
    # Pick the entry with closest n_embd
    candidates.sort(key=lambda kv: abs(kv[0][1] - n_embd))
    return candidates[0][1]


def run_preflight(model: nn.Module, config, device: torch.device,
                  train_loader, optimizer, scheduler, scaler,
                  use_nativebit: bool) -> tuple[float, float]:
    """Run 50 warmup steps to validate config + throughput before committing.

    Returns (steps_per_sec, initial_loss) for use by early gates.
    Aborts the process on failure — no interactive prompts.
    """
    n_embd = getattr(config, "n_embd", 0)
    preflight_steps = 50

    # --- Static config checks ---
    if is_tpu(device) and n_embd < 384:
        _abort(
            f"n_embd={n_embd} is too small for TPU. "
            f"TPU systolic arrays need n_embd>=384 (ideally >=768) "
            f"to saturate. Use --config tpu-small or tpu-medium."
        )

    if is_tpu(device) and config.batch_size < 16:
        _abort(
            f"batch_size={config.batch_size} is too small for TPU. "
            f"Need batch_size>=16 to amortize XLA overhead."
        )

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    tokens_per_step = config.batch_size * config.context_len

    print(f"  PREFLIGHT: {preflight_steps} steps, {num_params:.1f}M params, "
          f"{tokens_per_step} tok/step on {device_name(device)}")

    # --- Throughput measurement (skip first 5 for XLA compile warmup) ---
    model.train()
    train_iter = iter(train_loader)
    warmup_skip = 10 if is_tpu(device) else 3
    first_loss = None

    for i in range(preflight_steps):
        optimizer.zero_grad()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        elif is_tpu(device):
            loss.backward()
            from nativebit.device import optimizer_step as xla_opt_step
            xla_opt_step(optimizer)
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        mark_step()

        if i == warmup_skip:
            from nativebit.device import sync_device
            sync_device(device)
            t_start = time.time()
            first_loss = loss.item()

    from nativebit.device import sync_device
    sync_device(device)
    t_end = time.time()

    measured_steps = preflight_steps - warmup_skip - 1
    elapsed = t_end - t_start
    steps_per_sec = measured_steps / elapsed if elapsed > 0 else 0
    min_throughput = _get_min_throughput(device, n_embd)
    eta_hours = (config.max_steps - preflight_steps) / steps_per_sec / 3600 if steps_per_sec > 0 else float("inf")

    print(f"  PREFLIGHT: {steps_per_sec:.1f} steps/s "
          f"(min={min_throughput:.0f}), ETA={eta_hours:.1f}h for {config.max_steps} steps")

    if steps_per_sec < min_throughput:
        _abort(
            f"Throughput {steps_per_sec:.1f} steps/s is below minimum "
            f"{min_throughput:.0f} steps/s for {device_name(device)} + n_embd={n_embd}. "
            f"Model is too small for this hardware or XLA is recompiling. "
            f"Use a larger config (--config tpu-medium) or check for graph breaks."
        )

    # --- NativeBit-specific: check initial dead entries ---
    if use_nativebit and hasattr(model, "update_all_utilization"):
        model.update_all_utilization()
        nb_layers = model.get_nativebit_layers()
        total_dead, total_entries = 0, 0
        for layer in nb_layers:
            stats = layer.get_utilization_stats()
            total_dead += stats["dead_entries"]
            total_entries += stats["total_entries"]
        dead_pct = total_dead / max(total_entries, 1) * 100
        if dead_pct > 25:
            _abort(
                f"Dead entries at {dead_pct:.1f}% after {preflight_steps} steps. "
                f"Codebook init is broken or block_size/n_codebook is misconfigured."
            )
        print(f"  PREFLIGHT: dead entries={total_dead}/{total_entries} ({dead_pct:.1f}%)")

    print(f"  PREFLIGHT: PASSED — continuing to full training\n")
    return steps_per_sec, first_loss if first_loss is not None else 0.0


def train(model: nn.Module, config, device: torch.device,
          experiment_name: str = "nativebit", log_dir: str = "logs",
          data_dir: str = "data", use_nativebit: bool = True) -> dict:
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

    # Emit header line for dashboard (max_steps, config metadata)
    logger.log_header(config)

    # Model summary
    counts = model.count_parameters() if hasattr(model, "count_parameters") else {}
    print(f"\n=== {experiment_name} ===")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print(f"  Device: {device}")
    print(f"  Steps: {config.max_steps}")
    print()

    # AMP — GradScaler only for CUDA fp16; TPU bf16 doesn't need it
    use_scaler = needs_grad_scaler(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # --- Preflight: validate config + throughput before committing ---
    preflight_sps, preflight_loss = run_preflight(
        model, config, device, train_loader, optimizer, scheduler, scaler,
        use_nativebit=use_nativebit,
    )

    # Training loop (continues from where preflight left off)
    step = 0
    train_iter = iter(train_loader)
    # Track for early gates
    _gate_initial_loss = preflight_loss
    _gate_preflight_sps = preflight_sps
    _gate_checked_200 = False
    _gate_checked_500 = False

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
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Defer .item() — accumulate loss on device, sync only at log steps
        loss_val = loss.detach()

        # Gradient clipping + optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        clip_codebook_grads(model, config.codebook_grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        elif is_tpu(device):
            # xm.optimizer_step includes mark_step()
            from nativebit.device import optimizer_step as xla_opt_step
            xla_opt_step(optimizer)
        else:
            optimizer.step()

        scheduler.step()
        mark_step()  # XLA: trigger execution; no-op on CUDA/CPU

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

        # Logging — .item() only at log steps to avoid XLA graph breaks
        if step % config.log_every == 0:
            loss_scalar = loss_val.item() if torch.is_tensor(loss_val) else loss_val

            # Fast fail (only checked at log steps to avoid per-step .item())
            if math.isnan(loss_scalar) or loss_scalar > 1000:
                print("FAIL")
                sys.exit(1)

            grad_info = compute_gradient_info(model) if hasattr(model, "get_nativebit_layers") else None
            current_lr = optimizer.param_groups[0]["lr"]
            record = logger.log_step(step, loss_scalar, current_lr, model, grad_info)

            ppl_str = f"{record['perplexity']:>10.2f}"
            dead_str = f"dead={record.get('dead_entries', 0)}" if "dead_entries" in record else ""
            grad_str = f"cb/w={record.get('grad_ratio_cb_w', 0):.3f}" if "grad_ratio_cb_w" in record else ""
            print(f"  step={step:>5d}  loss={record['loss']:.4f}  ppl={ppl_str}  lr={record['lr']:.2e}  {dead_str}  {grad_str}")

            if step % (config.log_every * 4) == 0:
                logger.save_codebook_snapshot(step, model)

            # --- Early gates: abort hopeless runs ---
            if step == 200 and not _gate_checked_200:
                _gate_checked_200 = True
                # Check dead entries
                if use_nativebit and "dead_pct" in record and record["dead_pct"] > 20:
                    _abort(
                        f"Dead entries at {record['dead_pct']:.1f}% by step 200. "
                        f"Codebook collapse in progress. "
                        f"Try: lower codebook_lr, smaller block_size, or fewer n_codebook entries."
                    )
                # Check loss decreased from init
                if _gate_initial_loss > 0 and loss_scalar > _gate_initial_loss * 0.95:
                    _abort(
                        f"Loss barely decreased by step 200: "
                        f"{_gate_initial_loss:.3f} -> {loss_scalar:.3f}. "
                        f"Learning rate may be too low or model is broken."
                    )

            if step == 500 and not _gate_checked_500:
                _gate_checked_500 = True
                # Check dead entries again with stricter threshold
                if use_nativebit and "dead_pct" in record and record["dead_pct"] > 15:
                    _abort(
                        f"Dead entries still at {record['dead_pct']:.1f}% by step 500. "
                        f"Revival mechanism is not keeping up. "
                        f"This run will not converge well."
                    )
                # Loss should have dropped significantly by now
                if _gate_initial_loss > 0 and loss_scalar > _gate_initial_loss * 0.80:
                    _abort(
                        f"Loss only dropped {(1 - loss_scalar/_gate_initial_loss)*100:.0f}% by step 500 "
                        f"({_gate_initial_loss:.3f} -> {loss_scalar:.3f}). "
                        f"Expected at least 20% reduction. Check config."
                    )

        if step >= config.max_steps:
            break
        step += 1

    # Final metrics
    print(f"\nFinal metrics for {experiment_name}...")
    # Limit eval batches on TPU to avoid slow XLA recompilation
    eval_max = 50 if device.type == "xla" else 0
    val_loss = run_evaluation(model, valid_loader, device, max_batches=eval_max)
    test_loss = run_evaluation(model, test_loader, device, max_batches=eval_max)
    val_ppl = math.exp(min(val_loss, 20))
    test_ppl = math.exp(min(test_loss, 20))
    # Skip BPB on TPU — too slow (224K sequences × XLA recompilation)
    if device.type == "xla":
        val_bpb = 0.0
    else:
        val_bpb = compute_bpb(model, valid_loader, device)

    print(f"  Val loss: {val_loss:.4f}  Val PPL: {val_ppl:.2f}")
    print(f"  Test loss: {test_loss:.4f}  Test PPL: {test_ppl:.2f}")
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"num_steps:        {step}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    mem = get_memory_info(device)
    if "peak_mb" in mem:
        print(f"peak_memory_mb:   {mem['peak_mb']:.1f}")

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

    train_loss = loss_val.item() if torch.is_tensor(loss_val) else loss_val
    return {"train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl,
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
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "tpu-small", "tpu-medium", "tpu-large", "tpu-xl"],
                        help="Config preset (default for RTX 3070, tpu-* for Cloud TPU)")
    args = parser.parse_args()

    config_map = {
        "default": DefaultConfig,
    }
    # Lazy import TPU configs — avoids import on machines without them
    if args.config.startswith("tpu"):
        from configs.tpu import TPUSmallConfig, TPUMediumConfig, TPULargeConfig, TPUXLConfig
        config_map.update({
            "tpu-small": TPUSmallConfig,
            "tpu-medium": TPUMediumConfig,
            "tpu-large": TPULargeConfig,
            "tpu-xl": TPUXLConfig,
        })

    config = config_map[args.config]()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    config.seed = args.seed
    set_seed(config.seed)

    device = get_device()
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

    # torch.compile on CUDA only — XLA compiles automatically
    if is_cuda(device):
        model = torch.compile(model)

    results = train(model, config, device, args.name, args.log_dir, args.data_dir,
                    use_nativebit=use_nativebit)
    print(f"\n=== Final Results ({args.name}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
