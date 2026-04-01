"""Training metrics logger — saves structured logs to JSON lines and CSV."""

import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from .layers import NativeBitLinear


class TrainingLogger:
    """Logs training metrics to JSONL and a summary CSV.

    Logs every N steps: loss, perplexity, codebook stats, gradient ratios.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / f"{experiment_name}.jsonl"
        self.csv_path = self.log_dir / f"{experiment_name}.csv"
        self.codebook_dir = self.log_dir / f"{experiment_name}_codebooks"
        self.codebook_dir.mkdir(exist_ok=True)

        self.experiment_name = experiment_name
        self._csv_writer = None
        self._csv_file = None
        self._csv_header_written = False
        self.start_time = time.time()

    def log_header(self, config) -> None:
        """Emit a header line with config metadata for the dashboard."""
        header = {
            "type": "header",
            "max_steps": getattr(config, "max_steps", 0),
            "use_nativebit": True,  # overridden by caller if needed
            "n_codebook": getattr(config, "n_codebook", 8),
            "block_size": getattr(config, "block_size", 64),
            "batch_size": getattr(config, "batch_size", 8),
            "lr": getattr(config, "lr", 0),
            "dataset": getattr(config, "dataset", "wikitext-2"),
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(header) + "\n")

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        model: torch.nn.Module,
        grad_info: dict | None = None,
    ) -> dict:
        """Log metrics for a training step.

        Args:
            step: current training step.
            loss: training loss value.
            lr: current learning rate.
            model: the model (to extract codebook stats).
            grad_info: optional dict with gradient magnitude info.

        Returns:
            The logged metrics dict.
        """
        ppl = math.exp(min(loss, 20))  # cap to avoid overflow
        elapsed = time.time() - self.start_time

        record = {
            "step": step,
            "loss": round(loss, 6),
            "perplexity": round(ppl, 2),
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
        }

        # Codebook stats from NativeBitLinear layers (batched to minimize GPU syncs)
        nb_layers = [m for m in model.modules() if isinstance(m, NativeBitLinear)]
        if nb_layers:
            total_dead = 0
            total_entries = 0
            for layer in nb_layers:
                total_u = layer.utilization.sum(dim=-1, keepdim=True).clamp(min=1).float()
                frac = layer.utilization.float() / total_u
                total_dead += (frac < 0.01).sum()  # keep on GPU
                total_entries += layer.num_blocks * layer.n_entries

            # Single GPU sync for dead count
            record["dead_entries"] = int(total_dead.item())
            record["total_entries"] = total_entries
            record["dead_pct"] = round(record["dead_entries"] / max(total_entries, 1) * 100, 2)

        # Gradient info
        if grad_info:
            record.update(grad_info)

        # Write JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Write CSV
        self._write_csv(record)

        return record

    def save_codebook_snapshot(self, step: int, model: torch.nn.Module) -> None:
        """Save all codebook values for later visualization (binary format)."""
        nb_layers = [m for m in model.modules() if isinstance(m, NativeBitLinear)]
        snapshot = {}
        for i, layer in enumerate(nb_layers):
            snapshot[f"layer_{i}"] = layer.codebook.detach().cpu()
            snapshot[f"layer_{i}_util"] = layer.utilization.detach().cpu()

        path = self.codebook_dir / f"step_{step:06d}.pt"
        torch.save(snapshot, path)

    def _write_csv(self, record: dict) -> None:
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")
        if not self._csv_header_written:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(record.keys()))
            self._csv_writer.writeheader()
            self._csv_header_written = True
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()

    def __del__(self):
        self.close()


def compute_gradient_info(model: torch.nn.Module) -> dict:
    """Compute gradient magnitude ratio: codebook grads vs main weight grads.

    Accumulates norms on-device, single .item() at the end to avoid
    per-parameter XLA graph breaks.
    """
    device = next(model.parameters()).device
    cb_grad_norm = torch.tensor(0.0, device=device)
    cb_count = 0
    w_grad_norm = torch.tensor(0.0, device=device)
    w_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.data.float().norm()
        if "codebook" in name:
            cb_grad_norm = cb_grad_norm + g
            cb_count += 1
        elif "weight" in name and "ln" not in name and "tok_emb" not in name:
            w_grad_norm = w_grad_norm + g
            w_count += 1

    # Single sync point
    cb_avg = (cb_grad_norm / max(cb_count, 1)).item()
    w_avg = (w_grad_norm / max(w_count, 1)).item()
    ratio = cb_avg / max(w_avg, 1e-12)

    return {
        "grad_cb_avg": round(cb_avg, 6),
        "grad_w_avg": round(w_avg, 6),
        "grad_ratio_cb_w": round(ratio, 4),
    }
