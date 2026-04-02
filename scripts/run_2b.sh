#!/bin/bash
# run_2b.sh — 2.2B NativeBit experiment suite on TPU v6e-8
#
# Run on the TPU VM after uploading code. Each invocation does ONE job.
# Parallelize by running on separate TPU instances.
#
# Usage:
#   bash scripts/run_2b.sh float 42       # float baseline, seed 42
#   bash scripts/run_2b.sh nb 42          # NativeBit 3-bit, seed 42
#   bash scripts/run_2b.sh posthoc 42     # post-hoc on float checkpoint
#   bash scripts/run_2b.sh all 42         # all three sequentially
#
# Full parallel plan (6 TPU v6e-8 instances, 48/64 chip quota):
#   Instance 1: bash scripts/run_2b.sh float 42
#   Instance 2: bash scripts/run_2b.sh float 137
#   Instance 3: bash scripts/run_2b.sh float 256
#   Instance 4: bash scripts/run_2b.sh nb 42
#   Instance 5: bash scripts/run_2b.sh nb 137
#   Instance 6: bash scripts/run_2b.sh nb 256
#   Then on any instance: bash scripts/run_2b.sh posthoc 42  (after float done)

set -euo pipefail

MODE=${1:?Usage: run_2b.sh <float|nb|posthoc|all> <seed>}
SEED=${2:-42}

echo "============================================"
echo "  NativeBit 2.2B — mode=$MODE seed=$SEED"
echo "  Device: $(python3 -c 'import jax; print(jax.devices()[0])')"
echo "  Devices: $(python3 -c 'import jax; print(jax.device_count())')"
echo "============================================"

run_float() {
    echo "[$(date)] Starting float baseline (seed=$SEED)..."
    python -m nativebit_jax.train \
        --config tpu-2b \
        --no-nativebit \
        --name "2b_float_s${SEED}" \
        --seed "$SEED"
    echo "[$(date)] Float baseline done."
}

run_nb() {
    echo "[$(date)] Starting NativeBit 3-bit (seed=$SEED)..."
    python -m nativebit_jax.train \
        --config tpu-2b \
        --name "2b_nb3_s${SEED}" \
        --seed "$SEED"
    echo "[$(date)] NativeBit 3-bit done."
}

run_posthoc() {
    echo "[$(date)] Starting post-hoc benchmark (seed=$SEED)..."
    python benchmarks/benchmark_posthoc_2b.py --seed "$SEED"
    echo "[$(date)] Post-hoc benchmark done."
}

case "$MODE" in
    float)   run_float ;;
    nb)      run_nb ;;
    posthoc) run_posthoc ;;
    all)     run_float; run_nb; run_posthoc ;;
    *)       echo "Unknown mode: $MODE"; exit 1 ;;
esac

echo "[$(date)] Done. Results in logs/"
