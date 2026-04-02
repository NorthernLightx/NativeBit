#!/bin/bash
# provision_2b.sh — Provision TPU v6e-8 instances for 2B experiments
#
# Creates 6 spot instances (3 float + 3 NB seeds), uploads code, installs deps.
# Uses europe-west4-a zone with v2-alpha-tpuv6e runtime.
#
# Usage:
#   bash scripts/provision_2b.sh          # provision all 6
#   bash scripts/provision_2b.sh create   # create instances only
#   bash scripts/provision_2b.sh setup    # upload code + install deps only
#   bash scripts/provision_2b.sh launch   # start training on all instances
#   bash scripts/provision_2b.sh delete   # tear down all instances

set -euo pipefail

PROJECT="REDACTED_PROJECT_ID"
ZONE="europe-west4-a"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
SEEDS=(42 137 256)
MODES=(float float float nb nb nb)
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/$USER/NativeBit"

# Instance names: nb-2b-float-s42, nb-2b-nb-s42, etc
instances=()
for i in "${!SEEDS[@]}"; do
    instances+=("nb-2b-float-s${SEEDS[$i]}")
done
for i in "${!SEEDS[@]}"; do
    instances+=("nb-2b-nb-s${SEEDS[$i]}")
done

ssh_cmd() {
    local inst=$1; shift
    gcloud compute tpus tpu-vm ssh "$inst" \
        --project="$PROJECT" --zone="$ZONE" --worker=0 \
        --command="$*"
}

do_create() {
    echo "=== Creating ${#instances[@]} TPU instances ==="
    for inst in "${instances[@]}"; do
        echo "  Creating $inst..."
        gcloud compute tpus queued-resources create "$inst" \
            --node-id="$inst" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --accelerator-type="$ACCEL" \
            --runtime-version="$RUNTIME" \
            --spot &
    done
    wait
    echo "  Waiting for ACTIVE state..."
    for inst in "${instances[@]}"; do
        while true; do
            state=$(gcloud compute tpus queued-resources describe "$inst" \
                --project="$PROJECT" --zone="$ZONE" \
                --format="value(state.state)" 2>/dev/null || echo "PENDING")
            if [[ "$state" == "ACTIVE" ]]; then
                echo "  $inst: ACTIVE"
                break
            fi
            sleep 10
        done
    done
}

do_setup() {
    echo "=== Setting up ${#instances[@]} instances ==="
    for inst in "${instances[@]}"; do
        echo "  [$inst] Accepting SSH host key..."
        echo y | gcloud compute tpus tpu-vm ssh "$inst" \
            --project="$PROJECT" --zone="$ZONE" --worker=0 \
            --command="echo connected" 2>/dev/null || true

        echo "  [$inst] Creating dirs..."
        ssh_cmd "$inst" "mkdir -p $REMOTE_DIR"

        echo "  [$inst] Uploading code..."
        gcloud compute tpus tpu-vm scp --recurse \
            "$CODE_DIR/nativebit_jax" "$CODE_DIR/configs" \
            "$CODE_DIR/benchmarks" "$CODE_DIR/scripts" \
            "$inst:$REMOTE_DIR/" \
            --project="$PROJECT" --zone="$ZONE" --worker=0

        echo "  [$inst] Installing deps..."
        ssh_cmd "$inst" "cd $REMOTE_DIR && pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install -q flax optax orbax-checkpoint tiktoken tqdm datasets" &
    done
    wait
    echo "  All instances ready."
}

do_launch() {
    echo "=== Launching experiments ==="
    for i in "${!instances[@]}"; do
        inst="${instances[$i]}"
        mode="${MODES[$i]}"
        # Extract seed from instance name
        seed=$(echo "$inst" | grep -oP 's\K[0-9]+')
        echo "  [$inst] Launching $mode seed=$seed in tmux..."
        ssh_cmd "$inst" "cd $REMOTE_DIR && tmux new-session -d -s train 'bash scripts/run_2b.sh $mode $seed 2>&1 | tee logs/${inst}.log'"
    done
    echo "  All experiments launched. Check with:"
    echo "    gcloud compute tpus tpu-vm ssh <instance> --command='tmux attach -t train' ..."
}

do_delete() {
    echo "=== Deleting all instances ==="
    for inst in "${instances[@]}"; do
        echo "  Deleting $inst..."
        gcloud compute tpus queued-resources delete "$inst" \
            --project="$PROJECT" --zone="$ZONE" --quiet --force &
    done
    wait
    echo "  All deleted."
}

case "${1:-all}" in
    create) do_create ;;
    setup)  do_setup ;;
    launch) do_launch ;;
    delete) do_delete ;;
    all)    do_create; do_setup; do_launch ;;
    *)      echo "Usage: $0 <create|setup|launch|delete|all>"; exit 1 ;;
esac
