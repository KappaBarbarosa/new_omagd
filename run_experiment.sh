#!/bin/bash
# =============================================================================
# OMAGD Experiment Pipeline Runner
# =============================================================================
# 
# This script runs the OMAGD training pipeline:
#   Stage 1: Train Tokenizer (VQ-VAE) + Collect pretrain buffer
#   Stage 2: Train Mask Predictor (with frozen tokenizer, shared buffer)
#   Stage 3: QMIX Training with Graph Reconstruction
#
# For baselines, use the existing scripts:
#   ./run_gnn_baseline.sh
#   ./run_gnn_full_obs.sh
#
# Usage:
#   ./run_experiment.sh                    # Run with defaults (8m_vs_9m, seed=1)
#   ./run_experiment.sh 3m                 # Specify map
#   ./run_experiment.sh 8m_vs_9m 1 2 3     # Specify map and multiple seeds
#
# Examples:
#   # Run full pipeline (auto-resumes if previous run failed)
#   ./run_experiment.sh 8m_vs_9m
#
#   # Specify stage3 training steps
#   ./run_experiment.sh 8m_vs_9m --t-max 2000000
#
#   # Force start new experiment (don't auto-resume)
#   ./run_experiment.sh 8m_vs_9m --new
#
#   # Run only stage1 and stage2 (pretrain only)
#   ./run_experiment.sh 8m_vs_9m 1 --stages stage1,stage2
#
#   # Dry run (show commands without executing)
#   ./run_experiment.sh 8m_vs_9m --dry-run
#
# =============================================================================

set -e  # Exit on error

# Default values
MAP_NAME="${1:-8m_vs_9m}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Check if first arg is a flag
if [[ "$1" == "--"* ]]; then
    MAP_NAME="8m_vs_9m"
    EXTRA_ARGS="$@"
else
    shift 2>/dev/null || true
    
    # Collect seeds (numeric arguments)
    SEEDS=()
    EXTRA_ARGS=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            [0-9]*)
                SEEDS+=("$1")
                ;;
            *)
                EXTRA_ARGS+=("$1")
                ;;
        esac
        shift
    done
    
    # Default seed if none provided
    if [ ${#SEEDS[@]} -eq 0 ]; then
        SEEDS=(1)
    fi
    
    EXTRA_ARGS="${EXTRA_ARGS[*]}"
fi

# Set up environment
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Show configuration
echo "=============================================="
echo "  OMAGD Experiment Pipeline"
echo "=============================================="
echo "  Map:      ${MAP_NAME}"
echo "  GPU:      ${GPU_ID}"
if [ -n "${SEEDS[*]}" ]; then
    echo "  Seeds:    ${SEEDS[*]}"
fi
echo "  Extra:    ${EXTRA_ARGS}"
echo "=============================================="

# Build seed arguments
SEED_ARGS=""
if [ ${#SEEDS[@]} -gt 0 ]; then
    SEED_ARGS="--seed ${SEEDS[*]}"
fi

# Run the pipeline
cd "$(dirname "$0")"

python src/run_pipeline.py \
    --map "${MAP_NAME}" \
    --gpu "${GPU_ID}" \
    ${SEED_ARGS} \
    ${EXTRA_ARGS}
