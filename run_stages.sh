#!/bin/bash
# =============================================================================
# Run Individual Stages of OMAGD Pipeline
# =============================================================================
# 
# Usage:
#   ./run_stages.sh stage1 8m_vs_9m      # Run Stage 1 (Tokenizer)
#   ./run_stages.sh stage2 8m_vs_9m      # Run Stage 2 (Mask Predictor)
#   ./run_stages.sh stage3 8m_vs_9m      # Run Stage 3 (QMIX + Reconstruction)
#   ./run_stages.sh pretrain 8m_vs_9m    # Run Stage 1 + Stage 2
#   ./run_stages.sh all 8m_vs_9m         # Run full pipeline
#
# For baselines, use existing scripts:
#   ./run_gnn_baseline.sh
#   ./run_gnn_full_obs.sh
#
# =============================================================================

set -e

STAGE="${1:-all}"
MAP_NAME="${2:-8m_vs_9m}"
SEED="${3:-1}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "$0")"

case "${STAGE}" in
    stage1|stage2|stage3)
        echo "Running ${STAGE} on ${MAP_NAME} with seed ${SEED}..."
        python src/run_pipeline.py \
            --map "${MAP_NAME}" \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --stages "${STAGE}"
        ;;
    all)
        echo "Running full pipeline (stage1→stage2→stage3) on ${MAP_NAME} with seed ${SEED}..."
        python src/run_pipeline.py \
            --map "${MAP_NAME}" \
            --seed ${SEED} \
            --gpu ${GPU_ID}
        ;;
    pretrain)
        echo "Running pretrain (stage1+stage2) on ${MAP_NAME} with seed ${SEED}..."
        python src/run_pipeline.py \
            --map "${MAP_NAME}" \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --stages "stage1,stage2"
        ;;
    *)
        echo "Unknown stage: ${STAGE}"
        echo ""
        echo "Valid stages:"
        echo "  stage1   - Train Tokenizer (VQ-VAE)"
        echo "  stage2   - Train Mask Predictor"
        echo "  stage3   - QMIX with Reconstruction"
        echo "  pretrain - Stage 1 + Stage 2"
        echo "  all      - Full pipeline"
        exit 1
        ;;
esac
