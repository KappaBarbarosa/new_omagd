#!/bin/bash
# =============================================================================
# Transfer Learning Script - Test tokenizer on different maps
# =============================================================================
# Usage: ./run_transfer.sh [tokenizer_path]
# Tests how well a tokenizer trained on one map transfers to other maps.
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
export CUDA_VISIBLE_DEVICES=1

TOKENIZER_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-14_23-40-23/pretrain_stage1_best"

if [ -z "$TOKENIZER_PATH" ]; then
    echo "ERROR: Please provide path to pretrained tokenizer"
    echo "Usage: ./run_transfer.sh [tokenizer_path]"
    echo "Example: ./run_transfer.sh results/models/.../pretrain_stage1_best"
    exit 1
fi

# List of target maps for transfer testing
TARGET_MAPS=("8m" "5m_vs_6m" "8m_vs_9m" "10m_vs_11m")

for TARGET_MAP in "${TARGET_MAPS[@]}"; do
    echo "=============================================="
    echo " Transfer Learning: Testing on ${TARGET_MAP}"
    echo " Using tokenizer from: ${TOKENIZER_PATH}"
    echo "=============================================="
    
    python src/main.py --config=omagd --env-config=sc2 \
        with env_args.map_name=${TARGET_MAP} \
        recontructer_stage=stage1 \
        use_graph_reconstruction=True \
        pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    
    echo "Completed transfer test on ${TARGET_MAP}"
    echo ""
done

echo "All transfer experiments completed!"
