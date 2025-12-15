#!/bin/bash
# =============================================================================
# Stage 1 Training Script - Train Tokenizer (VQ-VAE)
# =============================================================================
# Runs tokenizer training on multiple maps
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
export CUDA_VISIBLE_DEVICES=0

# List of maps to train on
MAPS=("8m" "5m_vs_6m" "8m_vs_9m" "10m_vs_11m")

for MAP_NAME in "${MAPS[@]}"; do
    echo "=============================================="
    echo " Stage 1: Training Tokenizer on ${MAP_NAME}"
    echo "=============================================="
    
    python src/main.py --config=omagd --env-config=sc2 \
        with env_args.map_name=${MAP_NAME} \
        recontructer_stage=stage1 \
        use_graph_reconstruction=True \
        use_wandb=True
    
    echo "Completed training on ${MAP_NAME}"
    echo ""
done

echo "All Stage 1 training completed!"
