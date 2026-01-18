#!/bin/bash
# =============================================================================
# Stage 1 Multi-GPU Training Script - Train Tokenizer (VQ-VAE)
# =============================================================================
# Usage: ./run_stage1_multi_gpu.sh [num_gpus] [map_name]
# Example: ./run_stage1_multi_gpu.sh 4 5m_vs_6m
#
# This script uses torchrun for distributed training across multiple GPUs.
# Data is collected only on the main process (GPU 0) and broadcasted to others.

export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d

# Configuration
NUM_GPUS=${1:-4}  # Number of GPUs to use (default: 4)
MAP_NAME=${2:-"5m_vs_6m"}  # Map name (default: 5m_vs_6m)
MASTER_PORT=${3:-29500}  # Master port for distributed training

# Set visible GPUs (adjust based on your system)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "=============================================="
echo " Stage 1: Training Tokenizer (Multi-GPU)"
echo "=============================================="
echo " Map: ${MAP_NAME}"
echo " GPUs: ${NUM_GPUS}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " Master Port: ${MASTER_PORT}"
echo "=============================================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    src/main.py --config=omagd --env-config=sc2 \
    with env_args.map_name=${MAP_NAME} \
    mac=n_mac \
    t_max=500000 \
    recontructer_stage=stage1 \
    pretrain_only=True \
    use_graph_reconstruction=True \
    use_wandb=True \
    cpu_inference=False

echo ""
echo "=============================================="
echo " Stage 1 training completed on ${MAP_NAME}"
echo "=============================================="
