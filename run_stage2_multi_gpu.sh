#!/bin/bash
# =============================================================================
# Stage 2 Multi-GPU Training Script - Train Mask Predictor
# =============================================================================
# Usage: ./run_stage2_multi_gpu.sh [num_gpus] [map_name] [tokenizer_path]
# Example: ./run_stage2_multi_gpu.sh 4 5m_vs_6m results/models/.../pretrain_stage1_best
#
# This script uses torchrun for distributed training across multiple GPUs.
# Data is collected only on the main process (GPU 0) and broadcasted to others.
# The tokenizer from Stage 1 must be provided and will be frozen during training.

export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d

# Configuration
NUM_GPUS=${1:-4}  # Number of GPUs to use (default: 4)
MAP_NAME=${2:-"5m_vs_6m"}  # Map name (default: 5m_vs_6m)
MASTER_PORT=${4:-29501}  # Master port for distributed training

# Tokenizer path from Stage 1 (update this path!)
TOKENIZER_PATH=${3:-"results/models/sc2_5m_vs_6m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_origin__2026-01-12_16-53-01/pretrain_stage1_best"}

# Set visible GPUs (adjust based on your system)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "=============================================="
echo " Stage 2: Training Mask Predictor (Multi-GPU)"
echo "=============================================="
echo " Map: ${MAP_NAME}"
echo " GPUs: ${NUM_GPUS}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " Master Port: ${MASTER_PORT}"
echo " Tokenizer Path: ${TOKENIZER_PATH}"
echo "=============================================="

# Verify tokenizer exists
if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "ERROR: Tokenizer path does not exist: ${TOKENIZER_PATH}"
    echo "Please run Stage 1 first or provide correct tokenizer path."
    exit 1
fi

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    src/main.py --config=omagd --env-config=sc2 \
    with env_args.map_name=${MAP_NAME} \
    seed=1 \
    mac="n_mac" \
    recontructer_stage=stage2 \
    use_graph_reconstruction=True \
    cpu_inference=False \
    pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    use_wandb=True \
    pretrain_only=True

echo ""
echo "=============================================="
echo " Stage 2 training completed on ${MAP_NAME}"
echo "=============================================="
