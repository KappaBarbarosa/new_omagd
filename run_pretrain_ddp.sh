#!/bin/bash
#==============================================
# Multi-GPU Pretraining with DDP
# Stage 1: Tokenizer training
# Stage 2: Mask predictor training
#==============================================

# Configure GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs to use
NUM_GPUS=2

# DDP port
DDP_PORT=12355

# Stage selection: stage1 or stage2
STAGE="stage1"

# Tokenizer path (only needed for stage2)
TOKENIZER_PATH=""

cd /home/marl2025/new_omagd

echo "======================================"
echo "Starting Multi-GPU Pretraining with DDP"
echo "Stage: ${STAGE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "======================================"

# Stage 1: Tokenizer training
if [ "$STAGE" == "stage1" ]; then
    python src/main.py \
        --config=omagd_origin \
        --env-config=sc2 \
        with \
        env_args.map_name=5m_vs_6m \
        use_graph_reconstruction=True \
        recontructer_stage=stage1 \
        pretrain_only=True \
        use_cuda=True \
        graph_pretrain_epochs=200 \
        graph_pretrain_episodes=5000 \
        graph_pretrain_batch_size=128 \
        save_model=True \
        use_wandb=True \
        run=ddp \
        learner=nq_graph_learner_ddp \
        use_ddp=True \
        world_size=${NUM_GPUS} \
        ddp_port=${DDP_PORT}
fi

# Stage 2: Mask predictor training
if [ "$STAGE" == "stage2" ]; then
    python src/main.py \
        --config=omagd_hungarian \
        --env-config=sc2 \
        with \
        env_args.map_name=5m_vs_6m \
        use_graph_reconstruction=True \
        recontructer_stage=stage2 \
        pretrain_only=True \
        pretrained_tokenizer_path="${TOKENIZER_PATH}" \
        use_cuda=True \
        graph_pretrain_epochs=200 \
        graph_pretrain_episodes=5000 \
        graph_pretrain_batch_size=128 \
        save_model=True \
        use_wandb=True \
        run=ddp \
        learner=nq_graph_learner_ddp \
        use_ddp=True \
        world_size=${NUM_GPUS} \
        ddp_port=${DDP_PORT}
fi
