#!/bin/bash
#==============================================
# GNN Stage 3: Multi-GPU QMIX Training with DDP
# Uses DistributedDataParallel for cross-GPU training
#==============================================

# Configure GPUs - modify this based on your available GPUs
# Example: "0,1" for 2 GPUs, "0,1,2,3" for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs to use (should match CUDA_VISIBLE_DEVICES)
NUM_GPUS=2

# DDP port for process communication (change if port conflict)
DDP_PORT=12355

# Pretrained model paths
TOKENIZER_PATH="results/models/sc2_5m_vs_6m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_origin__2026-01-12_16-53-01/pretrain_stage1_best"
MASK_PREDICTOR_PATH="results/models/sc2_5m_vs_6m-obs_aid=1-obs_act=1/algo=omagd_hungarian-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_hungarian__2026-01-12_17-23-58/pretrain_stage2_best"

cd /home/marl2025/new_omagd

echo "======================================"
echo "Starting Multi-GPU Training with DDP"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "DDP Port: ${DDP_PORT}"
echo "======================================"

python src/main.py \
    --config=omagd \
    --env-config=sc2 \
    with \
    env_args.map_name=5m_vs_6m \
    mac=gnn_graph_mac \
    agent=gnn_rnn \
    gnn_layer_num=2 \
    use_graph_reconstruction=True \
    use_cuda=True \
    cpu_inference=False \
    pretrain_only=False \
    recontructer_stage=stage3 \
    pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    pretrained_mask_predictor_path="${MASK_PREDICTOR_PATH}" \
    t_max=5000000 \
    test_interval=10000 \
    save_model=True \
    save_model_interval=100000 \
    log_interval=10000 \
    local_results_path="results/gnn_stage3_ddp" \
    use_wandb=True \
    run=ddp \
    learner=nq_graph_learner_ddp \
    use_ddp=True \
    world_size=${NUM_GPUS} \
    ddp_port=${DDP_PORT}
