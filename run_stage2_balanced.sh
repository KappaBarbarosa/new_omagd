#!/bin/bash
# Train Stage 2 with balanced buffer
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
export CUDA_VISIBLE_DEVICES=0

BUFFER_PATH="/home/marl2025/new_omagd/results/buffers/sc2_3m-obs_aid=1-obs_act=1/balanced_buffer_1.pt"
TOKENIZER_PATH="/home/marl2025/new_omagd/results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-16_21-09-00/pretrain_stage1"

python src/main.py \
    --config=omagd \
    --env-config=sc2 \
    with \
    env_args.map_name=3m \
    recontructer_stage=stage2 \
    balanced_buffer_path=${BUFFER_PATH} \
    pretrained_tokenizer_path=${TOKENIZER_PATH} \
    graph_pretrain_epochs=10 \
    graph_pretrain_batch_size=128 \
    seed=1 \
    graph_pretrain_log_interval=1 \
    graph_pretrain_eval_interval=1 \
    save_model=True \
    use_wandb=True
