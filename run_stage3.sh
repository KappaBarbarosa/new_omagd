#!/bin/bash
#==============================================
# Stage 3: QMIX Training with Reconstructed Observations
# Uses pretrained tokenizer and mask predictor
#==============================================

# Paths to pretrained models (update these!)
export CUDA_VISIBLE_DEVICES=1
TOKENIZER_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-16_21-09-00/pretrain_stage1"
MASK_PREDICTOR_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_origin__2026-01-06_22-39-26/pretrain_stage2_best"

cd /home/marl2025/new_omagd

python src/main.py \
    --config=omagd \
    --env-config=sc2 \
    with \
    env_args.map_name=3m \
    use_cuda=True \
    pretrain_only=False \
    recontructer_stage=stage3 \
    pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    pretrained_mask_predictor_path="${MASK_PREDICTOR_PATH}" \
    t_max=500000 \
    test_interval=10000 \
    save_model=True \
    save_model_interval=100000 \
    log_interval=10000 \
    local_results_path="results/stage3"