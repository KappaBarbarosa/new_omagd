#!/bin/bash
# =============================================================================
# Stage 2 Training Script - Train Mask Predictor (with frozen tokenizer)
# =============================================================================
# Usage: ./run_stage2.sh [map_name] [tokenizer_path]
# Example: ./run_stage2.sh 3m results/models/.../pretrain_stage1_best
#
# The tokenizer_path should point to the directory containing tokenizer.th
# from Stage 1 training.
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
export CUDA_VISIBLE_DEVICES=1
MAP_NAME=${1:-"3m"}
TOKENIZER_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-16_21-09-00/pretrain_stage1"
# MASK_PREDICTOR_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_origin__2026-01-06_22-04-26/pretrain_stage2_best"
python src/main.py --config=omagd --env-config=sc2 \
    with env_args.map_name=${MAP_NAME} \
    seed=1 \
    recontructer_stage=stage2 \
    use_graph_reconstruction=True \
    pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    # pretrained_mask_predictor_path="${MASK_PREDICTOR_PATH}" \
    use_wandb=True