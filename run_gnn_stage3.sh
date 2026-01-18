#!/bin/bash
#==============================================
# GNN Stage 3: QMIX Training with Graph Reconstruction
# Uses pretrained tokenizer and mask predictor
#==============================================
export CUDA_VISIBLE_DEVICES=0

TOKENIZER_PATH="results/models/sc2_5m_vs_6m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_origin__2026-01-12_16-53-01/pretrain_stage1_best"
MASK_PREDICTOR_PATH="results/models/sc2_5m_vs_6m-obs_aid=1-obs_act=1/algo=omagd_hungarian-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd_hungarian__2026-01-12_17-23-58/pretrain_stage2_best "

cd /home/marl2025/new_omagd

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
    local_results_path="results/gnn_stage3" \
    use_wandb=True
