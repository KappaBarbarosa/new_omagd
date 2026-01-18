#!/bin/bash
#==============================================
# GNN Full Obs QMIX Training (Upper Bound)
# Uses full_obs (no range limit) with graph structure
#==============================================
export CUDA_VISIBLE_DEVICES=1

cd /home/marl2025/new_omagd

python src/main.py \
    --config=qmix \
    --env-config=sc2 \
    with \
    env_args.map_name=5m_vs_6m \
    mac=gnn_graph_mac \
    agent=gnn_rnn \
    gnn_layer_num=2 \
    use_full_obs=True \
    use_cuda=True \
    cpu_inference=False \
    t_max=5000000 \
    test_interval=10000 \
    test_nepisode=32 \
    save_model=True \
    save_model_interval=100000 \
    log_interval=10000 \
    local_results_path="results/gnn_fullobs"
