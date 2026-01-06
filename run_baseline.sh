#!/bin/bash
#==============================================
# Baseline QMIX Training (No Graph Reconstruction)
# For comparison with Stage 3
#==============================================

cd /home/marl2025/new_omagd

python src/main.py \
    --config=qmix \
    --env-config=sc2 \
    with \
    env_args.map_name=3m \
    use_cuda=True \
    t_max=500000 \
    test_interval=10000 \
    test_nepisode=32 \
    save_model=True \
    save_model_interval=100000 \
    log_interval=10000 \
    local_results_path="results/baseline_qmix"
