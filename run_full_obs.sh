#!/bin/bash
#==============================================
# QMIX Training with Full Observations (Upper Bound)
# Uses full_obs (no range limit) for decision making
#==============================================
export CUDA_VISIBLE_DEVICES=0

cd /home/marl2025/new_omagd

python src/main.py \
    --config=qmix \
    --env-config=sc2 \
    with \
    env_args.map_name=3m \
    use_full_obs=True \
    use_cuda=True \
    cpu_inference=False \
    t_max=500000 \
    test_interval=10000 \
    test_nepisode=32 \
    save_model=True \
    save_model_interval=100000 \
    log_interval=10000 \
    local_results_path="results/fullobs_qmix"

