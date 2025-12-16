#!/bin/bash
# Run Stage 2 Evaluation Only Mode
# Generates detailed per-episode report without training

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default tokenizer path (can be overridden)
TOKENIZER_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-15_20-12-37/pretrain_stage1_best"

# Default mask predictor path (can be overridden)
MASK_PREDICTOR_PATH="results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-15_22-49-01/pretrain_stage2_best"

echo "=============================================="
echo " Stage 2: Evaluation Only Mode"
echo " Generating detailed episode report"
echo "=============================================="


python src/main.py --config=omagd \
    --env-config=sc2 \
    with env_args.map_name="3m" \
    graph_pretrain_eval_episodes=40 \
    recontructer_stage="stage2" \
    evaluation_only=True \
    pretrained_tokenizer_path="${TOKENIZER_PATH}" \
    pretrained_mask_predictor_path="${MASK_PREDICTOR_PATH}" \
    save_model=False \
    use_wandb=False
