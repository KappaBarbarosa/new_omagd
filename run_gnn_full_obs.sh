#!/bin/bash
# =============================================================================
# GNN Full Obs QMIX Training (Upper Bound)
# Uses full_obs (no range limit) with graph structure
# =============================================================================
#
# 在這裡定義要跑的實驗：(map -> t_max)
#
declare -A EXPERIMENTS=(
    # ["3m"]=500000
    # ["8m"]=500000
    # ["3m_vs_3z"]=2000000
    # ["3m_vs_4z"]=2000000
    # ["3m_vs_5z"]=2000000
    # ["5m_vs_6m"]=5000000
    ["8m_vs_9m"]=5000000
    ["10m_vs_11m"]=5000000
    ["6h_vs_8z"]=5000000
    ["corridor"]=5000000
    ["MMM2"]=5000000
    ["3s5z_vs_3s6z"]=5000000
    # ["27m_vs_30m"]=5000000
    # ["2c_vs_64zg"]=5000000
    # ["bane_vs_bane"]=5000000
)

# 設定 seeds
SEEDS=(1)

# =============================================================================
# 以下不需要修改
# =============================================================================

set -e

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "$0")"

echo "=============================================="
echo "  GNN Full Obs QMIX (Upper Bound)"
echo "=============================================="
echo "  GPU:      ${GPU_ID}"
echo "  Seeds:    ${SEEDS[*]}"
echo ""
echo "  Experiments:"
for MAP in "${!EXPERIMENTS[@]}"; do
    echo "    - ${MAP}: ${EXPERIMENTS[$MAP]} steps"
done
echo "=============================================="
echo ""

TOTAL=${#EXPERIMENTS[@]}
COMPLETED=0
FAILED=0
FAILED_MAPS=()
CURRENT=0

START_TIME=$(date +%s)

for MAP in "${!EXPERIMENTS[@]}"; do
    T_MAX="${EXPERIMENTS[$MAP]}"
    
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "======================================================"
        echo "  [${CURRENT}] Map: ${MAP} | Seed: ${SEED} | T_max: ${T_MAX}"
        echo "======================================================"
        echo ""
        
        if python src/main.py \
            --config=qmix \
            --env-config=sc2 \
            with \
            env_args.map_name="${MAP}" \
            mac=gnn_graph_mac \
            agent=gnn_rnn \
            gnn_layer_num=2 \
            use_full_obs=True \
            use_cuda=True \
            cpu_inference=False \
            t_max="${T_MAX}" \
            seed="${SEED}" \
            test_interval=10000 \
            test_nepisode=32 \
            save_model=True \
            save_model_interval=100000 \
            log_interval=10000 \
            name="gnn_fullobs_${MAP}" \
            local_results_path="results/gnn_fullobs/${MAP}"; then
            
            COMPLETED=$((COMPLETED + 1))
            echo ""
            echo "[SUCCESS] Completed ${MAP} seed=${SEED}"
        else
            FAILED=$((FAILED + 1))
            FAILED_MAPS+=("${MAP}_s${SEED}")
            echo ""
            echo "[FAILED] ${MAP} seed=${SEED} failed, continuing..."
        fi
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "  Experiment Summary"
echo "=============================================="
echo "  Completed: ${COMPLETED}"
echo "  Failed:    ${FAILED}"
echo "  Duration:  $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s"

if [ ${#FAILED_MAPS[@]} -gt 0 ]; then
    echo "  Failed:    ${FAILED_MAPS[*]}"
fi

echo "=============================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
