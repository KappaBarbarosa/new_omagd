#!/bin/bash
# =============================================================================
# Multi-Map Experiment Runner
# =============================================================================
# 
# Runs the OMAGD pipeline on multiple maps sequentially.
#
# Usage:
#   ./run_multi_map.sh <maps> <t_max> [seeds...] [options]
#
# Arguments:
#   maps    - Space-separated list of maps in quotes (required)
#   t_max   - Training steps for Stage 3 (required, e.g., 2000000)
#   seeds   - Optional seed numbers (default: 1)
#
# Examples:
#   # Run full pipeline on multiple maps with 2M steps for stage3
#   ./run_multi_map.sh "3m 8m_vs_9m 5m_vs_6m" 2000000
#
#   # Run with specific seeds
#   ./run_multi_map.sh "3m 8m_vs_9m" 5000000 1 2 3
#
#   # Run only pretrain (no stage3 training needed)
#   ./run_multi_map.sh "3m 8m_vs_9m" 0 --stages stage1,stage2
#
# =============================================================================

set -e

# Check required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <maps> <t_max> [seeds...] [options]"
    echo ""
    echo "Arguments:"
    echo "  maps    - Space-separated list of maps in quotes"
    echo "  t_max   - Training steps for Stage 3 (use 0 for pretrain only)"
    echo ""
    echo "Examples:"
    echo "  $0 \"3m 8m_vs_9m\" 2000000"
    echo "  $0 \"3m 8m_vs_9m\" 5000000 1 2 3"
    echo "  $0 \"3m 8m_vs_9m\" 0 --stages stage1,stage2"
    exit 1
fi

# Parse required arguments
MAPS="$1"
T_MAX="$2"
shift 2

# Collect seeds and extra args
SEEDS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        [0-9]*)
            SEEDS+=("$1")
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

# Default seed if none provided
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(1)
fi

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "$0")"

# Convert maps string to array
read -ra MAP_ARRAY <<< "$MAPS"

echo "=============================================="
echo "  Multi-Map Experiment Runner"
echo "=============================================="
echo "  Maps:     ${MAP_ARRAY[*]}"
echo "  T_max:    ${T_MAX}"
echo "  Seeds:    ${SEEDS[*]}"
echo "  GPU:      ${GPU_ID}"
echo "  Extra:    ${EXTRA_ARGS[*]}"
echo "=============================================="
echo ""

# Track results
TOTAL_MAPS=${#MAP_ARRAY[@]}
COMPLETED=0
FAILED=0
FAILED_MAPS=()

START_TIME=$(date +%s)

for MAP in "${MAP_ARRAY[@]}"; do
    echo ""
    echo "======================================================"
    echo "  Starting: ${MAP} ($(($COMPLETED + 1))/${TOTAL_MAPS})"
    echo "======================================================"
    echo ""
    
    # Build seed arguments
    SEED_ARGS=""
    if [ ${#SEEDS[@]} -gt 0 ]; then
        SEED_ARGS="--seed ${SEEDS[*]}"
    fi
    
    if python src/run_pipeline.py \
        --map "${MAP}" \
        --gpu "${GPU_ID}" \
        --t-max "${T_MAX}" \
        ${SEED_ARGS} \
        ${EXTRA_ARGS[*]}; then
        
        COMPLETED=$((COMPLETED + 1))
        echo ""
        echo "[SUCCESS] Completed ${MAP}"
    else
        FAILED=$((FAILED + 1))
        FAILED_MAPS+=("$MAP")
        echo ""
        echo "[FAILED] ${MAP} failed, continuing to next map..."
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "  Multi-Map Experiment Summary"
echo "=============================================="
echo "  Total maps:  ${TOTAL_MAPS}"
echo "  Completed:   ${COMPLETED}"
echo "  Failed:      ${FAILED}"
echo "  Duration:    $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s"

if [ ${#FAILED_MAPS[@]} -gt 0 ]; then
    echo "  Failed maps: ${FAILED_MAPS[*]}"
fi

echo "=============================================="

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
