#!/bin/bash
# Usage: bash scripts/run_sft.sh configs/sft_32b.yaml

set -euo pipefail

CONFIG_PATH=${1:?"Usage: bash scripts/run_sft.sh <config.yaml>"}

export NCCL_DEBUG=WARN

# Clear stale HF cached remote model code on every node
rm -rf ~/.cache/huggingface/modules/transformers_modules/ 2>/dev/null || true
rm -f /tmp/.hf_tokenizer_cached 2>/dev/null || true

GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_PORT=${MASTER_PORT:-6105}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== IndustrialCoder SFT Training ==="
echo "Config:         ${CONFIG_PATH}"
echo "GPUs per node:  ${GPUS_PER_NODE}"
echo "Nodes:          ${NNODES}"
echo "World size:     ${WORLD_SIZE}"
echo "Working dir:    ${SCRIPT_DIR}"
echo "================================"

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    train.py --config ${CONFIG_PATH}
