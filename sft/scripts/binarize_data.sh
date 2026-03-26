#!/bin/bash
# Binarize raw JSONL data to tokenized .npy format
# Usage: bash scripts/binarize_data.sh <input.jsonl> <output_path> [tokenizer_path] [max_len]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

INPUT_PATH=${1:?"Usage: bash scripts/binarize_data.sh <input.jsonl> <output_path> [tokenizer_path] [max_len]"}
OUTPUT_PATH=${2:?"Usage: bash scripts/binarize_data.sh <input.jsonl> <output_path> [tokenizer_path] [max_len]"}
TOKENIZER_PATH=${3:-"/path/to/tokenizer"}
MAX_LEN=${4:-16384}
WORKERS=64

echo "=== Binarize Data ==="
echo "Input:     ${INPUT_PATH}"
echo "Output:    ${OUTPUT_PATH}.npy"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Max len:   ${MAX_LEN}"
echo "Workers:   ${WORKERS}"
echo "====================="

python binarize_data.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --max_len ${MAX_LEN} \
    --workers ${WORKERS} \
    --chunk_size 200000000 \
    --save_format .npy
