#!/bin/bash
# Activate your virtual environment (if applicable)
# source .venv_sft/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"
bash scripts/run_sft.sh configs/sft_32b.yaml
