# EmbedCGen evaluation

This directory contains the EC-Coder Core600 evaluation runner for embedded C generation on the STM32F407. The runner requests a C program from a model, compiles it with ARM GCC, and checks checkpoint and register readback evidence in Renode.

## Setup

Use Python 3.10 or later on Linux or WSL. From the Industrial-Coder repository root, enter the benchmark directory and install its Python dependencies.

```sh
cd evaluation/EmbedCGen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place or symlink the toolchain and simulator installations so that the runner can resolve these paths.

| Component | Path from `evaluation/EmbedCGen/` |
| --- | --- |
| Arm GNU Toolchain 14.3.Rel1 for Linux x86-64 and `arm-none-eabi` | `tools/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc` |
| Renode 1.16.0 | `tools/renode/renode` |
| STM32F4 platform supplied by Renode | `tools/renode/platforms/cpus/stm32f4.repl` |

Keep the Renode installation together so its platform descriptions can resolve their includes. The runner defaults to raw register readback without applying an SVD file. If you enable `CORE600_RENODE_APPLY_SVD=1`, supply `data/svd/STMicro/STM32F40x.svd` separately.

## Run Core600

Browse the 600 tasks in `data/test.jsonl`. [Browse the test split](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/EC-Coder-Core600/blob/main/data/test.jsonl)

Download `evaluation/cases.json` for the evaluator and set `CASES` to its local path. The test set has case IDs `790001` through `790600`. The evaluator file SHA-256 is `a804cd065c9e2f43c2d463103670448d39498687cbbc41631f67a8f8c3c247e0`. [Download the evaluator input](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/EC-Coder-Core600/resolve/main/evaluation/cases.json)

The runner retains an older default data path from the source snapshot. Always pass `--cases` explicitly when evaluating the submitted set.

```bash
CASES=/path/to/cases.json
MODEL=your-model-id
python -m scripts.check_core600_prompt_oracle_parity "$CASES"
read -rsp 'Model API key ' SF_API_KEY
printf '\n'
export SF_API_KEY
python -m scripts.core600_bench_runner \
  --model "$MODEL" --cases "$CASES" --semantic-oracle \
  --start 0 --end 1 --output output_benchmark/smoke
```

The contract check runs locally without a model request. The runner defaults to the SiliconFlow chat-completions endpoint. For a compatible endpoint, pass its full chat-completions URL with `--api-url`. After inspecting the one-task run, remove `--start` and `--end` and choose a new output directory for all 600 tasks. Check `results.jsonl`, `summary.json`, `compile_logs/`, and `renode_logs/` for the recorded outcomes and available evidence.

## Offline checks and provenance

```sh
python -m scripts.core600_bench_runner --help
python -m pip install pytest
python -m pytest -q
```

The SHA-256 of the original EC-Coder Core600 runner is `0010ecf2dd77a9669ed940b6d7366f92a82153a49c137c7164ba088bf4e51d05`. The tests exercise marker and simulator-log handling with local fixtures. Historical model results and full server logs are separate from this repository.
