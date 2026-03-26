<div align="center">

# InCoder-32B: Industrial Code Foundation Model

**The first 32B code LLM purpose-built for industrial code intelligence**

[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder)
[![GitHub](https://img.shields.io/badge/GitHub-CSJianYang-black)](https://github.com/CSJianYang/Industrial-Coder)

</div>

---

## Overview

**InCoder-32B** is the first 32B-parameter code foundation model unifying code intelligence across industrial engineering domains. While general code LLMs excel at standard programming tasks, they degrade significantly when faced with hardware semantics, specialized language constructs, and strict resource constraints. InCoder-32B is purpose-built to address these challenges across:

- **Chip Design** (Verilog / RTL)
- **GPU Kernel Optimization** (CUDA / Triton)
- **Embedded Systems** (ARM Cortex-M, STM32)
- **Compiler Optimization** (x86-64 assembly, LLVM)
- **3D Modeling** (CAD/CAM via CadQuery / OpenCascade)

Native long-context support up to **128K tokens**.

---

## Model Family

| Model | Format | HuggingFace |
|---|:---:|---|
| InCoder-32B-Base | BF16 | [🤗 IndustrialCoder-Base](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder-Base) |
| InCoder-32B | BF16 / FP16 | [🤗 InCoder](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder) |
| InCoder-32B-Thinking | BF16 | [🤗 IndustrialCoder-Thinking](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder-Thinking) |
| InCoder-32B-FP8 | FP8 | [🤗 InCoder-32B-FP8](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder-32B-FP8) |
| InCoder-32B-AWQ-INT4 | AWQ INT4 | [🤗 InCoder-32B-AWQ-INT4](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder-32B-AWQ-INT4) |
| InCoder-32B-GPTQ-INT4 | GPTQ INT4 | [🤗 InCoder-32B-GPTQ-INT4](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder-32B-GPTQ-INT4) |

---

## Performance

### General Code Benchmarks

| Model | Size | HumanEval | HumanEval+ | MBPP | MBPP+ | BCB Full | BCB Hard | FullStack |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-Coder-32B-Instruct | 32B | 93.3 | 86.6 | 90.2 | 77.8 | 48.0 | 24.3 | 57.4 |
| Kimi-K2-Instruct | 32B/1T | 94.5 | 89.6 | 91.8 | 74.1 | **49.8** | 30.4 | 63.5 |
| Kimi-K2-Thinking | 32B/1T | 98.2 | 92.7 | 91.8 | **82.3** | 46.8 | 28.4 | 58.8 |
| **InCoder-32B** | **32B** | **94.5** | **89.6** | **91.8** | **78.3** | **49.8** | **31.1** | **57.1** |

### Agentic & Tool-Use Benchmarks

| Model | Size | Terminal-Bench v1.0 | SWE-bench Verified | Mind2Web | BFCL V3 | τ²-bench Avg |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-235B-A22B-Thinking | 22/235B | 8.8 | 44.6 | 43.2 | 71.9 | 45.6 |
| Kimi-K2-Instruct | 32B/1T | **47.1** | 69.2 | 55.7 | — | — |
| **InCoder-32B** | **32B** | 35.0 | **74.8** | **55.8** | **61.0** | **80.6** |

### Industrial Code Benchmarks

| Domain | Benchmark | InCoder-32B | Claude-Sonnet-4.6 | Qwen3.5-397B-A17B |
|---|---|:---:|:---:|:---:|
| **Chip Design** | VeriScope Score | 80.7 | **87.7** | 73.1 |
| **Chip Design** | VeriRepair Fix (%) | 80.0 | **83.3** | 86.7 |
| **Chip Design** | RealBench Func@1 (Mod) | **62.7** | 37.2 | 28.3 |
| **Chip Design** | ArchXBench *t* | 51.0 | **58.2** | 53.5 |
| **GPU Optim.** | KernelBench L1/L2/L3 | **22.2/36.0/14.0** | 11.1/28.0/2.0 | 4.0/10.0/0.0 |
| **GPU Optim.** | TritonBench G-exe (%) | **100.0** | 98.1 | **100.0** |
| **3D Modeling** | CAD-Coder Compile (%) | **82.0** | 77.0 | 79.0 |
| **3D Modeling** | CAD-Coder IoU | **53.5** | 32.4 | 88.0* |
| **Code Optim.** | EmbedCGen Main | 35.2 | **81.0** | 34.0 |
| **Code Optim.** | SuperCoder Acc | **91.0** | 88.0 | 81.0 |

> InCoder-32B leads all open-weight baselines across industrial domains and surpasses the proprietary Claude-Sonnet-4.6 on CAD-Coder IoU and KernelBench L1/L2/L3.

---

## Training Pipeline: Code-Flow

InCoder-32B is trained through a three-stage **Code-Flow** pipeline:

```
Pre-training  ──►  Mid-training  ──►  Post-training
(Annealing)       (8K → 128K ctx)    (SFT + Execution Grounding)
```

### Stage 1 · Pre-training & Annealing
- Industrial code collected from public repositories, technical literature, and domain-specific web data
- License filtering, PII removal, and multi-level deduplication (exact hash, token-level near-duplicate, repo-level fork consolidation)
- Trained on **4,096 GPUs** with autoregressive LM + fill-in-the-middle (FIM)

### Stage 2 · Mid-training (Context Extension)
- **Sub-stage 1**: 8K → 32K for file-level tasks (e.g. RTL module completion)
- **Sub-stage 2**: 32K → 128K for long-context tasks (e.g. extended debugging sessions)
- Synthetic industrial QA: *scenario spec → seed code generation → QA pair synthesis with automated verification*
- Agent trajectories via Thought-Action-Observation cycles with feedback from hardware simulators, synthesis tools, C/C++ compilers, and formal verification engines

### Stage 3 · Post-training (Execution-Grounded SFT)
- **2.5M samples** from real industrial coding tasks grounded in execution across hardware design, GPU kernels, systems programming, and embedded firmware
- **Feedback-Driven Repair**: closed-loop repair trajectories from compiler errors, runtime logs, waveform differences, and profiling bottlenecks
- Yields both an **instruction-tuned variant** and a **thinking variant**

---

## Industrial Simulation Environments

| Domain | Toolchain | Correctness Criteria |
|---|---|---|
| **Chip Design** | Icarus Verilog, Verilator, Yosys | RTL simulation pass + synthesis feasibility |
| **GPU Optimization** | NVIDIA `nvcc` via PyTorch, `@triton.jit` | Numerical correctness + CUDA timing |
| **3D Modeling** | CadQuery / OpenCascade | Volumetric IoU vs. reference geometry |
| **Embedded Systems** | arm-none-eabi-gcc + Renode (STM32F407) | HAL register behavior + interrupt fidelity |
| **Compiler Optim.** | x86-64 GCC/LLVM test harness | Semantic equivalence + measurable speedup |

---

## Quickstart

### Install

```bash
pip install -U "transformers>=4.57.1" accelerate safetensors
```

### Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Multilingual-Multimodal-NLP/IndustrialCoder"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Optimize this CUDA kernel for better memory coalescing."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.85, top_k=20)

print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### vLLM (recommended for production)

```bash
pip install vllm

# BF16
vllm serve Multilingual-Multimodal-NLP/IndustrialCoder \
    --tensor-parallel-size 4 --max-model-len 32768

# FP8
vllm serve Multilingual-Multimodal-NLP/IndustrialCoder-32B-FP8 \
    --tensor-parallel-size 2 --max-model-len 32768
```

### AWQ / GPTQ (single GPU, ~20 GB VRAM)

```python
# AWQ INT4
model_name = "Multilingual-Multimodal-NLP/IndustrialCoder-32B-AWQ-INT4"

# GPTQ INT4
model_name = "Multilingual-Multimodal-NLP/IndustrialCoder-32B-GPTQ-INT4"
```

### Recommended Sampling Parameters

| Use case | temperature | top_p | top_k | max_new_tokens |
|---|:---:|:---:|:---:|:---:|
| General coding | 0.6 | 0.85 | 20 | 2048 |
| Industrial / precise | 0.2 | 0.95 | — | 4096 |
| Thinking variant | 0.6 | 0.85 | 20 | 8192 |

---

## Evaluation Benchmarks

**General Code (14 benchmarks):** EvalPlus, MBPP/MBPP+, BigCodeBench, FullStackBench, CRUXEval, LiveCodeBench, Mercury, Spider, BIRD, Terminal-Bench v1/v2, SWE-bench Verified, Mind2Web, BFCL V3, τ²-bench

**Industrial Code (9 benchmarks across 4 domains):**

| Domain | Benchmarks |
|---|---|
| Chip Design | VeriScope, VeriRepair, RealBench, ArchXBench |
| GPU Optimization | KernelBench, TritonBench |
| Code Optimization | EmbedCGen, SuperCoder |
| 3D Modeling | CAD-Coder |

---

## Disclaimer

The model may generate incorrect or unsafe code. Always review and test outputs in a sandboxed environment before production use. Industrial code (RTL, embedded firmware, GPU kernels) requires expert human review before deployment.

---

## Fine-tuning

We provide a lightweight SFT (Supervised Fine-Tuning) framework for InCoder-32B in the [`sft/`](./sft/) directory.

### Setup

```bash
cd sft/
pip install -r requirements.txt
```

### Data Preparation

Prepare a JSONL file where each line is a JSON object with a `"messages"` field (ChatML format):

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Tokenize it to `.npy` format:

```bash
bash scripts/binarize_data.sh /path/to/data.jsonl /path/to/output /path/to/IndustrialCoder-Base 16384
```

### Training

1. Edit `configs/sft_32b.yaml` — set `model_name_or_path`, `data_path`, and `output_dir`.
2. Launch training:

```bash
bash start.sh
# or directly:
bash scripts/run_sft.sh configs/sft_32b.yaml
```

Multi-node training is supported via `MASTER_ADDR`, `WORLD_SIZE`, `RANK` environment variables.

### Download Model

```bash
python download_model.py \
    --repo_id Multilingual-Multimodal-NLP/IndustrialCoder-Base \
    --local_dir /path/to/save/model
```