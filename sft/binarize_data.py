import jsonlines
import os
import numpy as np
import transformers
import tqdm
import sys
from typing import Dict, List, Optional, Any, Union
import argparse
import itertools
import json
from utils import utils

IGNORE_INDEX = -100


def setup_tokenizer(tokenizer):
    """Set special tokens globally to avoid adding them multiple times."""
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|repo_name|>",
            "<|file_sep|>", "<|im_start|>", "<|im_end|>"
        ]
    })
    return tokenizer


# ============================================================================
# Tool XML Formatting (matches no_think_chat_template_xml.jinja)
# ============================================================================

TOOL_CALL_INSTRUCTION = (
    "If you choose to call a function ONLY reply in the following format:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def parse_tools(tools: Optional[Union[str, List, Dict]]) -> Optional[List[Dict[str, Any]]]:
    """Parse tools from various formats into a list of tool dicts."""
    if tools is None or tools == "" or tools == []:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(tools, dict):
        tools = [tools]
    if isinstance(tools, list) and len(tools) > 0:
        return tools
    return None


def format_tools_xml(tools: List[Dict[str, Any]]) -> str:
    """Format tools as XML matching the jinja template exactly."""
    parts = ["# Tools\n\nYou have access to the following functions:\n\n<tools>"]

    for tool in tools:
        if tool.get("type") == "function" and tool.get("function"):
            func = tool["function"]
        else:
            func = tool

        parts.append(f"\n<function>\n<name>{func.get('name', '')}</name>")

        if func.get("description"):
            parts.append(f"\n<description>{func['description']}</description>")

        parts.append("\n<parameters>")

        params = func.get("parameters", {})
        if isinstance(params, dict) and params.get("properties"):
            for param_name, param_fields in params["properties"].items():
                parts.append("\n<parameter>")
                parts.append(f"\n<name>{param_name}</name>")
                if param_fields.get("type"):
                    parts.append(f"\n<type>{param_fields['type']}</type>")
                if param_fields.get("description"):
                    parts.append(f"\n<description>{param_fields['description']}</description>")
                parts.append("\n</parameter>")

        parts.append("\n</parameters>\n</function>")

    parts.append("\n</tools>")
    parts.append(f"\n\n{TOOL_CALL_INSTRUCTION}")

    return "".join(parts)


# ============================================================================
# Text Assembly (matches jinja template exactly, then tokenize once)
# ============================================================================

DEFAULT_SYSTEM = "You are IndustrialCoder, a helpful assistant developed by Beihang University."


def build_full_text_and_masks(
    sources: List[Dict],
    tools: Optional[Union[str, List, Dict]] = None,
    system_message: str = DEFAULT_SYSTEM,
) -> tuple:
    """Build the full conversation text and identify assistant content spans.

    Returns:
        full_text: The complete conversation string (matching jinja template output).
        assistant_spans: List of (start_char, end_char) for assistant content
                         that should compute loss.
    """
    parsed_tools = parse_tools(tools)

    # --- System message ---
    if sources and sources[0].get("role") == "system":
        sys_content = sources[0].get("content", "") or system_message
        start_idx = 1
    else:
        sys_content = system_message
        start_idx = 0

    if parsed_tools:
        sys_content = sys_content + "\n\n" + format_tools_xml(parsed_tools)

    full_text = f"<|im_start|>system\n{sys_content}<|im_end|>\n"

    # Track char positions where assistant CONTENT starts/ends (for loss masking)
    assistant_spans = []

    # --- Conversation turns ---
    i = start_idx
    while i < len(sources):
        msg = sources[i]
        role = msg["role"]

        if role == "tool":
            # Group consecutive tool messages into one user turn
            tool_parts = []
            while i < len(sources) and sources[i]["role"] == "tool":
                tool_parts.append(f"<tool_response>\n{sources[i]['content']}\n</tool_response>")
                i += 1
            content = "\n".join(tool_parts)
            full_text += f"<|im_start|>user\n{content}<|im_end|>\n"

        elif role == "user" or role == "system":
            full_text += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
            i += 1

        elif role == "assistant":
            prefix = f"<|im_start|>assistant\n"
            content = msg.get("content", "")
            suffix = "<|im_end|>\n"
            # Record the span of assistant content (excluding prefix and suffix)
            content_start = len(full_text) + len(prefix)
            content_end = content_start + len(content)
            assistant_spans.append((content_start, content_end))
            full_text += prefix + content + suffix
            i += 1

        else:
            # Unknown role: treat as user
            full_text += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
            i += 1

    return full_text, assistant_spans


def chatml_format_preprocess(
    sources: List[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = DEFAULT_SYSTEM,
    only_last_turn_loss: bool = False,
    return_test_input_ids: bool = False,
    tools: Optional[Union[str, List, Dict]] = None,
) -> Optional[Dict]:
    """Preprocess conversation into tokenized input_ids and labels.

    Strategy: Build full text first (matching jinja template), tokenize once,
    then compute loss mask by finding assistant turn boundaries via special token IDs.
    This guarantees exact consistency with the jinja template.
    """
    full_text, assistant_spans = build_full_text_and_masks(
        sources, tools=tools, system_message=system_message
    )

    # Tokenize the full text at once (consistent with jinja template)
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(input_ids) > max_len:
        return None

    # Find assistant turn boundaries using special token IDs
    im_start_id = tokenizer("<|im_start|>", add_special_tokens=False)["input_ids"][0]
    im_end_id = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
    # Tokenize "assistant\n" to get the prefix pattern after <|im_start|>
    assistant_prefix_ids = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
    prefix_len = len(assistant_prefix_ids)

    # Build label: IGNORE_INDEX everywhere
    label = [IGNORE_INDEX] * len(input_ids)

    # Find all assistant turns: <|im_start|> + "assistant\n" + content + <|im_end|>
    assistant_turn_ranges = []  # list of (content_start_idx, im_end_idx)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_start_id:
            # Check if followed by "assistant\n" tokens
            candidate = input_ids[i + 1: i + 1 + prefix_len]
            if candidate == assistant_prefix_ids:
                content_start = i + 1 + prefix_len
                # Find the matching <|im_end|>
                im_end_idx = None
                for j in range(content_start, len(input_ids)):
                    if input_ids[j] == im_end_id:
                        im_end_idx = j
                        break
                if im_end_idx is not None:
                    assistant_turn_ranges.append((content_start, im_end_idx))
                    i = im_end_idx + 1
                    continue
        i += 1

    # Apply loss mask
    turns_to_use = [assistant_turn_ranges[-1]] if (only_last_turn_loss and assistant_turn_ranges) else assistant_turn_ranges

    for (content_start, im_end_idx) in turns_to_use:
        # Unmask assistant content tokens + <|im_end|>
        for idx in range(content_start, im_end_idx + 1):
            label[idx] = input_ids[idx]

    if return_test_input_ids:
        return dict(test_input_ids=input_ids, input_ids=input_ids, label=label)
    else:
        return dict(input_ids=input_ids, label=label, length=[len(input_ids)])


# ============================================================================
# Multiprocessing Worker
# ============================================================================

def read_file_from_position_with_chatml_format_processor(args):
    filename, start_position, end_position, worker_id, args = args
    tokenizer = args["tokenizer"]
    max_len = args["max_len"]
    objs = []
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        current_position = utils.find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return objs
        for cnt in tqdm.tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()
            if not line:
                break
            try:
                raw = json.loads(line)
            except Exception:
                print("Invalid json!")
                continue
            try:
                result = chatml_format_preprocess(
                    raw["messages"], tokenizer, max_len=max_len,
                    only_last_turn_loss=raw.get("only_last_turn_loss", False),
                    tools=raw.get("tools", None),
                )
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
            if result is not None:
                objs.append(result)
            if f.tell() >= end_position:
                break
    print(f"worker_id {worker_id} completed")
    return objs


# ============================================================================
# Save Functions
# ============================================================================

def convert_to_uint32(x):
    return np.array(x, dtype=np.uint32)

def convert_to_int32(x):
    return np.array(x, dtype=np.int32)

def save_mmap(objs, key, output_path, padding_value):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = []
    max_length = 0
    for obj in tqdm.tqdm(objs):
        vec = obj[key]
        data.append(vec)
        max_length = max(max_length, len(vec))
    n_samples = len(data)
    utils.save_json(data={
        "n_samples": n_samples,
        "max_len": max_length,
    }, output_path=f"{output_path}.shape.json")
    data_shape = (n_samples, max_length)
    data_mmap = np.memmap(output_path, dtype=np.int32, mode='w+', shape=data_shape)
    for i, vec in enumerate(data):
        padded_vec = vec + [padding_value] * (max_length - len(vec))
        data_mmap[i] = padded_vec
    data_mmap.flush()


def tokenize_file(workers=64, chunk_size=10000, input_path="./raw/sft.jsonl",
                  output_path="./processed/sft.jsonl", tokenizer=None,
                  max_len=32768, save_format=".npy"):
    output_objs = utils.multi_tasks_from_file(
        input_path, workers=workers,
        task=read_file_from_position_with_chatml_format_processor,
        chunk_size=chunk_size, args={"tokenizer": tokenizer, "max_len": max_len}
    )
    print(f"Total tokenized samples: {len(output_objs)}")
    if save_format == ".jsonl":
        utils.write_jsonl_file(output_objs, output_path)
        print(f"Successfully saved to {output_path}")
    elif save_format == ".npy":
        for obj in output_objs:
            obj["input_ids"] = convert_to_uint32(obj["input_ids"])
            obj["label"] = convert_to_int32(obj["label"])
            if "test_input_ids" in obj:
                obj["test_input_ids"] = convert_to_uint32(obj["test_input_ids"])
        np.save(f"{output_path}.npy", output_objs, allow_pickle=True)
        print(f"Successfully saved to {output_path}.npy")
    elif save_format == ".mmap":
        save_mmap(output_objs, key="input_ids", output_path=f"{output_path}.input_ids.mmap", padding_value=tokenizer.pad_token_id)
        save_mmap(output_objs, key="label", output_path=f"{output_path}.labels.mmap", padding_value=IGNORE_INDEX)
        save_mmap(output_objs, key="length", output_path=f"{output_path}.lengths.mmap", padding_value=IGNORE_INDEX)
        print(f"Successfully saved mmap files to {output_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Binarize JSONL data for SFT training')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file (without extension)')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--chunk_size', type=float, default=0.1 * 2 ** 30, help='Chunk size in bytes')
    parser.add_argument('--max_len', type=int, default=16384, help='Maximum sequence length')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--save_format', type=str, default=".npy", choices=[".npy", ".jsonl", ".mmap"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        add_eos_token=False,
        add_bos_token=False,
        pad_token='<|endoftext|>',
        eos_token='<|im_end|>',
        cache_dir=None,
        model_max_length=args.max_len * 5,
        truncation=True,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer = setup_tokenizer(tokenizer)
    tokenize_file(
        workers=args.workers, chunk_size=args.chunk_size,
        input_path=args.input_path, output_path=args.output_path,
        tokenizer=tokenizer, max_len=args.max_len, save_format=args.save_format
    )
