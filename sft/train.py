import time
import logging
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import Trainer

from utils import utils
from utils import training_datasets

IGNORE_INDEX = -100
logging.basicConfig(level=logging.DEBUG)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use Flash Attention."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    truncate_source: bool = field(default=False)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if args.data_path.endswith(".npy") or args.data_path.endswith(".jsonl"):
        train_dataset = training_datasets.SupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, args=args)
    elif args.data_path.endswith(".mmap"):
        train_dataset = training_datasets.MMAPSupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, args=args)
    else:
        raise ValueError(f"Unsupported data format: {args.data_path}")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def is_master():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class SFTTrainer(Trainer):
    """Custom SFT Trainer aligned with MS-Swift's approach.

    Key difference from default Trainer: computes loss manually with
    logits upcasted to fp32 to prevent NaN in cross-entropy backward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prevent Trainer from scaling loss by num_items_in_batch
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop labels so the model does NOT compute loss internally
        labels = inputs.pop("labels")

        # Forward pass (model returns logits only, no loss)
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        # Compute loss in fp32 (MS-Swift approach: logits.float())
        logits_fp32 = logits.float()
        shift_logits = logits_fp32[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        return (loss, outputs) if return_outputs else loss


class SaveModelCallback(transformers.TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if is_master():
            print(f"Model saved at: {args.output_dir}/checkpoint-{state.global_step}/")
        return control


class LoggingCallback(transformers.TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_time = None
        self.last_step = 0
        self.total_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            current_time = time.time()
            world_size = args.world_size
            current_step = state.global_step
            if self.start_time is None:
                self.start_time = current_time
                self.last_time = current_time
            if args.include_num_input_tokens_seen:
                tokens_processed = (state.num_input_tokens_seen - self.total_tokens) * world_size
                self.total_tokens = state.num_input_tokens_seen
            else:
                batch_size = args.per_device_train_batch_size
                max_seq_length = args.model_max_length
                steps_elapsed = current_step - self.last_step
                tokens_processed = batch_size * max_seq_length * steps_elapsed * world_size
                self.last_step = current_step
            time_elapsed = current_time - self.last_time
            tokens_per_second = tokens_processed / time_elapsed if time_elapsed > 0 else 0
            self.last_time = current_time
            log_message = {
                "loss": logs.get("loss", None),
                "learning_rate": logs.get("learning_rate", None),
                "epoch": logs.get("epoch", None),
                "step": current_step,
                "grad_norm": logs.get("grad_norm", None),
                "world_size": world_size,
                "tokens_per_second": int(tokens_per_second)
            }
            if is_master():
                print(log_message)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, latest_checkpoint)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # Support: python train.py --config path/to/config.yaml
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[2])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(**{**model_args.__dict__, **data_args.__dict__, **training_args.__dict__})

    # Warm up HF cache on local_rank 0 first
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cache_flag = "/tmp/.hf_tokenizer_cached"
    if local_rank == 0:
        if os.path.exists(cache_flag):
            os.remove(cache_flag)
        transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True,
        )
        open(cache_flag, "w").close()
    else:
        while not os.path.exists(cache_flag):
            time.sleep(0.5)

    # Load model 
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else None,
        trust_remote_code=True,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        pad_token='<|endoftext|>',
        eos_token='<|im_end|>',
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})

    data_module = make_supervised_data_module(tokenizer=tokenizer, args=args)

    if local_rank == 0:
        print(f"[INFO] Samples: {len(data_module['train_dataset'])}")
        print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")
        print(f"[INFO] Gradient checkpointing: {training_args.gradient_checkpointing}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[LoggingCallback, SaveModelCallback],
    )

    # Resume from checkpoint if available
    resume_from_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if resume_from_checkpoint and is_master():
        print(f"[INFO] Resuming from: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
