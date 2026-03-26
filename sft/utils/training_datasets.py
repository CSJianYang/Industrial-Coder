import json
import torch
import os
from torch.utils.data import Dataset
from typing import Dict
import transformers
import logging
import numpy as np
from utils import utils

logging.basicConfig(level=logging.DEBUG)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if data_path.endswith(".npy"):
            self.input_ids = np.load(data_path, allow_pickle=True)
        else:
            self.input_ids = utils.read_jsonl_file(data_path)
        original_data_num = len(self.input_ids)
        logging.info("Completely Loading tokenized sentences...")

        def truncate(sentence):
            return torch.tensor(sentence[:args.model_max_length] + [tokenizer.eos_token_id] if len(sentence) > args.model_max_length else sentence, dtype=torch.long)

        if args.truncate_source:
            self.labels = [truncate(example["label"]) for example in self.input_ids]
            self.input_ids = [truncate(example["input_ids"]) for example in self.input_ids]
        else:
            self.labels = [torch.tensor(example["label"], dtype=torch.long) for example in self.input_ids if len(example["input_ids"]) <= args.model_max_length]
            self.input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in self.input_ids if len(example["input_ids"]) <= args.model_max_length]
        print(f"Samples: {original_data_num} -> {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class MMAPSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with memory-mapped files."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
        super(Dataset, self).__init__()
        logging.warning("Loading data...")
        input_ids_path = data_path
        labels_path = data_path.replace(".input_ids.mmap", ".labels.mmap")
        lengths_path = data_path.replace(".input_ids.mmap", ".lengths.mmap")

        with open(input_ids_path + ".shape.json", 'r') as f:
            input_ids_shape_info = json.load(f)
        with open(labels_path + ".shape.json", 'r') as f:
            labels_shape_info = json.load(f)
        with open(lengths_path + ".shape.json", 'r') as f:
            lengths_shape_info = json.load(f)

        self.model_max_length = args.model_max_length
        self.truncate_source = args.truncate_source

        self.input_ids = np.memmap(
            input_ids_path, dtype=np.int32, mode='r',
            shape=(input_ids_shape_info['n_samples'], input_ids_shape_info['max_len'])
        )
        self.labels = np.memmap(
            labels_path, dtype=np.int32, mode='r',
            shape=(labels_shape_info['n_samples'], labels_shape_info['max_len'])
        )
        self.lengths = np.memmap(
            lengths_path, dtype=np.int32, mode='r',
            shape=(lengths_shape_info['n_samples'], lengths_shape_info['max_len'])
        )
        logging.info(f"Loaded {len(self.input_ids)} samples using mmap")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        length = int(self.lengths[i])
        input_ids = torch.tensor(self.input_ids[i][:length], dtype=torch.long)
        labels = torch.tensor(self.labels[i][:length], dtype=torch.long)
        if self.truncate_source:
            input_ids = input_ids[:self.model_max_length]
            labels = labels[:self.model_max_length]
        return dict(input_ids=input_ids, labels=labels)
