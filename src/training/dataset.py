from __future__ import annotations

from typing import List, Union

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

from src.shared.logging import setup_logging

logger = setup_logging("Dataset")


def format_system_prompt(system_prompt: str, instruction: str, output: str) -> str:
    """Format example into Llama 3 chat template."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
    )


def load_and_process_dataset(
    dataset_paths: Union[str, List[str]],
    system_prompt: str,
    tokenizer,
    max_seq_length: int,
):
    """Load JSONL, format prompts, tokenize and return (DatasetDict, collator)."""
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    logger.info("Loading datasets: %s", dataset_paths)
    # Load each file separately to tolerate schema differences (e.g. optional "input" column),
    # then concatenate. This avoids CastError when files have different column sets.
    from datasets import concatenate_datasets

    parts = []
    for path in dataset_paths:
        ds = load_dataset("json", data_files=[path], split="train")
        # Drop any column that isn't needed for training
        keep = {"instruction", "output"}
        to_drop = [c for c in ds.column_names if c not in keep]
        if to_drop:
            ds = ds.remove_columns(to_drop)
        parts.append(ds)
    dataset = concatenate_datasets(parts)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(
        "Dataset split: train=%s val=%s",
        len(dataset_split["train"]),
        len(dataset_split["test"]),
    )

    def formatting_prompts_func(example):
        text = format_system_prompt(system_prompt, example["instruction"], example["output"])
        return {"text": text}

    dataset_dict = dataset_split.map(formatting_prompts_func)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding=False)

    all_cols = dataset_dict["train"].column_names
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=all_cols)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return tokenized_datasets, collator
