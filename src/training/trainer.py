from __future__ import annotations

import os
from typing import Any

import bitsandbytes as bnb
import torch
from tqdm import tqdm

from src.shared.checkpoints import find_latest_checkpoint
from src.shared.logging import setup_logging
from src.training.dataset import load_and_process_dataset
from src.training.model import load_base_model, load_tokenizer, setup_lora_model

logger = setup_logging()


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_dir = config["project"]["output_dir"]
        self.model_name = config["model"]["base_model"]
        self.train_cfg = config["training"]

    def train(self) -> None:
        logger.info("Starting Training for Project: %s", self.config["project"]["name"])

        tokenizer = load_tokenizer(self.model_name)
        model = load_base_model(self.model_name, self.config["model"]["load_in_4bit"])

        latest_checkpoint, start_epoch = find_latest_checkpoint(self.output_dir)
        model, mode = setup_lora_model(
            model,
            self.output_dir,
            resume_checkpoint=latest_checkpoint,
            load_existing_adapter=True,
        )

        if mode in ("incremental", "fresh"):
            start_epoch = 0

        model.print_trainable_parameters()

        dataset_files = self.train_cfg.get("dataset_paths") or self.train_cfg.get("dataset_path")
        if not dataset_files:
            raise ValueError("Configuration must contain 'dataset_paths' or 'dataset_path'")

        if isinstance(dataset_files, str):
            dataset_files = [dataset_files]

        valid_files = []
        for path in dataset_files:
            if os.path.exists(path):
                valid_files.append(path)
            else:
                logger.warning("Dataset file not found (skipping): %s", path)

        if not valid_files:
            raise FileNotFoundError(f"No valid dataset files found from: {dataset_files}")

        tokenized_datasets, collator = load_and_process_dataset(
            valid_files,
            self.config["system_prompt"],
            tokenizer,
            self.train_cfg["max_seq_length"],
        )

        learning_rate = float(self.train_cfg["learning_rate"])
        optimizer = bnb.optim.AdamW32bit(model.parameters(), lr=learning_rate)

        epochs = self.train_cfg.get("epochs", 1)
        max_steps = self.train_cfg.get("max_steps")
        batch_size = self.train_cfg["batch_size"]
        gradient_accumulation_steps = self.train_cfg["gradient_accumulation_steps"]

        train_dataloader = torch.utils.data.DataLoader(
            tokenized_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        val_dataloader = torch.utils.data.DataLoader(
            tokenized_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        logger.info("Starting Training Loop...")
        step = 0
        best_val_loss = float("inf")
        if max_steps:
            epochs = 999999

        for epoch in range(start_epoch, epochs):
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(0)
                attention_mask = batch["attention_mask"].to(0)
                labels = batch["labels"].to(0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                total_train_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_loss = total_train_loss * gradient_accumulation_steps
                    progress_bar.set_postfix({"loss": current_loss})
                    total_train_loss = 0

                    global_step = (step + 1) // gradient_accumulation_steps
                    save_strategy = self.train_cfg.get("save_strategy", "epoch")
                    save_steps = int(self.train_cfg.get("save_steps", 500))

                    if save_strategy == "steps" and global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                        model.save_pretrained(checkpoint_dir)

                    if max_steps and global_step >= max_steps:
                        logger.info("Reached max_steps (%s). Stopping training.", max_steps)
                        break

                step += 1

            if max_steps and (step // gradient_accumulation_steps) >= max_steps:
                break

            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                    input_ids = batch["input_ids"].to(0)
                    attention_mask = batch["attention_mask"].to(0)
                    labels = batch["labels"].to(0)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
            logger.info("Epoch %s Complete. Validation Loss: %.4f", epoch + 1, avg_val_loss)

            if self.train_cfg.get("save_strategy", "epoch") == "epoch":
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
                model.save_pretrained(checkpoint_dir)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dir = os.path.join(self.output_dir, "best_model")
                logger.info("New Best Model! (Loss: %.4f). Saving to %s...", best_val_loss, best_dir)
                model.save_pretrained(best_dir)

        logger.info("Saving final model (latest) to %s...", self.output_dir)
        model.save_pretrained(self.output_dir)
        logger.info("Training Complete.")
