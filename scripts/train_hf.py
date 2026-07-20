#!/usr/bin/env python3
"""Minimal Hugging Face QAT/distillation runner for OpenBonsai experiments.

This runner is meant for small-scale hypothesis discrimination. Production
1.7B+ runs need FSDP/TPU sharding and should cache teacher targets or place the
teacher on separate workers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from openbonsai.qat import QATConfig, convert_model, distillation_loss, hidden_alignment_loss, set_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    accelerator = Accelerator(gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.get("teacher", cfg["model"]), torch_dtype=torch.bfloat16)
    teacher.eval().requires_grad_(False)
    qat = QATConfig(**cfg["qat"])
    convert_model(student, qat)

    dataset = load_dataset("json", data_files=cfg["train_jsonl"], split="train")
    sequence_length = cfg.get("sequence_length", 2048)

    def tokenize(batch):
        encoded = tokenizer(
            batch[cfg.get("text_field", "text")],
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
        )
        encoded["labels"] = [
            [token if keep else -100 for token, keep in zip(ids, mask)]
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
        ]
        return encoded

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("micro_batch_size", 1),
        shuffle=True,
        collate_fn=default_data_collator,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.get("learning_rate", 2e-5),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    student, teacher, optimizer, loader = accelerator.prepare(student, teacher, optimizer, loader)
    max_steps = int(cfg["max_steps"])
    warm_fraction = float(cfg.get("full_precision_warm_fraction", 0.05))

    completed_steps = 0
    while completed_steps < max_steps:
        for batch in loader:
            if completed_steps >= max_steps:
                break
            progress = max(
                0.0,
                (completed_steps / max_steps - warm_fraction) / max(1.0 - warm_fraction, 1e-8),
            )
            set_progress(student, progress)
            with torch.no_grad():
                teacher_inputs = {key: value for key, value in batch.items() if key != "labels"}
                teacher_out = teacher(**teacher_inputs, output_hidden_states=True)
            with accelerator.accumulate(student):
                student_out = student(**batch, output_hidden_states=True)
                loss, parts = distillation_loss(
                    student_out.logits,
                    teacher_out.logits,
                    batch["labels"],
                    temperature=cfg.get("temperature", 2.0),
                    ce_weight=cfg.get("ce_weight", 0.25),
                    kl_weight=cfg.get("kl_weight", 0.75),
                )
                hidden = hidden_alignment_loss(
                    student_out.hidden_states,
                    teacher_out.hidden_states,
                    attention_mask=batch.get("attention_mask"),
                )
                loss = loss + cfg.get("hidden_weight", 0.1) * hidden
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed_steps += 1
                if accelerator.is_main_process and completed_steps % cfg.get("log_every", 10) == 0:
                    print(json.dumps({
                        "step": completed_steps,
                        "loss": float(loss.detach()),
                        "hidden": float(hidden.detach()),
                        **parts,
                    }))

    set_progress(student, 1.0)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(student)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(cfg["output_dir"], safe_serialization=True)
        tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
