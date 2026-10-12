from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .data import CausalCollator, tokenize_dataset
from .objectives import soft_target_loss


class ObjectiveTrainer(Trainer):
    def __init__(self, *args, objective: str, reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.reference_model = reference_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        budgets = inputs.pop("gradient_budgets")
        outputs = model(**inputs, use_cache=False)
        if self.objective == "sft":
            loss = outputs.loss
        else:
            reference_inputs = {key: inputs[key] for key in ("input_ids", "attention_mask")}
            if next(self.reference_model.parameters()).device != inputs["input_ids"].device:
                self.reference_model.to(inputs["input_ids"].device)
            with torch.no_grad():
                base_logits = self.reference_model(**reference_inputs, use_cache=False).logits
            loss = soft_target_loss(outputs.logits, base_logits, inputs["labels"], budgets)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--track", choices=("reasoning", "agentic", "all"), default="reasoning")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=("sft", "soft"), default="soft")
    parser.add_argument("--budgets", type=Path, default=Path("configs/reasoning-1.5b.json"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = load_dataset(args.dataset, split="train")
    if args.track != "all" and "track" in raw.column_names:
        raw = raw.filter(lambda row: row["track"] == args.track)
    train_data = tokenize_dataset(raw, tokenizer, args.max_length)
    budgets = json.loads(args.budgets.read_text()) if args.objective == "soft" else {"default": 1.0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else "auto", trust_remote_code=True
    )
    reference = None
    if args.objective == "soft":
        reference = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16 if args.bf16 else "auto", trust_remote_code=True
        )
        reference.eval()
        reference.requires_grad_(False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = ObjectiveTrainer(
        model=model,
        reference_model=reference,
        objective=args.objective,
        args=training_args,
        train_dataset=train_data,
        data_collator=CausalCollator(tokenizer, budgets),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

