from __future__ import annotations

from typing import Any

from datasets import Dataset


DOMAIN_ALIASES = {
    "math": "Math",
    "code": "Code",
    "science": "Science",
    "general": "General",
    "customer_service": "Customer",
    "customer_tool": "Customer",
    "coding": "Coding",
    "search": "Search",
    "terminal": "Tool Interaction",
    "tool_calling": "Tool Interaction",
}


def messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("messages"):
        return row["messages"]
    return [*row.get("prompt", []), *row.get("completion", [])]


def normalized_domain(row: dict[str, Any]) -> str:
    raw = str(row.get("domain") or row.get("task") or "")
    return DOMAIN_ALIASES.get(raw, raw)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Render conversations and supervise assistant spans only."""

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = messages_from_row(row)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = rendered["input_ids"]
        assistant_mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
        if assistant_mask is None:
            raise ValueError("tokenizer chat template does not expose an assistant mask")
        labels = [token if keep else -100 for token, keep in zip(input_ids, assistant_mask)]
        return {
            "input_ids": input_ids,
            "attention_mask": rendered.get("attention_mask", [1] * len(input_ids)),
            "labels": labels,
            "domain": normalized_domain(row),
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


class CausalCollator:
    def __init__(self, tokenizer, budgets: dict[str, float]):
        self.tokenizer = tokenizer
        self.budgets = budgets

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        domains = [row["domain"] for row in rows]
        labels = [row["labels"] for row in rows]
        model_rows = [
            {key: value for key, value in row.items() if key not in ("domain", "labels")}
            for row in rows
        ]
        batch = self.tokenizer.pad(model_rows, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [values + [-100] * (max_len - len(values)) for values in labels]
        )
        batch["gradient_budgets"] = torch.tensor(
            [self.budgets.get(domain, self.budgets["default"]) for domain in domains],
            dtype=torch.float32,
        )
        return batch

