from __future__ import annotations

import json
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
        messages = []
        for source in row["messages"]:
            message = {
                "role": source.get("role", ""),
                "content": source.get("content") or "",
            }
            for key in ("name", "tool_call_id"):
                if source.get(key):
                    message[key] = source[key]
            if source.get("tool_calls"):
                message["tool_calls"] = (
                    json.loads(source["tool_calls"])
                    if isinstance(source["tool_calls"], str)
                    else source["tool_calls"]
                )
            messages.append(message)
        return messages
    return [*row.get("prompt", []), *row.get("completion", [])]


def normalized_domain(row: dict[str, Any]) -> str:
    raw = str(row.get("domain") or row.get("task") or "")
    return DOMAIN_ALIASES.get(raw, raw)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Render conversations and supervise assistant spans only."""

    def find_pattern(values: list[int], pattern: list[int], start: int) -> int:
        for index in range(start, len(values) - len(pattern) + 1):
            if values[index : index + len(pattern)] == pattern:
                return index
        return -1

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = messages_from_row(row)
        tools = row.get("tools")
        if isinstance(tools, str) and tools:
            tools = json.loads(tools)
        input_ids = tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=True, add_generation_prompt=False
        )
        assistant_mask = [False] * len(input_ids)
        start_marker = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        end_marker = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        cursor = 0
        while True:
            header = find_pattern(input_ids, start_marker, cursor)
            if header < 0:
                break
            content_start = header + len(start_marker)
            end = find_pattern(input_ids, end_marker, content_start)
            content_end = len(input_ids) if end < 0 else end + len(end_marker)
            assistant_mask[content_start:content_end] = [True] * (content_end - content_start)
            cursor = content_end
        input_ids = input_ids[:max_length]
        assistant_mask = assistant_mask[:max_length]
        labels = [token if keep else -100 for token, keep in zip(input_ids, assistant_mask)]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
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

