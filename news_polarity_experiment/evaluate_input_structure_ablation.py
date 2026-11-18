#!/usr/bin/env python3
"""Fixed-weight input-structure ablation for UltraBERT."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

from run_modernbert import LABEL_TO_ID, collator, device_for, evaluate, load_records, load_tokenizer


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def variant_text(record: dict[str, Any], separator: str, variant: str) -> str:
    headline = str(record.get("headline", "")).strip()
    body = str(record.get("text", "")).strip()
    lead = " ".join(SENTENCE_RE.split(body)[:3]).strip()
    if variant == "headline_only":
        return headline
    if variant == "body_only":
        return body
    if variant == "headline_lead":
        return f"{headline} {separator} {lead}"
    if variant == "headline_full":
        return f"{headline} {separator} {body}"
    raise ValueError(variant)


class VariantDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int, variant: str):
        texts = [variant_text(record, tokenizer.sep_token or "[SEP]", variant) for record in records]
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        self.labels = [LABEL_TO_ID[record["label"]] for record in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = load_records(args.data)
    tokenizer = load_tokenizer(str(args.checkpoint), None)
    device = device_for(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, trust_remote_code=True).to(device)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "articles": len(records),
        "max_length": args.max_length,
        "device": str(device),
        "lead_definition": "first three sentences of the cleaned body",
        "runs": [],
    }
    for variant in ("headline_only", "body_only", "headline_lead", "headline_full"):
        loader = DataLoader(
            VariantDataset(records, tokenizer, args.max_length, variant),
            batch_size=1,
            shuffle=False,
            collate_fn=collator(tokenizer),
        )
        metrics = evaluate(model, loader, device)
        result["runs"].append({"variant": variant, "metrics": metrics})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}), flush=True)
        del loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
