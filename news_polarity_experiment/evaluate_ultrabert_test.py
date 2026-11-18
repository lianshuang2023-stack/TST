#!/usr/bin/env python3
"""Evaluate a saved UltraBERT checkpoint on a selected data split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from run_modernbert import (
    EncodedArticles,
    collator,
    device_for,
    evaluate,
    load_records,
    load_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Split to evaluate; defaults to the sealed test split for compatibility.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = device_for(None)
    tokenizer = load_tokenizer(str(args.checkpoint), None)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).to(device)
    records = load_records(args.data_dir / f"{args.split}.jsonl")
    loader = DataLoader(
        EncodedArticles(records, tokenizer, args.max_length),
        batch_size=1,
        shuffle=False,
        collate_fn=collator(tokenizer),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "articles": len(records),
        "max_length": args.max_length,
        "device": str(device),
        "metrics": evaluate(model, loader, device),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

