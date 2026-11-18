#!/usr/bin/env python3
"""Evaluate a frozen ModernBERT checkpoint on one named data split."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_modernbert_base import Articles, collate, evaluate, load_records, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing evaluation: {args.output}")
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True).to(device)
    records = load_records(args.data_dir / f"{args.split}.jsonl")
    loader = DataLoader(
        Articles(records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate(tokenizer),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "articles": len(records),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": evaluate(model, loader, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
