#!/usr/bin/env python3
"""Memory-bounded fixed-shape evaluation for a large JSONL test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

from run_modernbert import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    classification_metrics,
    collator,
    device_for,
    input_text,
    load_records,
    load_tokenizer,
)


class LazyEncodedArticles(Dataset):
    def __init__(self, records, tokenizer, max_length: int, dynamic_buckets: bool):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_buckets = dynamic_buckets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        encoded = self.tokenizer(
            input_text(record, self.tokenizer.sep_token or "[SEP]"),
            truncation=True,
            max_length=self.max_length,
            padding=False if self.dynamic_buckets else "max_length",
        )
        if self.dynamic_buckets:
            target = min(self.max_length, max(128, math.ceil(len(encoded["input_ids"]) / 128) * 128))
            pad_count = target - len(encoded["input_ids"])
            encoded["input_ids"].extend([self.tokenizer.pad_token_id] * pad_count)
            encoded["attention_mask"].extend([0] * pad_count)
        encoded["labels"] = LABEL_TO_ID[record["label"]]
        return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dynamic-buckets", action="store_true")
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_partial:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    if args.sort_by_length:
        records.sort(key=lambda record: len(record.get("text", "")))
    partial_path = args.output_dir / "partial_predictions.npz"
    completed_before = 0
    total_loss = 0.0
    labels, logits = [], []
    if args.resume_partial:
        if not partial_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint found: {partial_path}")
        partial = np.load(partial_path)
        labels = partial["labels"].astype(int).tolist()
        logits = partial["logits"].astype(np.float32).tolist()
        total_loss = float(partial["total_loss"][0])
        completed_before = len(labels)
        if completed_before != len(logits) or completed_before > len(records):
            raise ValueError("Invalid partial prediction checkpoint")
    pending_records = records[completed_before:]
    tokenizer = load_tokenizer(str(args.checkpoint), None)
    loader = DataLoader(
        LazyEncodedArticles(pending_records, tokenizer, args.max_length, args.dynamic_buckets),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator(tokenizer),
    )
    device = device_for(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, trust_remote_code=True).to(device)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            batch_labels = batch["labels"].clone()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            batch_count = int(batch_labels.shape[0])
            total_loss += float(output.loss.detach().float().cpu()) * batch_count
            labels.extend(batch_labels.numpy().astype(int).tolist())
            logits.extend(output.logits.detach().float().cpu().numpy())
            articles_complete = len(labels)
            if articles_complete % args.log_every < batch_count or articles_complete == len(records):
                progress = {
                    "phase": "evaluating",
                    "articles_complete": articles_complete,
                    "articles_total": len(records),
                    "fraction": articles_complete / len(records),
                    "mean_loss": total_loss / articles_complete,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                }
                (args.output_dir / "progress.json").write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                np.savez_compressed(
                    partial_path,
                    labels=np.asarray(labels, dtype=np.int16),
                    logits=np.asarray(logits, dtype=np.float32),
                    total_loss=np.asarray([total_loss], dtype=np.float64),
                )
                print(json.dumps(progress), flush=True)

    labels_array = np.asarray(labels, dtype=int)
    logits_array = np.asarray(logits, dtype=np.float32)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels_array, logits_array)
    metrics["cross_entropy_loss"] = total_loss / len(records)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "articles": len(records),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dynamic_buckets": args.dynamic_buckets,
        "sorted_by_character_length": args.sort_by_length,
        "device": str(device),
        "input_policy": "cleaned body only because the external CSV has no headline field",
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "true_label", "predicted_label", "confidence", "p_left", "p_center", "p_right"])
        for record, truth, prediction, probs in zip(records, labels_array, predictions, probabilities):
            writer.writerow([
                record.get("id", ""), ID_TO_LABEL[int(truth)], ID_TO_LABEL[int(prediction)],
                f"{probs[prediction]:.8f}", *[f"{value:.8f}" for value in probs],
            ])
    (args.output_dir / "progress.json").write_text(
        json.dumps({"phase": "complete", "metrics": metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
