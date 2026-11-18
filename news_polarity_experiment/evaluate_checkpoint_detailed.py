#!/usr/bin/env python3
"""Evaluate one UltraBERT checkpoint and save per-article predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from run_modernbert import (
    EncodedArticles,
    ID_TO_LABEL,
    classification_metrics,
    collator,
    device_for,
    load_records,
    load_tokenizer,
)


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    labels, logits = [], []
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch_labels = batch["labels"].clone()
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        size = int(batch_labels.shape[0])
        total_loss += float(output.loss.detach().float().cpu()) * size
        total += size
        labels.append(batch_labels.numpy())
        logits.append(output.logits.detach().float().cpu().numpy())
    labels = np.concatenate(labels)
    logits = np.concatenate(logits)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = classification_metrics(labels, logits)
    metrics["cross_entropy_loss"] = total_loss / total
    return labels, probabilities, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(args.data)
    tokenizer = load_tokenizer(str(args.checkpoint), None)
    loader = DataLoader(
        EncodedArticles(records, tokenizer, args.max_length),
        batch_size=1,
        shuffle=False,
        collate_fn=collator(tokenizer),
    )
    device = device_for(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, trust_remote_code=True).to(device)
    labels, probabilities, metrics = infer(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "articles": len(records),
        "max_length": args.max_length,
        "device": str(device),
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "date", "publisher", "topic", "headline", "true_label", "predicted_label",
            "confidence", "p_left", "p_center", "p_right", "text_preview",
        ])
        for record, truth, prediction, probs in zip(records, labels, predictions, probabilities):
            writer.writerow([
                record.get("id", ""), record.get("date", ""), record.get("publisher", ""),
                record.get("topic", ""), record.get("headline", ""), ID_TO_LABEL[int(truth)],
                ID_TO_LABEL[int(prediction)], f"{probs[prediction]:.8f}",
                *[f"{value:.8f}" for value in probs], str(record.get("text", ""))[:600],
            ])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
