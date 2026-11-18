#!/usr/bin/env python3
"""Post-hoc test evaluation for every recoverable UltraBERT training stage."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from run_modernbert import (
    EncodedArticles,
    classification_metrics,
    collator,
    device_for,
    load_records,
    load_tokenizer,
)


ROOT = Path(__file__).resolve().parent


def default_runs() -> list[dict[str, Any]]:
    artifacts = ROOT / "artifacts"
    runs = [
        {
            "stage": "128-token control",
            "epoch": 1,
            "max_length": 128,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len128" / "best_model",
            "is_primary_epoch": False,
        },
        {
            "stage": "Epoch 1",
            "epoch": 1,
            "max_length": 512,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len512" / "best_model",
            "is_primary_epoch": True,
        },
        {
            "stage": "Epoch 2",
            "epoch": 2,
            "max_length": 512,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len512_epoch2" / "best_model",
            "is_primary_epoch": True,
        },
        {
            "stage": "Epoch 3",
            "epoch": 3,
            "max_length": 512,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len512_epoch3" / "best_model",
            "is_primary_epoch": True,
        },
        {
            "stage": "Epoch 4",
            "epoch": 4,
            "max_length": 1024,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len1024_from_epoch3" / "best_model",
            "is_primary_epoch": True,
        },
        {
            "stage": "Epoch 6",
            "epoch": 6,
            "max_length": 1024,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len1024_epochs5_6" / "best_model",
            "is_primary_epoch": True,
        },
        {
            "stage": "Epoch 7",
            "epoch": 7,
            "max_length": 2048,
            "checkpoint": artifacts / "ultrabert_dev_seed42_len2048_epoch7" / "best_model",
            "is_primary_epoch": True,
        },
    ]
    return [run for run in runs if run["checkpoint"].is_dir()]


@torch.no_grad()
def evaluate_with_loss(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch_labels = batch["labels"]
        labels.append(batch_labels.numpy())
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        batch_size = int(batch_labels.shape[0])
        total_loss += float(output.loss.detach().float().cpu()) * batch_size
        total_examples += batch_size
        logits.append(output.logits.detach().float().cpu().numpy())
    metrics = classification_metrics(np.concatenate(labels), np.concatenate(logits))
    metrics["cross_entropy_loss"] = total_loss / total_examples
    return metrics


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = device_for(None)
    test_records = load_records(args.data_dir / "test.jsonl")
    result: dict[str, Any] = {
        "warning": "Post-hoc repeated test evaluation; do not use these results for unbiased model selection.",
        "split": "test",
        "articles": len(test_records),
        "device": str(device),
        "unavailable_runs": [
            {
                "stage": "Epoch 5",
                "reason": "The original checkpoint was overwritten by Epoch 6. A replay did not reproduce the original validation metrics and is excluded.",
            }
        ],
        "runs": [],
    }
    completed = set()
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        completed = {row["stage"] for row in result.get("runs", [])}

    for run in default_runs():
        if run["stage"] in completed:
            print(json.dumps({"stage": run["stage"], "status": "skipped_existing"}), flush=True)
            continue
        checkpoint = Path(run["checkpoint"])
        print(json.dumps({"stage": run["stage"], "status": "starting", "max_length": run["max_length"]}), flush=True)
        tokenizer = load_tokenizer(str(checkpoint), None)
        loader = DataLoader(
            EncodedArticles(test_records, tokenizer, int(run["max_length"])),
            batch_size=1,
            shuffle=False,
            collate_fn=collator(tokenizer),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, trust_remote_code=True
        ).to(device)
        metrics = evaluate_with_loss(model, loader, device)
        row = {
            **{key: value for key, value in run.items() if key != "checkpoint"},
            "checkpoint": str(checkpoint),
            "metrics": metrics,
        }
        result["runs"].append(row)
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "stage": run["stage"],
                    "status": "complete",
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "loss": metrics["cross_entropy_loss"],
                }
            ),
            flush=True,
        )
        model.to("cpu")
        del model, loader, tokenizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    write_json(args.output, result)


if __name__ == "__main__":
    main()
