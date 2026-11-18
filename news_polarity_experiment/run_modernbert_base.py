#!/usr/bin/env python3
"""Validation-first ModernBERT-base fine-tuning for AllSides labels.

This runner is intentionally separate from the existing UltraBERT scripts in
this workspace.  It uses the standard Transformers ModernBERT implementation
and an immutable, locally cached model revision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABELS = ("left", "center", "right")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if limit is None else records[:limit]


class Articles(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        separator = tokenizer.sep_token or "[SEP]"
        # Dataset preparation removes metadata and boilerplate.  We deliberately
        # do not pass publisher, URL, IDs, topic, or date to the tokenizer.
        texts = [f"{record.get('headline') or ''} {separator} {record.get('text') or ''}" for record in records]
        # MPS recompiles graphs for changing sequence shapes.  A fixed shape is
        # materially faster and more stable for this small, fixed-length
        # ablation; it does not add metadata or change any token values.
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        self.labels = [LABEL_TO_ID[record["label"]] for record in records]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = {key: tokens[index] for key, tokens in self.encodings.items()}
        value["labels"] = self.labels[index]
        return value


def collate(tokenizer: Any):
    def apply(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([row.pop("labels") for row in features], dtype=torch.long)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return apply


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    expected: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        expected.append(batch["labels"].numpy())
        device_batch = {name: tensor.to(device) for name, tensor in batch.items()}
        output = model(**device_batch)
        logits.append(output.logits.detach().float().cpu().numpy())
    y_true = np.concatenate(expected)
    y_pred = np.concatenate(logits).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "labels": list(LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", default=MODEL_ID, help="HF model ID or local model directory")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    model_source = args.model_source
    revision = MODEL_REVISION if model_source == MODEL_ID else None
    config = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_source": model_source,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_threads": args.num_threads,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "device": str(device),
        "input_policy": "cleaned headline + text; no publisher, URL, author, byline, ID, topic, date, or story_id",
    }
    write_json(args.output_dir / "config.json", config)

    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=True,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train = load_records(args.data_dir / "train.jsonl", args.limit_train)
    validation = load_records(args.data_dir / "validation.jsonl", args.limit_validation)
    loader = lambda rows, shuffle: DataLoader(
        Articles(rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate(tokenizer),
    )
    train_loader, validation_loader = loader(train, True), loader(validation, False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )

    history: list[dict[str, Any]] = []
    best_f1, no_improvement = -1.0, 0
    best_dir = args.output_dir / "best_model"
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            device_batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = model(**device_batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
            (loss / args.gradient_accumulation).backward()
            total_loss += loss.item()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_every_steps == 0 or step == len(train_loader):
                progress = {
                    "phase": "training",
                    "epoch": epoch,
                    "step": step,
                    "steps_in_epoch": len(train_loader),
                    "mean_loss_so_far": total_loss / step,
                    "elapsed_seconds_in_epoch": round(time.perf_counter() - started, 2),
                }
                write_json(args.output_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)

        validation_metrics = evaluate(model, validation_loader, device)
        event = {"epoch": epoch, "train_loss": total_loss / len(train_loader), "validation": validation_metrics}
        history.append(event)
        write_json(args.output_dir / "history.json", {"epochs": history})
        print(json.dumps(event), flush=True)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1, no_improvement = validation_metrics["macro_f1"], 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir, local_files_only=True).to(device)
    summary = {
        "config": config,
        "epochs_completed": len(history),
        "best_validation_macro_f1": best_f1,
        "validation": evaluate(best_model, validation_loader, device),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
