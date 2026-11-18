#!/usr/bin/env python3
"""Validation-first UltraBERT-150M training for news political-orientation labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)


LABELS = ("left", "center", "right")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass
class RunConfig:
    model_id: str
    revision: str | None
    max_length: int
    learning_rate: float
    weight_decay: float
    epochs: int
    schedule_epochs: int | None
    batch_size: int
    gradient_accumulation: int
    warmup_ratio: float
    seed: int
    device: str
    start_epoch: int = 1
    resume_from: str | None = None
    input_policy: str = "cleaned headline + text; excludes publisher, URL, author, byline, ID, topic, date, story_id"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def input_text(record: dict[str, Any], separator: str) -> str:
    # Keep all potentially label-revealing metadata out of the model input.
    return f"{record.get('headline', '').strip()} {separator} {record.get('text', '').strip()}"


class EncodedArticles(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        texts = [input_text(record, tokenizer.sep_token or "[SEP]") for record in records]
        # Fixed padding avoids recompiling a new MPS graph for every article
        # when batch_size=1 and article lengths vary.
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding="max_length"
        )
        self.labels = [LABEL_TO_ID[record["label"]] for record in records]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = {key: value[index] for key, value in self.encodings.items()}
        result["labels"] = self.labels[index]
        return result


def collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature.pop("labels") for feature in features], dtype=torch.long)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    accuracy = (prediction == labels).astype(float)
    value = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        membership = (confidence > lower) & (confidence <= upper)
        if membership.any():
            value += membership.mean() * abs(accuracy[membership].mean() - confidence[membership].mean())
    return float(value)


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(LABELS))[labels]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, labels=list(range(len(LABELS))), target_names=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(LABELS)))).tolist(),
        "labels": list(LABELS),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_15_bins": ece(probabilities, labels),
    }


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels, logits = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy())
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        logits.append(output.logits.detach().float().cpu().numpy())
    return classification_metrics(np.concatenate(labels), np.concatenate(logits))


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tokenizer(model_id: str, revision: str | None) -> Any:
    """Load UltraBERT's GPT2-compatible tokenizer across Transformers versions."""
    try:
        return AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
    except AttributeError as error:
        # Transformers 4.x expects extra_special_tokens as a mapping, while
        # UltraBERT's tokenizer_config uses the Transformers 5.x list format.
        if "keys" not in str(error):
            raise
        return GPT2TokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            extra_special_tokens={"sep_token": "<|sep|>", "mask_token": "<|mask|>"},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="dmis-lab/UltraBERT-150M")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=None,
        help="Optional scheduler horizon when intentionally running fewer epochs than the original plan.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--limit-validation", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--resume-from", type=Path, default=None, help="Load a saved checkpoint")
    parser.add_argument("--start-epoch", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = device_for(args.device)
    config = RunConfig(
        model_id=args.model_id,
        revision=args.revision,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        schedule_epochs=args.schedule_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        device=str(device),
        start_epoch=args.start_epoch,
        resume_from=str(args.resume_from) if args.resume_from else None,
    )
    save_json(args.output_dir / "config.json", asdict(config))

    model_source = args.resume_from if args.resume_from else args.model_id
    tokenizer = load_tokenizer(str(model_source), args.revision if not args.resume_from else None)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            revision=args.revision if not args.resume_from else None,
            trust_remote_code=True,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        ).to(device)
    except ImportError as error:
        raise RuntimeError(
            "UltraBERT remote code requires a compatible Transformers 5.x runtime; "
            "install the model's pinned runtime before training."
        ) from error

    train_records = load_records(args.data_dir / "train.jsonl")
    validation_records = load_records(args.data_dir / "validation.jsonl")
    if args.limit_train is not None:
        train_records = train_records[: args.limit_train]
    if args.limit_validation is not None:
        validation_records = validation_records[: args.limit_validation]
    make_loader = lambda records, shuffle: DataLoader(
        EncodedArticles(records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator(tokenizer),
    )
    train_loader, validation_loader = make_loader(train_records, True), make_loader(validation_records, False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    schedule_epochs = args.schedule_epochs if args.schedule_epochs is not None else args.epochs
    if schedule_epochs < args.epochs:
        raise ValueError("--schedule-epochs must be at least --epochs")
    total_updates = updates_per_epoch * schedule_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(total_updates * args.warmup_ratio), num_training_steps=total_updates
    )

    history: list[dict[str, Any]] = []
    best_f1, stalled_epochs = -1.0, 0
    best_dir = args.output_dir / "best_model"
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {output.loss.item()}")
            (output.loss / args.gradient_accumulation).backward()
            running_loss += output.loss.item()
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
                    "mean_loss_so_far": running_loss / step,
                    "elapsed_seconds_in_epoch": round(time.perf_counter() - epoch_started, 2),
                }
                save_json(args.output_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)

        validation = evaluate(model, validation_loader, device)
        event = {"epoch": epoch, "train_loss": running_loss / len(train_loader), "validation": validation}
        history.append(event)
        save_json(args.output_dir / "history.json", {"epochs": history})
        print(json.dumps(event, ensure_ascii=False), flush=True)

        if validation["macro_f1"] > best_f1:
            best_f1, stalled_epochs = validation["macro_f1"], 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_dir, trust_remote_code=True
    ).to(device)
    summary: dict[str, Any] = {
        "config": asdict(config),
        "best_validation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "validation": evaluate(best_model, validation_loader, device),
    }
    if args.evaluate_test:
        test_loader = make_loader(load_records(args.data_dir / "test.jsonl"), False)
        summary["test"] = evaluate(best_model, test_loader, device)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
