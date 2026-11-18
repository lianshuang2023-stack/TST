#!/usr/bin/env python3
"""Compute losses, predictions, confidence intervals, and calibration data."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from run_modernbert import EncodedArticles, collator, load_records, load_tokenizer


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ART / "splits_ultrabert_clean_70_15_15"
OUT = ART / "report_analysis"
LABELS = ("left", "center", "right")

CHECKPOINTS = {
    1: ART / "ultrabert_dev_seed42_len512" / "best_model",
    2: ART / "ultrabert_dev_seed42_len512_epoch2" / "best_model",
    3: ART / "ultrabert_dev_seed42_len512_epoch3" / "best_model",
}
HISTORIES = {
    1: ART / "ultrabert_dev_seed42_len512" / "history.json",
    2: ART / "ultrabert_dev_seed42_len512_epoch2" / "history.json",
    3: ART / "ultrabert_dev_seed42_len512_epoch3" / "history.json",
}


def make_loader(records, tokenizer):
    return DataLoader(
        EncodedArticles(records, tokenizer, 512),
        batch_size=1,
        shuffle=False,
        collate_fn=collator(tokenizer),
    )


@torch.no_grad()
def infer(checkpoint: Path, records, collect_predictions: bool = False):
    device = torch.device("cpu")
    tokenizer = load_tokenizer(str(checkpoint), None)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, trust_remote_code=True
    ).to(device)
    model.eval()
    losses: list[float] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []
    for batch in make_loader(records, tokenizer):
        labels = batch["labels"].clone()
        output = model(**{key: value.to(device) for key, value in batch.items()})
        losses.append(float(output.loss.item()))
        if collect_predictions:
            all_labels.extend(labels.numpy().tolist())
            all_logits.append(output.logits.float().cpu().numpy())
    del model
    gc.collect()
    result = {"loss": float(np.mean(losses))}
    if collect_predictions:
        logits = np.concatenate(all_logits)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        result.update(
            labels=np.asarray(all_labels, dtype=int),
            probabilities=probabilities,
            predictions=probabilities.argmax(axis=1),
        )
    return result


def bootstrap_ci(labels: np.ndarray, predictions: np.ndarray, trials: int = 2000):
    rng = np.random.default_rng(42)
    accuracy_values = np.empty(trials)
    f1_values = np.empty(trials)
    for index in range(trials):
        sample = rng.integers(0, len(labels), len(labels))
        y_true, y_pred = labels[sample], predictions[sample]
        accuracy_values[index] = accuracy_score(y_true, y_pred)
        f1_values[index] = f1_score(
            y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
        )
    return {
        "accuracy_95_ci": np.quantile(accuracy_values, [0.025, 0.975]).tolist(),
        "macro_f1_95_ci": np.quantile(f1_values, [0.025, 0.975]).tolist(),
        "bootstrap_trials": trials,
    }


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float((predictions[mask] == labels[mask]).mean()) if mask.any() else None,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validation = load_records(DATA / "validation.jsonl")
    test = load_records(DATA / "test.jsonl")

    epoch_rows = []
    for epoch in (1, 2, 3):
        history = json.loads(HISTORIES[epoch].read_text(encoding="utf-8"))["epochs"][0]
        validation_inference = infer(CHECKPOINTS[epoch], validation)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": history["train_loss"],
                "validation_loss": validation_inference["loss"],
                "validation_accuracy": history["validation"]["accuracy"],
                "validation_macro_f1": history["validation"]["macro_f1"],
            }
        )

    test_inference = infer(CHECKPOINTS[3], test, collect_predictions=True)
    labels = test_inference.pop("labels")
    probabilities = test_inference.pop("probabilities")
    predictions = test_inference.pop("predictions")

    existing_test = json.loads(
        (ART / "ultrabert_test_evaluation_epoch3.json").read_text(encoding="utf-8")
    )["metrics"]
    baseline = json.loads(
        (ART / "baseline_clean_validation.json").read_text(encoding="utf-8")
    )["validation"]
    length_128 = json.loads(
        (ART / "ultrabert_dev_seed42_len128" / "history.json").read_text(encoding="utf-8")
    )["epochs"][0]["validation"]

    analysis = {
        "epochs": epoch_rows,
        "test_loss": test_inference["loss"],
        "test_metrics": existing_test,
        "bootstrap": bootstrap_ci(labels, predictions),
        "calibration_bins": calibration_bins(labels, probabilities),
        "ablations": {
            "tfidf_baseline_macro_f1": baseline["macro_f1"],
            "ultrabert_128_epoch1_macro_f1": length_128["macro_f1"],
            "ultrabert_512_epoch1_macro_f1": epoch_rows[0]["validation_macro_f1"],
            "ultrabert_512_epoch2_macro_f1": epoch_rows[1]["validation_macro_f1"],
            "ultrabert_512_epoch3_macro_f1": epoch_rows[2]["validation_macro_f1"],
        },
    }
    (OUT / "analysis_metrics.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (OUT / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "true_label", "predicted_label", "p_left", "p_center", "p_right"])
        for record, truth, prediction, prob in zip(test, labels, predictions, probabilities):
            writer.writerow(
                [record.get("id", ""), LABELS[truth], LABELS[prediction], *[f"{value:.8f}" for value in prob]]
            )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

