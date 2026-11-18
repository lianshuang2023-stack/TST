#!/usr/bin/env python3
"""Validation-first TF-IDF + logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ("left", "center", "right")


def load_split(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_of(record: dict[str, Any]) -> str:
    return f"{record.get('headline', '').strip()}\n\n{record.get('text', '').strip()}"


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": list(LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    train, validation = (load_split(args.data_dir / f"{split}.jsonl") for split in ("train", "validation"))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform([text_of(record) for record in train])
    x_validation = vectorizer.transform([text_of(record) for record in validation])
    y_train = [record["label"] for record in train]
    y_validation = [record["label"] for record in validation]

    trials: dict[str, Any] = {}
    best_c, best_model, best_validation = None, None, None
    for c_value in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c_value, max_iter=1_000, random_state=42)
        model.fit(x_train, y_train)
        result = metrics(y_validation, model.predict(x_validation).tolist())
        trials[str(c_value)] = result
        if best_validation is None or result["macro_f1"] > best_validation["macro_f1"]:
            best_c, best_model, best_validation = c_value, model, result

    output: dict[str, Any] = {
        "input": "cleaned headline + text; no publisher, URL, author, byline, ID, topic, date, or story_id",
        "selected_c": best_c,
        "validation": best_validation,
        "validation_trials": trials,
    }
    if args.evaluate_test:
        test = load_split(args.data_dir / "test.jsonl")
        output["test"] = metrics([record["label"] for record in test], best_model.predict(vectorizer.transform([text_of(record) for record in test])).tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
