#!/usr/bin/env python3
"""Plot context-length and error analyses for the UltraBERT report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_modernbert import input_text, load_tokenizer  # noqa: E402


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ART / "splits_ultrabert_clean_70_15_15"
ANALYSIS = ART / "context_error_analysis"
FIGURES = ROOT / "report" / "figures"

BLUE = "#2F6BFF"
ORANGE = "#F28E2B"
GREEN = "#2E9D67"
PURPLE = "#7A5AF8"
RED = "#D64B4B"
GRAY = "#7B8190"
LABELS = ("left", "center", "right")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finish(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tokenize_records(records, tokenizer):
    texts = [input_text(record, tokenizer.sep_token or "[SEP]") for record in records]
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return np.asarray([len(ids) for ids in encodings["input_ids"]], dtype=int)


def bootstrap_macro_f1(y_true, y_pred, trials=2000):
    rng = np.random.default_rng(42)
    values = np.empty(trials)
    for index in range(trials):
        sample = rng.integers(0, len(y_true), len(y_true))
        values[index] = f1_score(
            y_true[sample],
            y_pred[sample],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    return np.quantile(values, [0.025, 0.975]).tolist()


def token_distribution_figure(all_lengths):
    thresholds = np.asarray([128, 512, 1024, 2048, 4096])
    coverage = np.asarray([(all_lengths <= threshold).mean() for threshold in thresholds]) * 100
    quantiles = np.quantile(all_lengths, [0.5, 0.75, 0.9, 0.95])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    bins = np.geomspace(max(16, all_lengths.min()), all_lengths.max(), 44)
    axes[0].hist(all_lengths, bins=bins, color=BLUE, alpha=0.86)
    axes[0].set_xscale("log", base=2)
    axes[0].axvline(512, color=ORANGE, linestyle="--", linewidth=1.8, label="512-token limit")
    axes[0].axvline(1024, color=GREEN, linestyle="--", linewidth=1.8, label="1,024-token limit")
    axes[0].axvline(2048, color=PURPLE, linestyle="--", linewidth=1.8, label="2,048-token limit")
    axes[0].set_xlabel("Full input length (model tokens, log2 scale)")
    axes[0].set_ylabel("Articles")
    axes[0].set_title(f"Length distribution (median {quantiles[0]:.0f} tokens)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    bars = axes[1].bar(np.arange(len(thresholds)), coverage, color=[GRAY, BLUE, GREEN, PURPLE, ORANGE])
    axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in coverage], padding=3, fontsize=9)
    axes[1].set_xticks(np.arange(len(thresholds)), [f"{value:,}" for value in thresholds])
    axes[1].set_ylim(0, 108)
    axes[1].set_xlabel("Maximum input length")
    axes[1].set_ylabel("Articles fully covered (%)")
    axes[1].set_title("Coverage without truncation")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Why long context matters: most articles exceed 512 model tokens", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "token-length-distribution-coverage")
    return {
        "articles": int(len(all_lengths)),
        "quantiles": {key: float(value) for key, value in zip(("p50", "p75", "p90", "p95"), quantiles)},
        "coverage": {str(int(threshold)): float(value / 100) for threshold, value in zip(thresholds, coverage)},
    }


def fixed_weight_context_figure():
    e3_512 = load_json(ART / "ultrabert_dev_seed42_len512_epoch3" / "summary.json")["validation"]["macro_f1"]
    e3_1024 = load_json(ART / "ultrabert_validation_epoch3_inference_len1024.json")["metrics"]["macro_f1"]
    e6_1024 = load_json(ART / "ultrabert_dev_seed42_len1024_epochs5_6" / "summary.json")["validation"]["macro_f1"]
    e6_2048 = load_json(ART / "ultrabert_validation_epoch6_inference_len2048.json")["metrics"]["macro_f1"]
    values = np.asarray([e3_512, e3_1024, e6_1024, e6_2048]) * 100
    labels = ["E3 weights\n512 input", "E3 weights\n1,024 input", "E6 weights\n1,024 input", "E6 weights\n2,048 input"]
    colors = [GRAY, BLUE, ORANGE, PURPLE]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=10)
    delta_labels = [
        (0, 1, f"context only  +{values[1] - values[0]:.2f} pp", BLUE),
        (1, 2, f"training only  +{values[2] - values[1]:.2f} pp", ORANGE),
        (2, 3, f"context only  +{values[3] - values[2]:.2f} pp", PURPLE),
    ]
    for left, right, label, color in delta_labels:
        y = max(values[left], values[right]) + 1.05
        ax.annotate("", xy=(right, y), xytext=(left, y), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5})
        ax.text((left + right) / 2, y + 0.18, label, ha="center", va="bottom", fontsize=9, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(90, 99.2)
    ax.set_ylabel("Validation macro-F1 (%)")
    ax.set_title("Controlled decomposition of context-length and training gains")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    finish(fig, "fixed-weights-context-comparison")
    return {
        "e3_weights_512": float(e3_512),
        "e3_weights_1024": float(e3_1024),
        "e6_weights_1024": float(e6_1024),
        "e6_weights_2048": float(e6_2048),
        "context_gain_e3_512_to_1024_pp": float(values[1] - values[0]),
        "training_gain_at_1024_e3_to_e6_pp": float(values[2] - values[1]),
        "context_gain_e6_1024_to_2048_pp": float(values[3] - values[2]),
    }


def normalized_confusion_figure(matrix):
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    annotations = np.empty_like(normalized, dtype=object)
    for row in range(3):
        for column in range(3):
            annotations[row, column] = f"{normalized[row, column] * 100:.1f}%\n(n={matrix[row, column]})"
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(
        normalized * 100,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        square=True,
        xticklabels=["Left", "Center", "Right"],
        yticklabels=["Left", "Center", "Right"],
        cbar_kws={"label": "Row-normalized share (%)"},
        annot_kws={"fontsize": 11},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized test confusion matrix — Epoch 3, 512-token input")
    fig.tight_layout()
    finish(fig, "normalized-confusion-matrix-e3-test")


def class_metrics_error_figure(report, matrix):
    keys = ("left", "center", "right")
    names = ("Left", "Center", "Right")
    precision = np.asarray([report[key]["precision"] for key in keys]) * 100
    recall = np.asarray([report[key]["recall"] for key in keys]) * 100
    f1 = np.asarray([report[key]["f1-score"] for key in keys]) * 100
    fp = matrix.sum(axis=0) - np.diag(matrix)
    fn = matrix.sum(axis=1) - np.diag(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    x = np.arange(3)
    width = 0.24
    axes[0].bar(x - width, precision, width, label="Precision", color=BLUE)
    axes[0].bar(x, recall, width, label="Recall", color=ORANGE)
    axes[0].bar(x + width, f1, width, label="F1", color=GREEN)
    for values, offset in ((precision, -width), (recall, 0), (f1, width)):
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(80, 96)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Per-class Precision / Recall / F1")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - 0.18, fp, 0.36, label="False positives (misreport)", color=RED)
    axes[1].bar(x + 0.18, fn, 0.36, label="False negatives (missed)", color=PURPLE)
    for values, offset in ((fp, -0.18), (fn, 0.18)):
        for index, value in enumerate(values):
            axes[1].text(index + offset, value + 0.45, str(int(value)), ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, max(fp.max(), fn.max()) + 6)
    axes[1].set_ylabel("Test articles")
    axes[1].set_title("Error source: false positives vs false negatives")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Epoch 3 class-level error analysis (test set, n=435)", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "per-class-metrics-error-source-e3-test")
    return {
        name.lower(): {
            "precision": float(precision[index] / 100),
            "recall": float(recall[index] / 100),
            "f1": float(f1[index] / 100),
            "false_positives": int(fp[index]),
            "false_negatives": int(fn[index]),
        }
        for index, name in enumerate(names)
    }


def length_binned_figure(test_records, test_lengths):
    prediction_path = ART / "report_analysis" / "test_predictions.csv"
    with prediction_path.open(encoding="utf-8", newline="") as handle:
        prediction_by_id = {row["id"]: row for row in csv.DictReader(handle)}
    y_true = np.asarray([LABEL_TO_ID[record["label"]] for record in test_records])
    y_pred = np.asarray([LABEL_TO_ID[prediction_by_id[record["id"]]["predicted_label"]] for record in test_records])

    names = ["≤256", "257–512", "513–1,024", "1,025–2,048", ">2,048"]
    bounds = [0, 256, 512, 1024, 2048, np.inf]
    rows = []
    for name, lower, upper in zip(names, bounds[:-1], bounds[1:]):
        mask = (test_lengths > lower) & (test_lengths <= upper)
        score = f1_score(y_true[mask], y_pred[mask], labels=[0, 1, 2], average="macro", zero_division=0)
        ci = bootstrap_macro_f1(y_true[mask], y_pred[mask])
        rows.append(
            {
                "length_bin": name,
                "count": int(mask.sum()),
                "macro_f1": float(score),
                "macro_f1_95_ci": ci,
                "class_support": np.bincount(y_true[mask], minlength=3).astype(int).tolist(),
            }
        )

    scores = np.asarray([row["macro_f1"] for row in rows]) * 100
    lower_error = scores - np.asarray([row["macro_f1_95_ci"][0] for row in rows]) * 100
    upper_error = np.asarray([row["macro_f1_95_ci"][1] for row in rows]) * 100 - scores
    counts = np.asarray([row["count"] for row in rows])

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars = ax.bar(np.arange(len(rows)), scores, color=[RED, ORANGE, BLUE, GREEN, PURPLE], width=0.64)
    ax.errorbar(np.arange(len(rows)), scores, yerr=np.vstack([lower_error, upper_error]), fmt="none", ecolor=GRAY, capsize=4, linewidth=1.3)
    for index, (bar, score, count) in enumerate(zip(bars, scores, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, score + upper_error[index] + 2.0, f"{score:.1f}%  (n={count})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(rows)), [row["length_bin"] for row in rows])
    ax.set_ylim(0, 112)
    ax.set_xlabel("Full article input length (model tokens)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Test Macro-F1 by article length — Epoch 3, 512-token input")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    finish(fig, "length-binned-macro-f1-e3-test")
    return rows


def main():
    checkpoint = ART / "ultrabert_dev_seed42_len512_epoch3" / "best_model"
    tokenizer = load_tokenizer(str(checkpoint), None)
    split_records = {split: load_jsonl(DATA / f"{split}.jsonl") for split in ("train", "validation", "test")}
    split_lengths = {split: tokenize_records(records, tokenizer) for split, records in split_records.items()}
    all_lengths = np.concatenate(list(split_lengths.values()))

    epoch3_test = load_json(ART / "ultrabert_test_evaluation_epoch3.json")["metrics"]
    matrix = np.asarray(epoch3_test["confusion_matrix"], dtype=int)
    results = {
        "scope": {
            "token_distribution": "train + validation + test; cleaned headline + body; model-token counts",
            "fixed_weight_comparison": "validation set",
            "confusion_class_and_length_errors": "Epoch 3 512-token checkpoint on repeatedly inspected test set; post-hoc analysis only",
        },
        "token_distribution": token_distribution_figure(all_lengths),
        "fixed_weight_context": fixed_weight_context_figure(),
    }
    normalized_confusion_figure(matrix)
    results["per_class"] = class_metrics_error_figure(epoch3_test["classification_report"], matrix)
    results["length_bins"] = length_binned_figure(split_records["test"], split_lengths["test"])

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
