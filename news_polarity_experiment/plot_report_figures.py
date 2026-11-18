#!/usr/bin/env python3
"""Create publication-ready figures for the UltraBERT experiment report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "artifacts" / "report_analysis"
FIGURE_DIR = ROOT / "report" / "figures"

BLUE = "#2F6BFF"
ORANGE = "#F28E2B"
GREEN = "#2E9D67"
PURPLE = "#7A5AF8"
RED = "#D64B4B"
GRAY = "#7B8190"


def finish(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_points(ax, xs, ys, fmt="{:.3f}", dy=0.015):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=9)


def training_curves(data):
    epochs = np.array([row["epoch"] for row in data["epochs"]])
    train_loss = np.array([row["train_loss"] for row in data["epochs"]])
    val_loss = np.array([row["validation_loss"] for row in data["epochs"]])
    accuracy = np.array([row["validation_accuracy"] for row in data["epochs"]])
    macro_f1 = np.array([row["validation_macro_f1"] for row in data["epochs"]])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color=BLUE, label="Train loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, color=ORANGE, label="Validation loss")
    for epoch, value in zip(epochs, train_loss):
        offset = -0.045 if epoch == 3 else 0.018
        ax.text(epoch, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for epoch, value in zip(epochs, val_loss):
        ax.text(epoch, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Loss across epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xticks(epochs)
    ax.set_ylim(0, max(train_loss) * 1.18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    ax.plot(epochs, accuracy * 100, marker="o", linewidth=2.2, color=GREEN, label="Validation accuracy")
    ax.plot(epochs, macro_f1 * 100, marker="s", linewidth=2.2, color=PURPLE, label="Validation macro-F1")
    label_points(ax, epochs, accuracy * 100, fmt="{:.1f}%", dy=0.7)
    label_points(ax, epochs, macro_f1 * 100, fmt="{:.1f}%", dy=-1.6)
    ax.set_title("Validation performance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(80, 96)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle("UltraBERT-150M training dynamics (512 tokens)", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "training-curves")


def ablation_comparison(data):
    values = data["ablations"]
    context_scores = [
        values["ultrabert_128_epoch1_macro_f1"] * 100,
        values["ultrabert_512_epoch1_macro_f1"] * 100,
    ]
    epoch_scores = [
        values["ultrabert_512_epoch1_macro_f1"] * 100,
        values["ultrabert_512_epoch2_macro_f1"] * 100,
        values["ultrabert_512_epoch3_macro_f1"] * 100,
    ]
    baseline = values["tfidf_baseline_macro_f1"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    bars = ax.bar(["128 tokens", "512 tokens"], context_scores, color=[GRAY, BLUE], width=0.62)
    ax.axhline(baseline, color=ORANGE, linestyle="--", linewidth=1.8, label=f"TF-IDF baseline {baseline:.1f}%")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_title("Context-length ablation at epoch 1")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(60, 95)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    x = np.array([1, 2, 3])
    ax.plot(x, epoch_scores, marker="o", linewidth=2.4, color=PURPLE)
    ax.fill_between(x, baseline, epoch_scores, color=PURPLE, alpha=0.10)
    ax.axhline(baseline, color=ORANGE, linestyle="--", linewidth=1.8, label=f"TF-IDF baseline {baseline:.1f}%")
    label_points(ax, x, epoch_scores, fmt="{:.1f}%", dy=0.7)
    ax.set_title("Training-depth ablation at 512 tokens")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_xticks(x)
    ax.set_ylim(80, 95)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Controlled ablations: context length and training depth", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "ablation-comparison")


def long_context_extension(data):
    artifacts = ROOT / "artifacts"
    inference_1024 = json.loads(
        (artifacts / "ultrabert_validation_epoch3_inference_len1024.json").read_text(encoding="utf-8")
    )
    epoch4 = json.loads(
        (artifacts / "ultrabert_dev_seed42_len1024_from_epoch3" / "history.json").read_text(encoding="utf-8")
    )["epochs"][0]
    epochs5_6 = json.loads(
        (artifacts / "ultrabert_dev_seed42_len1024_epochs5_6" / "history.json").read_text(encoding="utf-8")
    )["epochs"]
    inference_2048 = json.loads(
        (artifacts / "ultrabert_validation_epoch6_inference_len2048.json").read_text(encoding="utf-8")
    )
    epoch7 = json.loads(
        (artifacts / "ultrabert_dev_seed42_len2048_epoch7" / "history.json").read_text(encoding="utf-8")
    )["epochs"][0]

    labels = ["512\nE3", "1024\nno FT", "1024\nE4", "1024\nE5", "1024\nE6", "2048\nno FT", "2048\nE7"]
    accuracy = [
        data["epochs"][2]["validation_accuracy"],
        inference_1024["metrics"]["accuracy"],
        epoch4["validation"]["accuracy"],
        epochs5_6[0]["validation"]["accuracy"],
        epochs5_6[1]["validation"]["accuracy"],
        inference_2048["metrics"]["accuracy"],
        epoch7["validation"]["accuracy"],
    ]
    macro_f1 = [
        data["epochs"][2]["validation_macro_f1"],
        inference_1024["metrics"]["macro_f1"],
        epoch4["validation"]["macro_f1"],
        epochs5_6[0]["validation"]["macro_f1"],
        epochs5_6[1]["validation"]["macro_f1"],
        inference_2048["metrics"]["macro_f1"],
        epoch7["validation"]["macro_f1"],
    ]
    train_loss = [epoch4["train_loss"], epochs5_6[0]["train_loss"], epochs5_6[1]["train_loss"], epoch7["train_loss"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    x = np.arange(len(labels))
    axes[0].plot(x, np.array(accuracy) * 100, marker="o", linewidth=2.2, color=GREEN, label="Accuracy")
    axes[0].plot(x, np.array(macro_f1) * 100, marker="s", linewidth=2.2, color=PURPLE, label="Macro-F1")
    label_points(axes[0], x, np.array(macro_f1) * 100, fmt="{:.2f}%", dy=-1.0)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(90, 97)
    axes[0].set_ylabel("Validation score (%)")
    axes[0].set_title("Long-context validation progression")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=0.22)

    epoch_x = np.array([4, 5, 6, 7])
    axes[1].plot(epoch_x, train_loss, marker="o", linewidth=2.4, color=BLUE)
    label_points(axes[1], epoch_x, train_loss, fmt="{:.3f}", dy=0.008)
    axes[1].set_xticks(epoch_x)
    axes[1].set_ylim(0, 0.28)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].set_title("Long-context fine-tuning loss")
    axes[1].grid(axis="y", alpha=0.22)

    fig.suptitle("UltraBERT-150M: extending context from 512 to 2,048 tokens", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "long-context-512-2048")


def confusion_matrix_figure(data):
    matrix = np.array(data["test_metrics"]["confusion_matrix"], dtype=int)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    labels = ["Left", "Center", "Right"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"fontsize": 12})
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    sns.heatmap(normalized * 100, annot=True, fmt=".1f", cmap="Purples", cbar=False, square=True,
                xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"fontsize": 11})
    axes[1].set_title("Row-normalized (%)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    fig.suptitle("UltraBERT-150M test confusion matrix (n=435)", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "test-confusion-matrix")


def best_validation_confusion_figure():
    result = json.loads(
        (ROOT / "artifacts" / "ultrabert_validation_epoch6_inference_len2048.json").read_text(encoding="utf-8")
    )
    matrix = np.array(result["metrics"]["confusion_matrix"], dtype=int)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    labels = ["Left", "Center", "Right"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"fontsize": 12})
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    sns.heatmap(normalized * 100, annot=True, fmt=".1f", cmap="Purples", cbar=False, square=True,
                xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"fontsize": 11})
    axes[1].set_title("Row-normalized (%)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    fig.suptitle(
        "Best validation candidate: Epoch 6 weights, 2,048-token input (n=434)",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    finish(fig, "best-validation-confusion-matrix")


def best_validation_class_metrics_figure():
    result = json.loads(
        (ROOT / "artifacts" / "ultrabert_validation_epoch6_inference_len2048.json").read_text(encoding="utf-8")
    )
    report = result["metrics"]["classification_report"]
    classes = ["Left", "Center", "Right"]
    keys = ["left", "center", "right"]
    precision = np.array([report[key]["precision"] for key in keys]) * 100
    recall = np.array([report[key]["recall"] for key in keys]) * 100
    f1 = np.array([report[key]["f1-score"] for key in keys]) * 100
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.bar(x - width, precision, width, label="Precision", color=BLUE)
    ax.bar(x, recall, width, label="Recall", color=ORANGE)
    ax.bar(x + width, f1, width, label="F1", color=GREEN)
    for values, offset in ((precision, -width), (recall, 0), (f1, width)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(90, 100)
    ax.set_title("Per-class validation performance at the selected 2,048-token configuration")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    finish(fig, "best-validation-class-metrics")


def class_metrics_figure(data):
    report = data["test_metrics"]["classification_report"]
    classes = ["Left", "Center", "Right"]
    keys = ["left", "center", "right"]
    precision = np.array([report[key]["precision"] for key in keys]) * 100
    recall = np.array([report[key]["recall"] for key in keys]) * 100
    f1 = np.array([report[key]["f1-score"] for key in keys]) * 100
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.bar(x - width, precision, width, label="Precision", color=BLUE)
    ax.bar(x, recall, width, label="Recall", color=ORANGE)
    ax.bar(x + width, f1, width, label="F1", color=GREEN)
    for values, offset in ((precision, -width), (recall, 0), (f1, width)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + 0.6, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(80, 96)
    ax.set_title("Per-class test performance")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    finish(fig, "test-class-metrics")


def calibration_figure(data):
    bins = [row for row in data["calibration_bins"] if row["count"] > 0]
    confidence = np.array([row["mean_confidence"] for row in bins]) * 100
    accuracy = np.array([row["accuracy"] for row in bins]) * 100
    counts = np.array([row["count"] for row in bins])
    centers = np.array([(row["lower"] + row["upper"]) / 2 for row in bins]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot([40, 100], [40, 100], linestyle="--", color=GRAY, linewidth=1.5, label="Perfect calibration")
    ax.plot(confidence, accuracy, marker="o", linewidth=2.2, color=RED, label="UltraBERT")
    for x, y, count in zip(confidence, accuracy, counts):
        ax.scatter([x], [y], s=30 + count * 0.35, color=RED, alpha=0.65)
    ax.set_xlim(40, 101)
    ax.set_ylim(25, 101)
    ax.set_xlabel("Mean confidence (%)")
    ax.set_ylabel("Empirical accuracy (%)")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.20)

    ax = axes[1]
    ax.bar(centers, counts, width=8.2, color=BLUE, alpha=0.85)
    ax.set_xlabel("Confidence bin (%)")
    ax.set_ylabel("Test articles")
    ax.set_title("Confidence distribution")
    ax.grid(axis="y", alpha=0.20)
    fig.suptitle(
        f"Test calibration: ECE={data['test_metrics']['ece_15_bins']:.3f}, "
        f"Brier={data['test_metrics']['multiclass_brier']:.3f}",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    finish(fig, "test-calibration")


def posthoc_test_epoch_comparison():
    path = ROOT / "artifacts" / "ultrabert_test_trajectory_posthoc.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"Epoch 3", "Epoch 6", "Epoch 7"}
    runs = [row for row in data["runs"] if row["stage"] in wanted]
    runs.sort(key=lambda row: row["epoch"])
    labels = [f"{row['stage']}\n{row['max_length']} tokens" for row in runs]
    accuracy = np.array([row["metrics"]["accuracy"] for row in runs]) * 100
    macro_f1 = np.array([row["metrics"]["macro_f1"] for row in runs]) * 100
    losses = np.array([row["metrics"]["cross_entropy_loss"] for row in runs])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(len(runs))
    width = 0.34
    bars_a = axes[0].bar(x - width / 2, accuracy, width, color=GREEN, label="Accuracy")
    bars_f = axes[0].bar(x + width / 2, macro_f1, width, color=PURPLE, label="Macro-F1")
    axes[0].bar_label(bars_a, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].bar_label(bars_f, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(85, 98)
    axes[0].set_ylabel("Test score (%)")
    axes[0].set_title("Classification performance")
    axes[0].legend(frameon=False, loc="lower center")
    axes[0].grid(axis="y", alpha=0.22)

    bars_l = axes[1].bar(x, losses, width=0.58, color=ORANGE)
    axes[1].bar_label(bars_l, fmt="%.3f", padding=4, fontsize=10)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(losses) * 1.20)
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Probability loss")
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Post-hoc repeated test comparison (not for unbiased model selection)", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "posthoc-test-epoch-comparison")


def posthoc_test_confusion_comparison():
    path = ROOT / "artifacts" / "ultrabert_test_trajectory_posthoc.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"Epoch 3", "Epoch 6", "Epoch 7"}
    runs = [row for row in data["runs"] if row["stage"] in wanted]
    runs.sort(key=lambda row: row["epoch"])
    labels = ["Left", "Center", "Right"]
    fig, axes = plt.subplots(1, len(runs), figsize=(12.6, 3.8))
    for ax, row in zip(axes, runs):
        matrix = np.array(row["metrics"]["confusion_matrix"], dtype=int)
        normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
        annotations = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                annotations[i, j] = f"{matrix[i, j]}\n{normalized[i, j]:.1f}%"
        sns.heatmap(
            normalized,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            square=True,
            vmin=0,
            vmax=100,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"fontsize": 9},
        )
        ax.set_title(f"{row['stage']} ({row['max_length']} tokens)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Post-hoc test confusion matrices: count and row percentage", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    finish(fig, "posthoc-test-confusion-comparison")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    data = json.loads((ANALYSIS_DIR / "analysis_metrics.json").read_text(encoding="utf-8"))
    training_curves(data)
    ablation_comparison(data)
    long_context_extension(data)
    confusion_matrix_figure(data)
    best_validation_confusion_figure()
    best_validation_class_metrics_figure()
    class_metrics_figure(data)
    calibration_figure(data)
    posthoc_test_epoch_comparison()
    posthoc_test_confusion_comparison()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

