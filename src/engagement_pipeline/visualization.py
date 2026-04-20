from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: Any, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _available_metric(summary: Dict[str, Any], split: str, key: str) -> float | None:
    split_summary = summary.get("metrics", {}).get(split, {})
    if not split_summary or not split_summary.get("available"):
        return None
    value = split_summary.get(key)
    return None if value is None else float(value)


def write_training_visualizations(summary: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    visualization_dir = output_dir / "visualizations"
    generated: Dict[str, str] = {}

    metrics = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
    }
    splits: list[str] = []
    for split in ("train", "validation", "test"):
        split_available = summary.get("metrics", {}).get(split, {}).get("available", False)
        if not split_available:
            continue
        splits.append(split)
        for metric_name in metrics:
            metric_value = _available_metric(summary, split, metric_name)
            metrics[metric_name].append(0.0 if metric_value is None else metric_value)

    if splits:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(splits), dtype=np.float64)
        width = 0.22
        ax.bar(x - width, metrics["accuracy"], width=width, label="Accuracy")
        ax.bar(x, metrics["macro_f1"], width=width, label="Macro-F1")
        ax.bar(x + width, metrics["weighted_f1"], width=width, label="Weighted-F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([split.title() for split in splits])
        ax.set_ylabel("Score")
        ax.set_title("Training Metrics by Split")
        ax.legend()
        generated["metrics"] = _save_figure(fig, visualization_dir / "metrics_by_split.png")

    label_distribution = summary.get("label_distribution", {})
    all_labels = sorted(
        {
            label
            for split_distribution in label_distribution.values()
            for label in split_distribution.keys()
        },
        key=lambda value: int(value),
    )
    if all_labels:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(all_labels), dtype=np.float64)
        width = 0.22
        split_order = ("train", "validation", "test")
        offsets = (-width, 0.0, width)
        for offset, split in zip(offsets, split_order, strict=True):
            counts = [
                int(label_distribution.get(split, {}).get(label, 0))
                for label in all_labels
            ]
            ax.bar(x + offset, counts, width=width, label=split.title())
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels)
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Clip Count")
        ax.set_title("Class Distribution by Split")
        ax.legend()
        generated["label_distribution"] = _save_figure(
            fig,
            visualization_dir / "label_distribution.png",
        )

    labels = summary.get("labels")
    if not labels:
        labels = sorted(
            {
                int(label)
                for split_distribution in label_distribution.values()
                for label in split_distribution.keys()
            }
        )

    for split in ("train", "validation", "test"):
        split_summary = summary.get("metrics", {}).get(split, {})
        if not split_summary.get("available"):
            continue
        confusion = split_summary.get("confusion_matrix")
        if not confusion:
            continue

        matrix = np.asarray(confusion, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{split.title()} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(
                    column_index,
                    row_index,
                    int(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="black",
                )
        generated[f"{split}_confusion_matrix"] = _save_figure(
            fig,
            visualization_dir / f"{split}_confusion_matrix.png",
        )

    return generated


def write_ablation_visualizations(summary: Dict[str, Any], output_root: Path) -> Dict[str, str]:
    output_root = output_root.expanduser().resolve()
    visualization_dir = output_root / "visualizations"
    generated: Dict[str, str] = {}

    successful_rows = [
        row
        for row in summary.get("runs", [])
        if row.get("status") == "ok"
    ]
    if successful_rows:
        names = [str(row["name"]) for row in successful_rows]
        validation_macro_f1 = [float(row.get("validation_macro_f1") or 0.0) for row in successful_rows]
        test_macro_f1 = [float(row.get("test_macro_f1") or 0.0) for row in successful_rows]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.35), 4.8))
        x = np.arange(len(names), dtype=np.float64)
        width = 0.35
        ax.bar(x - width / 2.0, validation_macro_f1, width=width, label="Validation Macro-F1")
        ax.bar(x + width / 2.0, test_macro_f1, width=width, label="Test Macro-F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Ablation Performance")
        ax.legend()
        generated["ablation_macro_f1"] = _save_figure(
            fig,
            visualization_dir / "ablation_macro_f1.png",
        )

    summary_path = visualization_dir / "visualization_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(generated, output_file, indent=2, ensure_ascii=True)

    return generated
