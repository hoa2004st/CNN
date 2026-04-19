from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from engagement_pipeline.data_index import ClipRecord
from engagement_pipeline.training import TrainingConfig, train_classifier_from_feature_cache


@dataclass(frozen=True)
class AblationSpec:
    name: str
    feature_cache_root: str
    reduction_method: str
    use_smote: bool


def default_ablation_specs(
    openface_cache_root: Path,
    cnn_cache_root: Path,
    fused_cache_root: Path,
) -> List[AblationSpec]:
    return [
        AblationSpec(
            name="openface_only_baseline",
            feature_cache_root=str(openface_cache_root),
            reduction_method="none",
            use_smote=False,
        ),
        AblationSpec(
            name="cnn_only_baseline",
            feature_cache_root=str(cnn_cache_root),
            reduction_method="none",
            use_smote=False,
        ),
        AblationSpec(
            name="fused_no_reduction",
            feature_cache_root=str(fused_cache_root),
            reduction_method="none",
            use_smote=False,
        ),
        AblationSpec(
            name="fused_pca",
            feature_cache_root=str(fused_cache_root),
            reduction_method="pca",
            use_smote=False,
        ),
        AblationSpec(
            name="fused_svd",
            feature_cache_root=str(fused_cache_root),
            reduction_method="svd",
            use_smote=False,
        ),
        AblationSpec(
            name="fused_pca_smote",
            feature_cache_root=str(fused_cache_root),
            reduction_method="pca",
            use_smote=True,
        ),
    ]


def _extract_metric(summary: Dict[str, Any], split: str, key: str) -> float | None:
    split_data = summary.get("metrics", {}).get(split, {})
    if not split_data or not split_data.get("available"):
        return None
    value = split_data.get(key)
    return None if value is None else float(value)


def _write_results_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "status",
        "feature_cache_root",
        "reduction_method",
        "use_smote",
        "validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "error",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sort_key(row: Dict[str, Any]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (-1.0, -1.0, -1.0)
    return (
        float(row.get("validation_macro_f1") or 0.0),
        float(row.get("test_macro_f1") or 0.0),
        float(row.get("validation_accuracy") or 0.0),
    )


def run_ablation_suite(
    records: Sequence[ClipRecord],
    output_root: Path,
    specs: Sequence[AblationSpec],
    pooling_mode: str = "mean",
    classifier_name: str = "logistic_regression",
    class_weight: str = "balanced",
    n_components: int = 256,
    smote_k_neighbors: int = 5,
    use_scaler: bool = True,
    random_seed: int = 42,
    strict_features: bool = False,
    max_clips: int | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for spec in specs:
        run_dir = output_root / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            _, summary = train_classifier_from_feature_cache(
                records=records,
                feature_cache_root=Path(spec.feature_cache_root),
                output_dir=run_dir,
                config=TrainingConfig(
                    pooling_mode=pooling_mode,
                    reduction_method=spec.reduction_method,
                    n_components=n_components,
                    use_smote=spec.use_smote,
                    smote_k_neighbors=smote_k_neighbors,
                    classifier_name=classifier_name,
                    class_weight=class_weight,
                    use_scaler=use_scaler,
                    random_seed=random_seed,
                ),
                strict_features=strict_features,
                max_clips=max_clips,
            )

            rows.append(
                {
                    "name": spec.name,
                    "status": "ok",
                    "feature_cache_root": spec.feature_cache_root,
                    "reduction_method": spec.reduction_method,
                    "use_smote": spec.use_smote,
                    "validation_accuracy": _extract_metric(summary, "validation", "accuracy"),
                    "validation_macro_f1": _extract_metric(summary, "validation", "macro_f1"),
                    "test_accuracy": _extract_metric(summary, "test", "accuracy"),
                    "test_macro_f1": _extract_metric(summary, "test", "macro_f1"),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "name": spec.name,
                    "status": "failed",
                    "feature_cache_root": spec.feature_cache_root,
                    "reduction_method": spec.reduction_method,
                    "use_smote": spec.use_smote,
                    "validation_accuracy": None,
                    "validation_macro_f1": None,
                    "test_accuracy": None,
                    "test_macro_f1": None,
                    "error": str(exc),
                }
            )

    _write_results_csv(rows=rows, output_path=output_root / "ablation_results.csv")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "total_runs": len(rows),
        "succeeded": sum(1 for row in rows if row["status"] == "ok"),
        "failed": sum(1 for row in rows if row["status"] != "ok"),
        "pooling_mode": pooling_mode,
        "classifier_name": classifier_name,
        "class_weight": class_weight,
        "n_components": n_components,
        "smote_k_neighbors": smote_k_neighbors,
        "use_scaler": use_scaler,
        "strict_features": strict_features,
        "max_clips": max_clips,
        "runs": rows,
        "best_run": max(rows, key=_sort_key) if rows else None,
    }

    with (output_root / "ablation_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=True)

    return rows, summary
