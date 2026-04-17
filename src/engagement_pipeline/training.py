from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from engagement_pipeline.data_index import ClipRecord
from engagement_pipeline.openface import write_manifest_jsonl

POOLING_MODES = ("mean", "max", "mean_std")
REDUCTION_METHODS = ("none", "pca", "svd")
CLASSIFIER_CHOICES = ("logistic_regression", "linear_svm", "random_forest", "mlp")


@dataclass(frozen=True)
class TrainingConfig:
    pooling_mode: str = "mean"
    reduction_method: str = "none"
    n_components: int = 256
    use_smote: bool = False
    smote_k_neighbors: int = 5
    classifier_name: str = "logistic_regression"
    class_weight: str = "balanced"
    use_scaler: bool = True
    random_seed: int = 42
    max_iter: int = 2000
    rf_estimators: int = 300
    mlp_hidden_size: int = 256
    mlp_max_iter: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pooling_mode": self.pooling_mode,
            "reduction_method": self.reduction_method,
            "n_components": self.n_components,
            "use_smote": self.use_smote,
            "smote_k_neighbors": self.smote_k_neighbors,
            "classifier_name": self.classifier_name,
            "class_weight": self.class_weight,
            "use_scaler": self.use_scaler,
            "random_seed": self.random_seed,
            "max_iter": self.max_iter,
            "rf_estimators": self.rf_estimators,
            "mlp_hidden_size": self.mlp_hidden_size,
            "mlp_max_iter": self.mlp_max_iter,
        }


def _normalize_training_config(config: TrainingConfig) -> TrainingConfig:
    pooling_mode = config.pooling_mode.strip().lower()
    reduction_method = config.reduction_method.strip().lower()
    classifier_name = config.classifier_name.strip().lower()

    if pooling_mode not in POOLING_MODES:
        allowed = ", ".join(POOLING_MODES)
        raise ValueError(f"Unsupported pooling mode '{config.pooling_mode}'. Use one of: {allowed}")

    if reduction_method not in REDUCTION_METHODS:
        allowed = ", ".join(REDUCTION_METHODS)
        raise ValueError(
            f"Unsupported reduction method '{config.reduction_method}'. Use one of: {allowed}"
        )

    if classifier_name not in CLASSIFIER_CHOICES:
        allowed = ", ".join(CLASSIFIER_CHOICES)
        raise ValueError(
            f"Unsupported classifier '{config.classifier_name}'. Use one of: {allowed}"
        )

    if config.n_components <= 0:
        raise ValueError("n_components must be positive")
    if config.smote_k_neighbors <= 0:
        raise ValueError("smote_k_neighbors must be positive")
    if config.max_iter <= 0:
        raise ValueError("max_iter must be positive")

    class_weight = config.class_weight.strip().lower()
    if class_weight not in {"balanced", "none"}:
        raise ValueError("class_weight must be either 'balanced' or 'none'")

    return TrainingConfig(
        pooling_mode=pooling_mode,
        reduction_method=reduction_method,
        n_components=config.n_components,
        use_smote=config.use_smote,
        smote_k_neighbors=config.smote_k_neighbors,
        classifier_name=classifier_name,
        class_weight=class_weight,
        use_scaler=config.use_scaler,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        rf_estimators=config.rf_estimators,
        mlp_hidden_size=config.mlp_hidden_size,
        mlp_max_iter=config.mlp_max_iter,
    )


def _feature_file_for_record(feature_cache_root: Path, record: ClipRecord) -> Path:
    return feature_cache_root / record.split / record.clip_stem / "features.npz"


def _load_feature_matrix(feature_path: Path) -> np.ndarray:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    with np.load(feature_path, allow_pickle=False) as payload:
        if "features" not in payload.files:
            raise ValueError(f"Missing 'features' array in file: {feature_path}")
        features = np.asarray(payload["features"], dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be rank-2 in {feature_path}, found rank {features.ndim}")
    if features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError(f"Feature matrix is empty in {feature_path}")

    return features


def _pool_clip_features(features: np.ndarray, pooling_mode: str) -> np.ndarray:
    if pooling_mode == "mean":
        return np.mean(features, axis=0, dtype=np.float32)

    if pooling_mode == "max":
        return np.max(features, axis=0)

    if pooling_mode == "mean_std":
        feature_mean = np.mean(features, axis=0, dtype=np.float32)
        feature_std = np.std(features, axis=0, dtype=np.float32)
        return np.concatenate([feature_mean, feature_std], axis=0)

    allowed = ", ".join(POOLING_MODES)
    raise ValueError(f"Unsupported pooling mode '{pooling_mode}'. Use one of: {allowed}")


def _build_split_arrays(
    records: Sequence[ClipRecord],
    feature_cache_root: Path,
    pooling_mode: str,
    strict_features: bool,
    max_clips: int | None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    selected_records = list(records)
    if max_clips is not None and max_clips > 0:
        selected_records = selected_records[:max_clips]

    feature_vectors: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}

    manifest_rows: List[Dict[str, Any]] = []
    failed = 0
    loaded = 0

    expected_dim: int | None = None

    for record in selected_records:
        feature_path = _feature_file_for_record(feature_cache_root=feature_cache_root, record=record)
        base_row = {
            "split": record.split,
            "clip_id": record.clip_id,
            "clip_path": record.clip_path,
            "feature_path": str(feature_path),
            "label": int(record.engagement),
        }

        try:
            matrix = _load_feature_matrix(feature_path=feature_path)
            vector = _pool_clip_features(features=matrix, pooling_mode=pooling_mode)

            if expected_dim is None:
                expected_dim = int(vector.shape[0])
            elif int(vector.shape[0]) != expected_dim:
                raise ValueError(
                    f"Pooled feature dimension mismatch for {record.clip_id}: "
                    f"expected {expected_dim}, found {int(vector.shape[0])}"
                )

            feature_vectors[record.split].append(vector.astype(np.float32, copy=False))
            labels[record.split].append(int(record.engagement))
            loaded += 1
            manifest_rows.append(
                {
                    **base_row,
                    "num_frames": int(matrix.shape[0]),
                    "num_features": int(matrix.shape[1]),
                    "pooled_dim": int(vector.shape[0]),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            manifest_rows.append(
                {
                    **base_row,
                    "num_frames": 0,
                    "num_features": 0,
                    "pooled_dim": 0,
                    "error": str(exc),
                }
            )

    if strict_features and failed > 0:
        first_error = next((row["error"] for row in manifest_rows if row["error"]), "unknown")
        raise RuntimeError(
            f"Failed to load {failed} feature rows from {feature_cache_root}. "
            f"Use --allow-missing-features to continue. First error: {first_error}"
        )

    x_by_split: Dict[str, np.ndarray] = {}
    y_by_split: Dict[str, np.ndarray] = {}

    for split in ("train", "validation", "test"):
        if feature_vectors[split]:
            x_by_split[split] = np.stack(feature_vectors[split], axis=0)
            y_by_split[split] = np.asarray(labels[split], dtype=np.int64)
        else:
            dim = 0 if expected_dim is None else expected_dim
            x_by_split[split] = np.empty((0, dim), dtype=np.float32)
            y_by_split[split] = np.empty((0,), dtype=np.int64)

    split_summary = {
        "requested": len(selected_records),
        "loaded": loaded,
        "failed": failed,
        "split_counts": {
            split: int(x_by_split[split].shape[0])
            for split in ("train", "validation", "test")
        },
        "feature_dim": 0 if expected_dim is None else expected_dim,
    }

    return x_by_split, y_by_split, manifest_rows, split_summary


def _resolve_reducer_components(train_x: np.ndarray, method: str, requested: int) -> int:
    if method == "none":
        return 0

    if train_x.shape[0] < 2:
        raise ValueError("At least 2 training samples are required for PCA/SVD reduction")

    if method == "pca":
        upper = min(train_x.shape[0], train_x.shape[1])
    elif method == "svd":
        upper = max(1, min(train_x.shape[0] - 1, train_x.shape[1]))
    else:
        raise ValueError(f"Unsupported reduction method '{method}'")

    if upper < 1:
        raise ValueError(f"No valid reducer components for method '{method}'")

    return max(1, min(requested, upper))


def _apply_preprocessing(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, Dict[str, Any]]:
    scaler = None
    reducer = None

    train_processed = train_x
    val_processed = val_x
    test_processed = test_x

    if config.use_scaler:
        scaler = StandardScaler()
        train_processed = scaler.fit_transform(train_processed)
        val_processed = scaler.transform(val_processed) if val_processed.shape[0] > 0 else val_processed
        test_processed = scaler.transform(test_processed) if test_processed.shape[0] > 0 else test_processed

    resolved_components = 0
    if config.reduction_method != "none":
        resolved_components = _resolve_reducer_components(
            train_x=train_processed,
            method=config.reduction_method,
            requested=config.n_components,
        )

        if config.reduction_method == "pca":
            reducer = PCA(n_components=resolved_components, random_state=config.random_seed)
        else:
            reducer = TruncatedSVD(n_components=resolved_components, random_state=config.random_seed)

        train_processed = reducer.fit_transform(train_processed)
        val_processed = reducer.transform(val_processed) if val_processed.shape[0] > 0 else val_processed
        test_processed = reducer.transform(test_processed) if test_processed.shape[0] > 0 else test_processed

    info = {
        "resolved_components": resolved_components,
        "train_feature_dim_after_transform": int(train_processed.shape[1]),
    }
    return train_processed, val_processed, test_processed, scaler, reducer, info


def _build_classifier(config: TrainingConfig) -> Any:
    class_weight = None if config.class_weight == "none" else config.class_weight

    if config.classifier_name == "logistic_regression":
        return LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_seed,
            class_weight=class_weight,
        )

    if config.classifier_name == "linear_svm":
        return LinearSVC(
            random_state=config.random_seed,
            class_weight=class_weight,
            max_iter=config.max_iter,
        )

    if config.classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.rf_estimators,
            random_state=config.random_seed,
            class_weight=class_weight,
            n_jobs=-1,
        )

    if config.classifier_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(config.mlp_hidden_size,),
            max_iter=config.mlp_max_iter,
            random_state=config.random_seed,
        )

    allowed = ", ".join(CLASSIFIER_CHOICES)
    raise ValueError(f"Unsupported classifier '{config.classifier_name}'. Use one of: {allowed}")


def _evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, Any]:
    if y_true.shape[0] == 0:
        return {"available": False}

    return {
        "available": True,
        "num_samples": int(y_true.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_classifier_from_feature_cache(
    records: Sequence[ClipRecord],
    feature_cache_root: Path,
    output_dir: Path,
    config: TrainingConfig,
    strict_features: bool = True,
    max_clips: int | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized_config = _normalize_training_config(config=config)

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    feature_cache_root = feature_cache_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x_by_split, y_by_split, manifest_rows, load_summary = _build_split_arrays(
        records=records,
        feature_cache_root=feature_cache_root,
        pooling_mode=normalized_config.pooling_mode,
        strict_features=strict_features,
        max_clips=max_clips,
    )

    train_x = x_by_split["train"]
    val_x = x_by_split["validation"]
    test_x = x_by_split["test"]

    train_y = y_by_split["train"]
    val_y = y_by_split["validation"]
    test_y = y_by_split["test"]

    if train_x.shape[0] == 0:
        raise RuntimeError("No train samples available after feature loading")

    train_x_proc, val_x_proc, test_x_proc, scaler, reducer, transform_info = _apply_preprocessing(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        config=normalized_config,
    )

    class_distribution_before = {
        str(int(label)): int(np.sum(train_y == label)) for label in np.unique(train_y)
    }

    smote_info = {"applied": False}
    train_x_fit = train_x_proc
    train_y_fit = train_y
    if normalized_config.use_smote:
        class_counts = [int(np.sum(train_y == label)) for label in np.unique(train_y)]
        min_class_count = min(class_counts) if class_counts else 0

        if min_class_count <= 1:
            smote_info = {
                "applied": False,
                "skipped_reason": (
                    "SMOTE skipped because at least one class has <= 1 sample in training split"
                ),
            }
        else:
            effective_k = min(normalized_config.smote_k_neighbors, min_class_count - 1)
            smote = SMOTE(
                random_state=normalized_config.random_seed,
                k_neighbors=effective_k,
            )
            train_x_fit, train_y_fit = smote.fit_resample(train_x_proc, train_y)
            smote_info = {
                "applied": True,
                "k_neighbors": effective_k,
                "class_distribution_after": {
                    str(int(label)): int(np.sum(train_y_fit == label))
                    for label in np.unique(train_y_fit)
                },
            }

    classifier = _build_classifier(config=normalized_config)
    classifier.fit(train_x_fit, train_y_fit)

    labels = sorted(np.unique(train_y).tolist())
    train_pred = classifier.predict(train_x_proc)
    val_pred = classifier.predict(val_x_proc) if val_x_proc.shape[0] > 0 else np.empty((0,), dtype=np.int64)
    test_pred = classifier.predict(test_x_proc) if test_x_proc.shape[0] > 0 else np.empty((0,), dtype=np.int64)

    metrics = {
        "train": _evaluate_split(y_true=train_y, y_pred=train_pred, labels=labels),
        "validation": _evaluate_split(y_true=val_y, y_pred=val_pred, labels=labels),
        "test": _evaluate_split(y_true=test_y, y_pred=test_pred, labels=labels),
    }

    model_bundle = {
        "config": normalized_config.to_dict(),
        "labels": labels,
        "scaler": scaler,
        "reducer": reducer,
        "classifier": classifier,
    }
    model_path = output_dir / "model_bundle.joblib"
    joblib.dump(model_bundle, model_path)

    elapsed_sec = time.perf_counter() - start_time
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "duration_sec": round(elapsed_sec, 4),
        "feature_cache_root": str(feature_cache_root),
        "model_path": str(model_path),
        "strict_features": strict_features,
        "config": normalized_config.to_dict(),
        "feature_load": load_summary,
        "transform": transform_info,
        "smote": {
            "class_distribution_before": class_distribution_before,
            **smote_info,
        },
        "metrics": metrics,
    }

    write_manifest_jsonl(rows=manifest_rows, output_path=output_dir / "feature_manifest.jsonl")
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=True)

    return manifest_rows, summary
