from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from engagement_pipeline.data_index import ClipRecord

ALIGNMENT_MODES = ("truncate", "pad_repeat_last", "interpolate_max")
FUSION_METHODS = ("concat", "add")


@dataclass(frozen=True)
class FeatureFusionConfig:
    alignment_mode: str = "truncate"
    fusion_method: str = "concat"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alignment_mode": self.alignment_mode,
            "fusion_method": self.fusion_method,
        }


@dataclass
class FusionCacheResult:
    split: str
    clip_id: str
    clip_path: str
    cache_key: str
    cache_hit: bool
    feature_path: str
    metadata_path: str
    openface_feature_path: str
    cnn_feature_path: str
    num_frames: int
    num_features: int
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _stable_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_fusion_config(config: FeatureFusionConfig) -> FeatureFusionConfig:
    alignment = config.alignment_mode.strip().lower()
    method = config.fusion_method.strip().lower()

    if alignment not in ALIGNMENT_MODES:
        allowed = ", ".join(ALIGNMENT_MODES)
        raise ValueError(f"Unsupported alignment mode '{config.alignment_mode}'. Use one of: {allowed}")

    if method not in FUSION_METHODS:
        allowed = ", ".join(FUSION_METHODS)
        raise ValueError(f"Unsupported fusion method '{config.fusion_method}'. Use one of: {allowed}")

    return FeatureFusionConfig(alignment_mode=alignment, fusion_method=method)


def _load_feature_matrix(feature_path: Path) -> np.ndarray:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    with np.load(feature_path, allow_pickle=False) as payload:
        if "features" not in payload.files:
            raise ValueError(f"Missing 'features' array in file: {feature_path}")
        features = np.asarray(payload["features"], dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be rank-2 in {feature_path}, found rank {features.ndim}")
    if features.shape[0] <= 0:
        raise ValueError(f"Feature matrix has zero frames in {feature_path}")

    return features


def _resample_feature_length(features: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")

    current_frames = features.shape[0]
    if current_frames == target_frames:
        return features

    if current_frames == 1:
        return np.repeat(features, target_frames, axis=0)

    old_positions = np.linspace(0.0, 1.0, num=current_frames, dtype=np.float64)
    new_positions = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float64)

    result = np.empty((target_frames, features.shape[1]), dtype=np.float32)
    for column_index in range(features.shape[1]):
        result[:, column_index] = np.interp(
            new_positions,
            old_positions,
            features[:, column_index],
        ).astype(np.float32)

    return result


def _pad_last_frame(features: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")

    current_frames = features.shape[0]
    if current_frames == target_frames:
        return features
    if current_frames > target_frames:
        return features[:target_frames]

    missing = target_frames - current_frames
    last_frame = features[-1:, :]
    padding = np.repeat(last_frame, missing, axis=0)
    return np.concatenate([features, padding], axis=0)


def _align_feature_lengths(
    openface_features: np.ndarray,
    cnn_features: np.ndarray,
    alignment_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if alignment_mode == "truncate":
        target_frames = min(openface_features.shape[0], cnn_features.shape[0])
        return openface_features[:target_frames], cnn_features[:target_frames]

    if alignment_mode == "pad_repeat_last":
        target_frames = max(openface_features.shape[0], cnn_features.shape[0])
        return (
            _pad_last_frame(openface_features, target_frames=target_frames),
            _pad_last_frame(cnn_features, target_frames=target_frames),
        )

    if alignment_mode == "interpolate_max":
        target_frames = max(openface_features.shape[0], cnn_features.shape[0])
        return (
            _resample_feature_length(openface_features, target_frames=target_frames),
            _resample_feature_length(cnn_features, target_frames=target_frames),
        )

    allowed = ", ".join(ALIGNMENT_MODES)
    raise ValueError(f"Unsupported alignment mode '{alignment_mode}'. Use one of: {allowed}")


def _fuse_feature_matrices(
    openface_features: np.ndarray,
    cnn_features: np.ndarray,
    fusion_method: str,
) -> np.ndarray:
    if fusion_method == "concat":
        return np.concatenate([openface_features, cnn_features], axis=1)

    if fusion_method == "add":
        if openface_features.shape[1] != cnn_features.shape[1]:
            raise ValueError(
                "Add fusion requires equal feature dimensions, "
                f"got openface={openface_features.shape[1]} and cnn={cnn_features.shape[1]}"
            )
        return openface_features + cnn_features

    allowed = ", ".join(FUSION_METHODS)
    raise ValueError(f"Unsupported fusion method '{fusion_method}'. Use one of: {allowed}")


def _build_cache_key(
    record: ClipRecord,
    openface_feature_path: Path,
    cnn_feature_path: Path,
    config: FeatureFusionConfig,
) -> str:
    openface_stat = openface_feature_path.stat()
    cnn_stat = cnn_feature_path.stat()

    payload = {
        "split": record.split,
        "clip_id": record.clip_id,
        "openface_feature_path": str(openface_feature_path),
        "openface_size": int(openface_stat.st_size),
        "openface_mtime_ns": int(openface_stat.st_mtime_ns),
        "cnn_feature_path": str(cnn_feature_path),
        "cnn_size": int(cnn_stat.st_size),
        "cnn_mtime_ns": int(cnn_stat.st_mtime_ns),
        "config": config.to_dict(),
    }
    return _stable_hash(payload)


def fuse_or_load_features(
    record: ClipRecord,
    openface_cache_root: Path,
    cnn_cache_root: Path,
    fused_cache_root: Path,
    config: FeatureFusionConfig,
    overwrite: bool = False,
) -> FusionCacheResult:
    normalized_config = _normalize_fusion_config(config)

    openface_feature_path = openface_cache_root / record.split / record.clip_stem / "features.npz"
    cnn_feature_path = cnn_cache_root / record.split / record.clip_stem / "features.npz"

    if not openface_feature_path.exists():
        raise FileNotFoundError(f"Missing OpenFace features for clip {record.clip_id}: {openface_feature_path}")
    if not cnn_feature_path.exists():
        raise FileNotFoundError(f"Missing CNN features for clip {record.clip_id}: {cnn_feature_path}")

    cache_key = _build_cache_key(
        record=record,
        openface_feature_path=openface_feature_path,
        cnn_feature_path=cnn_feature_path,
        config=normalized_config,
    )

    clip_cache_dir = fused_cache_root / record.split / record.clip_stem
    feature_path = clip_cache_dir / "features.npz"
    metadata_path = clip_cache_dir / "meta.json"
    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and feature_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        if metadata.get("cache_key") == cache_key:
            with np.load(feature_path, allow_pickle=False) as payload:
                features = payload["features"]
                return FusionCacheResult(
                    split=record.split,
                    clip_id=record.clip_id,
                    clip_path=record.clip_path,
                    cache_key=cache_key,
                    cache_hit=True,
                    feature_path=str(feature_path),
                    metadata_path=str(metadata_path),
                    openface_feature_path=str(openface_feature_path),
                    cnn_feature_path=str(cnn_feature_path),
                    num_frames=int(features.shape[0]),
                    num_features=int(features.shape[1]),
                )

    openface_features = _load_feature_matrix(openface_feature_path)
    cnn_features = _load_feature_matrix(cnn_feature_path)

    aligned_openface, aligned_cnn = _align_feature_lengths(
        openface_features=openface_features,
        cnn_features=cnn_features,
        alignment_mode=normalized_config.alignment_mode,
    )
    fused_features = _fuse_feature_matrices(
        openface_features=aligned_openface,
        cnn_features=aligned_cnn,
        fusion_method=normalized_config.fusion_method,
    ).astype(np.float32, copy=False)

    np.savez_compressed(str(feature_path), features=fused_features)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "split": record.split,
        "clip_id": record.clip_id,
        "clip_stem": record.clip_stem,
        "clip_path": record.clip_path,
        "openface_feature_path": str(openface_feature_path),
        "cnn_feature_path": str(cnn_feature_path),
        "openface_shape": [int(openface_features.shape[0]), int(openface_features.shape[1])],
        "cnn_shape": [int(cnn_features.shape[0]), int(cnn_features.shape[1])],
        "fused_shape": [int(fused_features.shape[0]), int(fused_features.shape[1])],
        "config": normalized_config.to_dict(),
    }
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=True)

    return FusionCacheResult(
        split=record.split,
        clip_id=record.clip_id,
        clip_path=record.clip_path,
        cache_key=cache_key,
        cache_hit=False,
        feature_path=str(feature_path),
        metadata_path=str(metadata_path),
        openface_feature_path=str(openface_feature_path),
        cnn_feature_path=str(cnn_feature_path),
        num_frames=int(fused_features.shape[0]),
        num_features=int(fused_features.shape[1]),
    )


def fuse_features_for_records(
    records: Sequence[ClipRecord],
    openface_cache_root: Path,
    cnn_cache_root: Path,
    fused_cache_root: Path,
    config: FeatureFusionConfig,
    overwrite: bool = False,
    max_clips: int | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_records = list(records)
    if max_clips is not None and max_clips > 0:
        selected_records = selected_records[:max_clips]

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    cache_hits = 0
    cache_misses = 0
    failed = 0
    manifest_rows: List[Dict[str, Any]] = []

    fused_cache_root.mkdir(parents=True, exist_ok=True)

    for record in selected_records:
        try:
            result = fuse_or_load_features(
                record=record,
                openface_cache_root=openface_cache_root,
                cnn_cache_root=cnn_cache_root,
                fused_cache_root=fused_cache_root,
                config=config,
                overwrite=overwrite,
            )
            if result.cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
            manifest_rows.append(result.to_dict())
        except Exception as exc:  # noqa: BLE001
            failed += 1
            manifest_rows.append(
                {
                    "split": record.split,
                    "clip_id": record.clip_id,
                    "clip_path": record.clip_path,
                    "cache_key": "",
                    "cache_hit": False,
                    "feature_path": "",
                    "metadata_path": "",
                    "openface_feature_path": str(
                        openface_cache_root / record.split / record.clip_stem / "features.npz"
                    ),
                    "cnn_feature_path": str(cnn_cache_root / record.split / record.clip_stem / "features.npz"),
                    "num_frames": 0,
                    "num_features": 0,
                    "error": str(exc),
                }
            )

    elapsed_sec = time.perf_counter() - start_time
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "openface_cache_root": str(openface_cache_root),
        "cnn_cache_root": str(cnn_cache_root),
        "fused_cache_root": str(fused_cache_root),
        "total_requested": len(selected_records),
        "succeeded": len(selected_records) - failed,
        "failed": failed,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "duration_sec": round(elapsed_sec, 4),
        "config": _normalize_fusion_config(config).to_dict(),
    }
    return manifest_rows, summary
