from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from engagement_pipeline.data_index import ClipRecord

DEFAULT_FEATURE_FLAGS: Tuple[str, ...] = (
    "-2Dfp",
    "-3Dfp",
    "-pose",
    "-aus",
    "-gaze",
    "-pdmparams",
)


@dataclass(frozen=True)
class OpenFaceExtractionConfig:
    executable: str = "FeatureExtraction"
    feature_flags: Tuple[str, ...] = DEFAULT_FEATURE_FLAGS
    extra_args: Tuple[str, ...] = ()
    include_metadata_columns: bool = False
    success_only: bool = True
    copy_raw_csv: bool = True
    timeout_sec: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executable": self.executable,
            "feature_flags": list(self.feature_flags),
            "extra_args": list(self.extra_args),
            "include_metadata_columns": self.include_metadata_columns,
            "success_only": self.success_only,
            "copy_raw_csv": self.copy_raw_csv,
            "timeout_sec": self.timeout_sec,
        }


@dataclass
class OpenFaceCacheResult:
    split: str
    clip_id: str
    clip_path: str
    cache_key: str
    cache_hit: bool
    feature_path: str
    metadata_path: str
    raw_csv_path: str
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


def _build_cache_key(record: ClipRecord, config: OpenFaceExtractionConfig) -> str:
    clip_path = Path(record.clip_path)
    stat = clip_path.stat()
    payload = {
        "clip_path": str(clip_path),
        "clip_size": int(stat.st_size),
        "clip_mtime_ns": int(stat.st_mtime_ns),
        "split": record.split,
        "clip_id": record.clip_id,
        "config": config.to_dict(),
    }
    return _stable_hash(payload)


def _build_openface_command(
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    config: OpenFaceExtractionConfig,
) -> List[str]:
    command = [
        config.executable,
        "-f",
        str(video_path),
        "-out_dir",
        str(output_dir),
        "-of",
        output_stem,
    ]
    command.extend(config.feature_flags)
    command.extend(config.extra_args)
    return command


def _find_output_csv(output_dir: Path, clip_stem: str) -> Path:
    expected = output_dir / f"{clip_stem}.csv"
    if expected.exists():
        return expected

    candidates = sorted(output_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"OpenFace did not produce a CSV file in output directory: {output_dir}"
        )
    return candidates[0]


def _load_openface_features(
    csv_path: Path,
    include_metadata_columns: bool,
    success_only: bool,
) -> tuple[np.ndarray, List[str], int, int]:
    frame_table = pd.read_csv(csv_path, low_memory=False)
    numeric_table = frame_table.select_dtypes(include=["number"]).copy()
    if numeric_table.empty:
        raise ValueError(f"No numeric columns found in OpenFace CSV: {csv_path}")

    if success_only and "success" in numeric_table.columns:
        numeric_table = numeric_table[numeric_table["success"] > 0.5]

    if numeric_table.empty:
        raise ValueError(f"No valid OpenFace frames remained after filtering: {csv_path}")

    if include_metadata_columns:
        feature_columns = list(numeric_table.columns)
    else:
        metadata_columns = {"frame", "face_id", "timestamp", "confidence", "success"}
        feature_columns = [
            column_name
            for column_name in numeric_table.columns
            if column_name not in metadata_columns
        ]

    if not feature_columns:
        raise ValueError(
            f"No feature columns available after metadata filtering for CSV: {csv_path}"
        )

    feature_matrix = numeric_table.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    return feature_matrix, feature_columns, int(len(frame_table)), int(len(numeric_table))


def _load_cache_dimensions(feature_path: Path) -> tuple[int, int]:
    with np.load(feature_path, allow_pickle=False) as payload:
        features = payload["features"]
        if features.ndim != 2:
            raise ValueError(f"Cached feature tensor must be rank-2, found {features.ndim}")
        return int(features.shape[0]), int(features.shape[1])


def extract_or_load_openface_features(
    record: ClipRecord,
    cache_root: Path,
    config: OpenFaceExtractionConfig,
    overwrite: bool = False,
) -> OpenFaceCacheResult:
    if not record.clip_path:
        raise ValueError(f"Clip path is missing for clip {record.clip_id}")

    clip_path = Path(record.clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip file does not exist: {clip_path}")

    cache_key = _build_cache_key(record=record, config=config)
    clip_cache_dir = cache_root / record.split / record.clip_stem
    feature_path = clip_cache_dir / "features.npz"
    metadata_path = clip_cache_dir / "meta.json"
    raw_csv_path = clip_cache_dir / "openface_raw.csv"

    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and feature_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)

        if metadata.get("cache_key") == cache_key:
            num_frames, num_features = _load_cache_dimensions(feature_path=feature_path)
            return OpenFaceCacheResult(
                split=record.split,
                clip_id=record.clip_id,
                clip_path=str(clip_path),
                cache_key=cache_key,
                cache_hit=True,
                feature_path=str(feature_path),
                metadata_path=str(metadata_path),
                raw_csv_path=str(raw_csv_path) if raw_csv_path.exists() else "",
                num_frames=num_frames,
                num_features=num_features,
            )

    tmp_root = cache_root / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{record.clip_stem}_", dir=str(tmp_root)) as temp_output:
        temp_output_dir = Path(temp_output)
        command = _build_openface_command(
            video_path=clip_path,
            output_dir=temp_output_dir,
            output_stem=record.clip_stem,
            config=config,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
        if completed.returncode != 0:
            details = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            details = details.replace("\n", " ")
            raise RuntimeError(
                f"OpenFace extraction failed for {record.clip_id} (code {completed.returncode}): {details[:800]}"
            )

        openface_csv = _find_output_csv(output_dir=temp_output_dir, clip_stem=record.clip_stem)
        feature_matrix, feature_columns, raw_frame_count, valid_frame_count = _load_openface_features(
            csv_path=openface_csv,
            include_metadata_columns=config.include_metadata_columns,
            success_only=config.success_only,
        )

        np.savez_compressed(
            str(feature_path),
            features=feature_matrix,
            columns=np.asarray(feature_columns, dtype=np.str_),
        )

        if config.copy_raw_csv:
            shutil.copy2(openface_csv, raw_csv_path)

    source_stat = clip_path.stat()
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "split": record.split,
        "clip_id": record.clip_id,
        "clip_stem": record.clip_stem,
        "clip_path": str(clip_path),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "feature_shape": [int(feature_matrix.shape[0]), int(feature_matrix.shape[1])],
        "raw_frame_count": raw_frame_count,
        "valid_frame_count": valid_frame_count,
        "feature_columns": feature_columns,
        "config": config.to_dict(),
    }
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=True)

    return OpenFaceCacheResult(
        split=record.split,
        clip_id=record.clip_id,
        clip_path=str(clip_path),
        cache_key=cache_key,
        cache_hit=False,
        feature_path=str(feature_path),
        metadata_path=str(metadata_path),
        raw_csv_path=str(raw_csv_path) if raw_csv_path.exists() else "",
        num_frames=int(feature_matrix.shape[0]),
        num_features=int(feature_matrix.shape[1]),
    )


def extract_openface_features_for_records(
    records: Sequence[ClipRecord],
    cache_root: Path,
    config: OpenFaceExtractionConfig,
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

    cache_root.mkdir(parents=True, exist_ok=True)

    for record in selected_records:
        try:
            result = extract_or_load_openface_features(
                record=record,
                cache_root=cache_root,
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
                    "raw_csv_path": "",
                    "num_frames": 0,
                    "num_features": 0,
                    "error": str(exc),
                }
            )

    elapsed_sec = time.perf_counter() - start_time
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "cache_root": str(cache_root),
        "total_requested": len(selected_records),
        "succeeded": len(selected_records) - failed,
        "failed": failed,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "duration_sec": round(elapsed_sec, 4),
        "config": config.to_dict(),
    }
    return manifest_rows, summary


def write_manifest_jsonl(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True) + "\n")
