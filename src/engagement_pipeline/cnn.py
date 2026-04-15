from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from engagement_pipeline.data_index import ClipRecord
from engagement_pipeline.frame_sampling import sample_video_frames

try:
    import torch
    from torch import nn
    from torchvision import models as tv_models
except Exception as exc:  # pragma: no cover - runtime environment dependent
    torch = None
    nn = None
    tv_models = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

DEFAULT_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CNNExtractionConfig:
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    weights: str = "DEFAULT"
    image_size: int = 224
    num_samples: int = 60
    batch_size: int = 16
    device: str = "auto"
    normalize_mean: Tuple[float, float, float] = DEFAULT_IMAGENET_MEAN
    normalize_std: Tuple[float, float, float] = DEFAULT_IMAGENET_STD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "weights": self.weights,
            "image_size": self.image_size,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize_mean": list(self.normalize_mean),
            "normalize_std": list(self.normalize_std),
        }


@dataclass
class CNNCacheResult:
    split: str
    clip_id: str
    clip_path: str
    cache_key: str
    cache_hit: bool
    feature_path: str
    metadata_path: str
    model_name: str
    num_frames: int
    num_features: int
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _RuntimeModel:
    model: Any
    device: str


def _require_torch() -> None:
    if torch is None or tv_models is None or nn is None:
        details = ""
        if _TORCH_IMPORT_ERROR is not None:
            details = f" Original import error: {_TORCH_IMPORT_ERROR}"
        raise RuntimeError(
            "PyTorch and torchvision are required for CNN extraction. "
            "Install dependencies from requirements.txt before running this command."
            f"{details}"
        )


def _stable_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_cache_key(record: ClipRecord, config: CNNExtractionConfig) -> str:
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


def _resolve_device(device_request: str) -> str:
    normalized = device_request.strip().lower()
    if normalized == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if normalized in {"cpu", "cuda"}:
        return normalized
    raise ValueError(f"Unsupported device '{device_request}'. Use auto, cpu, or cuda.")


def _build_torchvision_model(config: CNNExtractionConfig) -> Any:
    model_name = config.model_name

    if hasattr(tv_models, "get_model"):
        kwargs: Dict[str, Any] = {}
        if config.pretrained:
            if hasattr(tv_models, "get_model_weights"):
                weight_enum = tv_models.get_model_weights(model_name)
                try:
                    weights = (
                        weight_enum.DEFAULT
                        if config.weights.upper() == "DEFAULT"
                        else weight_enum[config.weights]
                    )
                except KeyError as exc:
                    available = ", ".join(weight_enum.__members__.keys())
                    raise ValueError(
                        f"Invalid weights '{config.weights}' for model '{model_name}'. "
                        f"Available weights: {available}"
                    ) from exc
                kwargs["weights"] = weights
            else:
                kwargs["weights"] = None
        else:
            kwargs["weights"] = None

        try:
            return tv_models.get_model(model_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            available_models = ""
            if hasattr(tv_models, "list_models"):
                available_models = ", ".join(tv_models.list_models()[:40])
            raise ValueError(
                f"Unable to construct torchvision model '{model_name}'. "
                f"Sample available models: {available_models}"
            ) from exc

    builder = getattr(tv_models, model_name, None)
    if builder is None:
        raise ValueError(f"Unknown torchvision model '{model_name}'.")

    if config.weights.upper() != "DEFAULT":
        raise ValueError(
            "Custom weight identifiers are not supported on this torchvision version. "
            "Use --weights DEFAULT or upgrade torchvision."
        )

    try:
        return builder(weights="DEFAULT" if config.pretrained else None)
    except Exception:  # noqa: BLE001
        return builder(pretrained=config.pretrained)


def _strip_classifier(model: Any) -> Any:
    modified = False

    if hasattr(model, "fc"):
        model.fc = nn.Identity()
        modified = True

    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
        modified = True

    if hasattr(model, "head"):
        model.head = nn.Identity()
        modified = True

    if not modified:
        raise ValueError(
            "Model head stripping is not implemented for this architecture. "
            "Use a torchvision model with fc/classifier/head attributes."
        )

    return model


def _create_runtime_model(config: CNNExtractionConfig) -> _RuntimeModel:
    _require_torch()

    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = _resolve_device(config.device)
    model = _build_torchvision_model(config=config)
    model = _strip_classifier(model=model)
    model.eval()
    model.to(device)

    return _RuntimeModel(model=model, device=device)


def _preprocess_frames(frames: np.ndarray, config: CNNExtractionConfig) -> Any:
    resized = np.empty(
        (frames.shape[0], config.image_size, config.image_size, 3),
        dtype=np.float32,
    )
    for frame_index, frame in enumerate(frames):
        resized_frame = cv2.resize(
            frame,
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        resized[frame_index] = resized_frame.astype(np.float32) / 255.0

    tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(config.normalize_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config.normalize_std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _run_cnn_embeddings(frame_batch: np.ndarray, runtime: _RuntimeModel, config: CNNExtractionConfig) -> np.ndarray:
    processed = _preprocess_frames(frames=frame_batch, config=config)

    embedding_batches: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, processed.shape[0], config.batch_size):
            end = min(start + config.batch_size, processed.shape[0])
            batch = processed[start:end].to(runtime.device)
            output = runtime.model(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = torch.flatten(output, start_dim=1)
            if output.ndim != 2:
                raise ValueError(
                    f"CNN model output must be rank-2 after flattening, found rank {output.ndim}."
                )
            embedding_batches.append(output.detach().cpu().to(torch.float32).numpy())

    return np.concatenate(embedding_batches, axis=0)


def _load_cache_dimensions(feature_path: Path) -> tuple[int, int]:
    with np.load(feature_path, allow_pickle=False) as payload:
        features = payload["features"]
        if features.ndim != 2:
            raise ValueError(f"Cached feature tensor must be rank-2, found {features.ndim}")
        return int(features.shape[0]), int(features.shape[1])


def extract_or_load_cnn_features(
    record: ClipRecord,
    cache_root: Path,
    config: CNNExtractionConfig,
    overwrite: bool = False,
    runtime: _RuntimeModel | None = None,
) -> CNNCacheResult:
    if not record.clip_path:
        raise ValueError(f"Clip path is missing for clip {record.clip_id}")

    clip_path = Path(record.clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip file does not exist: {clip_path}")

    cache_key = _build_cache_key(record=record, config=config)
    clip_cache_dir = cache_root / record.split / record.clip_stem
    feature_path = clip_cache_dir / "features.npz"
    metadata_path = clip_cache_dir / "meta.json"

    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and feature_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)

        if metadata.get("cache_key") == cache_key:
            num_frames, num_features = _load_cache_dimensions(feature_path=feature_path)
            return CNNCacheResult(
                split=record.split,
                clip_id=record.clip_id,
                clip_path=str(clip_path),
                cache_key=cache_key,
                cache_hit=True,
                feature_path=str(feature_path),
                metadata_path=str(metadata_path),
                model_name=config.model_name,
                num_frames=num_frames,
                num_features=num_features,
            )

    active_runtime = runtime if runtime is not None else _create_runtime_model(config=config)

    frames = sample_video_frames(
        video_path=clip_path,
        num_samples=config.num_samples,
        to_rgb=True,
    )
    feature_matrix = _run_cnn_embeddings(frame_batch=frames, runtime=active_runtime, config=config)

    np.savez_compressed(str(feature_path), features=feature_matrix)

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
        "config": config.to_dict(),
    }
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=True)

    return CNNCacheResult(
        split=record.split,
        clip_id=record.clip_id,
        clip_path=str(clip_path),
        cache_key=cache_key,
        cache_hit=False,
        feature_path=str(feature_path),
        metadata_path=str(metadata_path),
        model_name=config.model_name,
        num_frames=int(feature_matrix.shape[0]),
        num_features=int(feature_matrix.shape[1]),
    )


def extract_cnn_features_for_records(
    records: Sequence[ClipRecord],
    cache_root: Path,
    config: CNNExtractionConfig,
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

    if not selected_records:
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at,
            "cache_root": str(cache_root),
            "total_requested": 0,
            "succeeded": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "duration_sec": 0.0,
            "config": config.to_dict(),
        }
        return manifest_rows, summary

    try:
        runtime = _create_runtime_model(config=config)
    except Exception as exc:  # noqa: BLE001
        failed = len(selected_records)
        error_message = str(exc)
        for record in selected_records:
            manifest_rows.append(
                {
                    "split": record.split,
                    "clip_id": record.clip_id,
                    "clip_path": record.clip_path,
                    "cache_key": "",
                    "cache_hit": False,
                    "feature_path": "",
                    "metadata_path": "",
                    "model_name": config.model_name,
                    "num_frames": 0,
                    "num_features": 0,
                    "error": error_message,
                }
            )

        elapsed_sec = time.perf_counter() - start_time
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at,
            "cache_root": str(cache_root),
            "total_requested": len(selected_records),
            "succeeded": 0,
            "failed": failed,
            "cache_hits": 0,
            "cache_misses": 0,
            "duration_sec": round(elapsed_sec, 4),
            "config": config.to_dict(),
        }
        return manifest_rows, summary

    for record in selected_records:
        try:
            result = extract_or_load_cnn_features(
                record=record,
                cache_root=cache_root,
                config=config,
                overwrite=overwrite,
                runtime=runtime,
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
                    "model_name": config.model_name,
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
