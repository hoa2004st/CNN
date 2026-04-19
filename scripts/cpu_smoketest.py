from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SPLIT_TO_DIR = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test",
}

SPLIT_TO_LABELS = {
    "train": "TrainLabels.csv",
    "validation": "ValidationLabels.csv",
    "test": "TestLabels.csv",
}

SPLIT_TO_MANIFEST = {
    "train": "Train.txt",
    "validation": "Validation.txt",
    "test": "Test.txt",
}

# (clip_id, engagement_label, random_seed)
MINI_DATA = {
    "train": [
        ("1100011001.avi", 0, 11),
        ("1100021001.avi", 1, 12),
        ("1100031001.avi", 2, 13),
        ("1100041001.avi", 3, 14),
    ],
    "validation": [
        ("2100011001.avi", 0, 21),
        ("2100021001.avi", 1, 22),
    ],
    "test": [
        ("3100011001.avi", 0, 31),
        ("3100021001.avi", 1, 32),
    ],
}


def _run(cmd: list[str], repo_root: Path) -> None:
    env = os.environ.copy()
    src_path = str((repo_root / "src").resolve())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def _write_tiny_video(video_path: Path, seed: int, num_frames: int = 8) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    for codec in ("MJPG", "XVID", "mp4v"):
        candidate = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            8.0,
            (64, 64),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")

    rng = np.random.default_rng(seed)
    for frame_index in range(num_frames):
        # Deterministic but non-constant frames so sampler/inference paths are exercised.
        base = np.full((64, 64, 3), fill_value=(frame_index * 13) % 255, dtype=np.uint8)
        noise = rng.integers(0, 35, size=(64, 64, 3), dtype=np.uint8)
        frame = cv2.add(base, noise)
        writer.write(frame)

    writer.release()

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create valid video file: {video_path}")


def _create_mini_daisee(daisee_root: Path) -> None:
    if daisee_root.exists():
        shutil.rmtree(daisee_root)

    dataset_root = daisee_root / "DataSet"
    labels_root = daisee_root / "Labels"
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for split, rows in MINI_DATA.items():
        split_dir_name = SPLIT_TO_DIR[split]
        split_dir = dataset_root / split_dir_name
        split_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = dataset_root / SPLIT_TO_MANIFEST[split]
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for clip_id, _, _ in rows:
                manifest_file.write(clip_id + "\n")

        labels_path = labels_root / SPLIT_TO_LABELS[split]
        with labels_path.open("w", encoding="utf-8", newline="") as labels_file:
            writer = csv.DictWriter(labels_file, fieldnames=["ClipID", "Engagement"])
            writer.writeheader()

            for clip_id, label, seed in rows:
                clip_stem = Path(clip_id).stem
                subject_id = clip_stem[:6]
                clip_path = split_dir / subject_id / clip_stem / clip_id
                _write_tiny_video(video_path=clip_path, seed=seed)
                writer.writerow({"ClipID": clip_id, "Engagement": int(label)})


def _load_index_rows(index_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _create_fake_openface_cache(index_path: Path, cache_root: Path, cnn_cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = _load_index_rows(index_path=index_path)
    for row in rows:
        split = str(row["split"])
        clip_stem = str(row["clip_stem"])
        clip_id = str(row["clip_id"])

        clip_dir = cache_root / split / clip_stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        cnn_feature_path = cnn_cache_root / split / clip_stem / "features.npz"
        if not cnn_feature_path.exists():
            raise FileNotFoundError(f"Missing CNN features while creating fake OpenFace cache: {cnn_feature_path}")

        with np.load(cnn_feature_path, allow_pickle=False) as payload:
            if "features" not in payload.files:
                raise ValueError(f"Missing 'features' array in CNN cache file: {cnn_feature_path}")
            cnn_features = np.asarray(payload["features"], dtype=np.float32)

        # Use a deterministic subset so fake OpenFace vectors differ in size from CNN vectors.
        fake_openface = cnn_features[:, :8].astype(np.float32, copy=False)
        np.savez_compressed(str(clip_dir / "features.npz"), features=fake_openface)

        metadata = {
            "clip_id": clip_id,
            "split": split,
            "shape": [int(fake_openface.shape[0]), int(fake_openface.shape[1])],
            "source": "cpu_smoketest_from_cnn_subset",
        }
        with (clip_dir / "meta.json").open("w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=True)


def run_smoketest(repo_root: Path, output_root: Path, include_ablations: bool) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    daisee_root = output_root / "mini_daisee"
    index_dir = output_root / "index"
    cnn_cache_dir = output_root / "cnn_cache"
    openface_cache_dir = output_root / "openface_cache"
    fused_cache_dir = output_root / "fused_cache"
    training_dir = output_root / "training"
    ablations_dir = output_root / "experiments"

    print("[TC1] Create tiny DAiSEE-like dataset and run build-index", flush=True)
    _create_mini_daisee(daisee_root=daisee_root)
    _run(
        [
            sys.executable,
            "-m",
            "engagement_pipeline.cli",
            "build-index",
            "--daisee-root",
            str(daisee_root),
            "--output-dir",
            str(index_dir),
        ],
        repo_root=repo_root,
    )

    index_path = index_dir / "dataset_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index output: {index_path}")

    print("[TC2] Run CPU-only CNN extraction on tiny dataset", flush=True)
    _run(
        [
            sys.executable,
            "-m",
            "engagement_pipeline.cli",
            "extract-cnn",
            "--index-path",
            str(index_path),
            "--cache-dir",
            str(cnn_cache_dir),
            "--manifest-path",
            str(cnn_cache_dir / "extraction_manifest.jsonl"),
            "--summary-path",
            str(cnn_cache_dir / "extraction_summary.json"),
            "--model-name",
            "resnet18",
            "--no-pretrained",
            "--device",
            "cpu",
            "--num-samples",
            "4",
            "--image-size",
            "64",
            "--batch-size",
            "4",
        ],
        repo_root=repo_root,
    )

    print("[TC3] Build synthetic OpenFace cache, then run fuse-features", flush=True)
    _create_fake_openface_cache(
        index_path=index_path,
        cache_root=openface_cache_dir,
        cnn_cache_root=cnn_cache_dir,
    )
    _run(
        [
            sys.executable,
            "-m",
            "engagement_pipeline.cli",
            "fuse-features",
            "--index-path",
            str(index_path),
            "--openface-cache-dir",
            str(openface_cache_dir),
            "--cnn-cache-dir",
            str(cnn_cache_dir),
            "--output-dir",
            str(fused_cache_dir),
            "--manifest-path",
            str(fused_cache_dir / "fusion_manifest.jsonl"),
            "--summary-path",
            str(fused_cache_dir / "fusion_summary.json"),
            "--alignment-mode",
            "truncate",
            "--fusion-method",
            "concat",
        ],
        repo_root=repo_root,
    )

    print("[TC4] Train classifier from fused features", flush=True)
    _run(
        [
            sys.executable,
            "-m",
            "engagement_pipeline.cli",
            "train-classifier",
            "--index-path",
            str(index_path),
            "--feature-cache-dir",
            str(fused_cache_dir),
            "--output-dir",
            str(training_dir),
            "--manifest-path",
            str(training_dir / "feature_manifest.jsonl"),
            "--summary-path",
            str(training_dir / "train_summary.json"),
            "--pooling-mode",
            "mean",
            "--reduction-method",
            "none",
            "--classifier-name",
            "logistic_regression",
            "--class-weight",
            "balanced",
        ],
        repo_root=repo_root,
    )

    if include_ablations:
        print("[TC5] Run ablations on tiny synthetic caches", flush=True)
        _run(
            [
                sys.executable,
                "-m",
                "engagement_pipeline.cli",
                "run-ablations",
                "--index-path",
                str(index_path),
                "--openface-cache-dir",
                str(openface_cache_dir),
                "--cnn-cache-dir",
                str(cnn_cache_dir),
                "--fused-cache-dir",
                str(fused_cache_dir),
                "--output-dir",
                str(ablations_dir),
                "--pooling-mode",
                "mean",
                "--classifier-name",
                "logistic_regression",
                "--n-components",
                "8",
                "--allow-missing-features",
            ],
            repo_root=repo_root,
        )

    required_outputs = [
        index_dir / "dataset_index.jsonl",
        cnn_cache_dir / "extraction_summary.json",
        fused_cache_dir / "fusion_summary.json",
        training_dir / "train_summary.json",
    ]
    for required_output in required_outputs:
        if not required_output.exists():
            raise FileNotFoundError(f"Missing expected output: {required_output}")

    print("\nCPU smoke test passed.", flush=True)
    print(f"Artifacts root: {output_root}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only smoke test runner for engagement pipeline")
    parser.add_argument(
        "--output-root",
        default="outputs/cpu_smoketest",
        help="Directory where smoke test artifacts will be generated",
    )
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also run run-ablations on the tiny synthetic dataset",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser().resolve()

    run_smoketest(
        repo_root=repo_root,
        output_root=output_root,
        include_ablations=args.include_ablations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
