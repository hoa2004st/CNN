from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from engagement_pipeline.cnn import CNNExtractionConfig, extract_cnn_features_for_records
from engagement_pipeline.data_index import (
    SPLIT_ORDER,
    build_full_index,
    filter_records,
    read_records_jsonl,
    write_json,
    write_records_jsonl,
)
from engagement_pipeline.frame_sampling import save_sampled_frames_npy, sample_video_frames
from engagement_pipeline.experiments import default_ablation_specs, run_ablation_suite
from engagement_pipeline.fusion import (
    ALIGNMENT_MODES,
    FUSION_METHODS,
    FeatureFusionConfig,
    fuse_features_for_records,
)
from engagement_pipeline.openface import (
    DEFAULT_FEATURE_FLAGS,
    OpenFaceExtractionConfig,
    extract_openface_features_for_records,
    write_manifest_jsonl,
)
from engagement_pipeline.path_utils import resolve_user_path
from engagement_pipeline.training import (
    CLASSIFIER_CHOICES,
    POOLING_MODES,
    REDUCTION_METHODS,
    TrainingConfig,
    train_classifier_from_feature_cache,
)


def _build_index_command(args: argparse.Namespace) -> int:
    daisee_root = resolve_user_path(args.daisee_root)
    output_dir = resolve_user_path(args.output_dir)

    records_by_split, diagnostics_by_split, leakage_report = build_full_index(
        daisee_root=daisee_root,
        target_label=args.target_label,
        strict_paths=not args.allow_missing_paths,
    )

    ordered_records = [
        record
        for split in SPLIT_ORDER
        for record in records_by_split.get(split, [])
    ]

    index_path = output_dir / "dataset_index.jsonl"
    summary_path = output_dir / "index_summary.json"

    write_records_jsonl(records=ordered_records, output_path=index_path)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "daisee_root": str(daisee_root),
        "target_label": args.target_label,
        "strict_paths": not args.allow_missing_paths,
        "split_counts": {
            split: len(records_by_split.get(split, [])) for split in SPLIT_ORDER
        },
        "diagnostics": {
            split: diagnostics.to_dict()
            for split, diagnostics in diagnostics_by_split.items()
        },
        "subject_leakage": leakage_report,
    }
    write_json(data=summary, output_path=summary_path)

    print(f"Wrote index records to: {index_path}")
    print(f"Wrote index summary to: {summary_path}")

    total_missing_paths = sum(
        len(diagnostics.missing_clip_paths)
        for diagnostics in diagnostics_by_split.values()
    )
    if total_missing_paths > 0:
        print(f"Warning: {total_missing_paths} label rows have missing clip files.")

    if leakage_report:
        print("Warning: subject overlap detected across splits.")
    else:
        print("Subject split check passed: no overlap detected.")

    return 0


def _sample_video_command(args: argparse.Namespace) -> int:
    video_path = resolve_user_path(args.video_path)
    if args.output_npy:
        output_path = resolve_user_path(args.output_npy)
        frames = save_sampled_frames_npy(
            video_path=video_path,
            output_path=output_path,
            num_samples=args.num_samples,
            to_rgb=not args.keep_bgr,
        )
        print(f"Saved sampled frames to: {output_path}")
    else:
        frames = sample_video_frames(
            video_path=video_path,
            num_samples=args.num_samples,
            to_rgb=not args.keep_bgr,
        )
        print("Sampled frames in memory (no output file requested).")

    print(f"Frame batch shape: {frames.shape}")
    return 0


def _extract_openface_command(args: argparse.Namespace) -> int:
    index_path = resolve_user_path(args.index_path)
    cache_root = resolve_user_path(args.cache_dir)
    manifest_path = resolve_user_path(args.manifest_path)
    summary_path = resolve_user_path(args.summary_path)

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )
    feature_flags = tuple(args.feature_flag) if args.feature_flag else DEFAULT_FEATURE_FLAGS
    config = OpenFaceExtractionConfig(
        executable=args.openface_bin,
        feature_flags=feature_flags,
        extra_args=tuple(args.extra_arg),
        include_metadata_columns=args.include_metadata_columns,
        success_only=not args.disable_success_filter,
        copy_raw_csv=not args.skip_raw_csv_copy,
        timeout_sec=args.timeout_sec,
    )

    manifest_rows, summary = extract_openface_features_for_records(
        records=records,
        cache_root=cache_root,
        config=config,
        overwrite=args.overwrite,
    )

    write_manifest_jsonl(rows=manifest_rows, output_path=manifest_path)
    write_json(data=summary, output_path=summary_path)

    print(f"Loaded index records from: {index_path}")
    print(f"Wrote extraction manifest to: {manifest_path}")
    print(f"Wrote extraction summary to: {summary_path}")
    print(
        "OpenFace extraction summary: "
        f"requested={summary['total_requested']}, "
        f"succeeded={summary['succeeded']}, "
        f"failed={summary['failed']}, "
        f"cache_hits={summary['cache_hits']}, "
        f"cache_misses={summary['cache_misses']}"
    )

    if summary["failed"] > 0:
        print("Warning: some clips failed extraction. See manifest rows with non-empty error.")

    return 0


def _extract_cnn_command(args: argparse.Namespace) -> int:
    index_path = resolve_user_path(args.index_path)
    cache_root = resolve_user_path(args.cache_dir)
    manifest_path = resolve_user_path(args.manifest_path)
    summary_path = resolve_user_path(args.summary_path)

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )
    config = CNNExtractionConfig(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        weights=args.weights,
        image_size=args.image_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )

    manifest_rows, summary = extract_cnn_features_for_records(
        records=records,
        cache_root=cache_root,
        config=config,
        overwrite=args.overwrite,
    )

    write_manifest_jsonl(rows=manifest_rows, output_path=manifest_path)
    write_json(data=summary, output_path=summary_path)

    print(f"Loaded index records from: {index_path}")
    print(f"Wrote CNN extraction manifest to: {manifest_path}")
    print(f"Wrote CNN extraction summary to: {summary_path}")
    print(
        "CNN extraction summary: "
        f"requested={summary['total_requested']}, "
        f"succeeded={summary['succeeded']}, "
        f"failed={summary['failed']}, "
        f"cache_hits={summary['cache_hits']}, "
        f"cache_misses={summary['cache_misses']}"
    )

    if summary["failed"] > 0:
        print("Warning: some clips failed extraction. See manifest rows with non-empty error.")

    return 0


def _fuse_features_command(args: argparse.Namespace) -> int:
    index_path = resolve_user_path(args.index_path)
    openface_cache_root = resolve_user_path(args.openface_cache_dir)
    cnn_cache_root = resolve_user_path(args.cnn_cache_dir)
    fused_cache_root = resolve_user_path(args.output_dir)
    manifest_path = resolve_user_path(args.manifest_path)
    summary_path = resolve_user_path(args.summary_path)

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )
    config = FeatureFusionConfig(
        alignment_mode=args.alignment_mode,
        fusion_method=args.fusion_method,
    )

    manifest_rows, summary = fuse_features_for_records(
        records=records,
        openface_cache_root=openface_cache_root,
        cnn_cache_root=cnn_cache_root,
        fused_cache_root=fused_cache_root,
        config=config,
        overwrite=args.overwrite,
    )

    write_manifest_jsonl(rows=manifest_rows, output_path=manifest_path)
    write_json(data=summary, output_path=summary_path)

    print(f"Loaded index records from: {index_path}")
    print(f"Wrote fusion manifest to: {manifest_path}")
    print(f"Wrote fusion summary to: {summary_path}")
    print(
        "Fusion summary: "
        f"requested={summary['total_requested']}, "
        f"succeeded={summary['succeeded']}, "
        f"failed={summary['failed']}, "
        f"cache_hits={summary['cache_hits']}, "
        f"cache_misses={summary['cache_misses']}"
    )

    if summary["failed"] > 0:
        print("Warning: some clips failed fusion. See manifest rows with non-empty error.")

    return 0


def _train_classifier_command(args: argparse.Namespace) -> int:
    index_path = resolve_user_path(args.index_path)
    feature_cache_root = resolve_user_path(args.feature_cache_dir)
    output_dir = resolve_user_path(args.output_dir)
    manifest_path = resolve_user_path(args.manifest_path)
    summary_path = resolve_user_path(args.summary_path)

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )
    config = TrainingConfig(
        pooling_mode=args.pooling_mode,
        reduction_method=args.reduction_method,
        n_components=args.n_components,
        use_smote=args.enable_smote,
        smote_k_neighbors=args.smote_k_neighbors,
        classifier_name=args.classifier_name,
        class_weight=args.class_weight,
        use_scaler=not args.disable_scaler,
        random_seed=args.random_seed,
    )

    manifest_rows, summary = train_classifier_from_feature_cache(
        records=records,
        feature_cache_root=feature_cache_root,
        output_dir=output_dir,
        config=config,
        strict_features=not args.allow_missing_features,
    )

    write_manifest_jsonl(rows=manifest_rows, output_path=manifest_path)
    write_json(data=summary, output_path=summary_path)

    print(f"Loaded index records from: {index_path}")
    print(f"Using feature cache root: {feature_cache_root}")
    print(f"Wrote training manifest to: {manifest_path}")
    print(f"Wrote training summary to: {summary_path}")
    print(f"Model bundle: {summary['model_path']}")

    val_metrics = summary.get("metrics", {}).get("validation", {})
    test_metrics = summary.get("metrics", {}).get("test", {})
    if val_metrics.get("available"):
        print(
            f"Validation: accuracy={val_metrics.get('accuracy', 0):.4f}, "
            f"macro_f1={val_metrics.get('macro_f1', 0):.4f}"
        )
    if test_metrics.get("available"):
        print(
            f"Test: accuracy={test_metrics.get('accuracy', 0):.4f}, "
            f"macro_f1={test_metrics.get('macro_f1', 0):.4f}"
        )

    return 0


def _run_ablations_command(args: argparse.Namespace) -> int:
    index_path = resolve_user_path(args.index_path)
    output_dir = resolve_user_path(args.output_dir)
    openface_cache_root = resolve_user_path(args.openface_cache_dir)
    cnn_cache_root = resolve_user_path(args.cnn_cache_dir)
    fused_cache_root = resolve_user_path(args.fused_cache_dir)

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )
    specs = default_ablation_specs(
        openface_cache_root=openface_cache_root,
        cnn_cache_root=cnn_cache_root,
        fused_cache_root=fused_cache_root,
    )

    _, summary = run_ablation_suite(
        records=records,
        output_root=output_dir,
        specs=specs,
        pooling_mode=args.pooling_mode,
        classifier_name=args.classifier_name,
        class_weight=args.class_weight,
        n_components=args.n_components,
        smote_k_neighbors=args.smote_k_neighbors,
        use_scaler=not args.disable_scaler,
        random_seed=args.random_seed,
        strict_features=not args.allow_missing_features,
    )

    print(f"Loaded index records from: {index_path}")
    print(f"Ablation output dir: {output_dir}")
    print(f"Ablation summary: {output_dir / 'ablation_summary.json'}")
    print(f"Ablation table: {output_dir / 'ablation_results.csv'}")
    print(
        "Ablation status: "
        f"total={summary['total_runs']}, "
        f"succeeded={summary['succeeded']}, "
        f"failed={summary['failed']}"
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engagement-pipeline",
        description="DAiSEE engagement data utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_record_selection_arguments(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument(
            "--split",
            action="append",
            default=[],
            help="Restrict processing to one or more splits (repeatable: train, validation, test).",
        )
        target_parser.add_argument(
            "--max-clips",
            type=int,
            default=0,
            help="Optional cap on number of selected records to process after filtering (0 means all).",
        )
        target_parser.add_argument(
            "--max-clips-per-split",
            type=int,
            default=0,
            help="Optional per-split cap applied before any global max-clips limit (0 means all).",
        )

    index_parser = subparsers.add_parser(
        "build-index",
        help="Build Engagement index and split validation report from DAiSEE labels.",
    )
    index_parser.add_argument(
        "--daisee-root",
        required=True,
        help="Path to DAiSEE root directory containing DataSet/ and Labels/.",
    )
    index_parser.add_argument(
        "--output-dir",
        default="outputs/index",
        help="Output directory for generated index files.",
    )
    index_parser.add_argument(
        "--target-label",
        default="Engagement",
        help="Target label column from DAiSEE labels CSV.",
    )
    index_parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        help="Keep rows with missing clip files in the index (clip_path will be empty).",
    )
    index_parser.set_defaults(handler=_build_index_command)

    sample_parser = subparsers.add_parser(
        "sample-video",
        help="Uniformly sample frames from a video clip.",
    )
    sample_parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the source video clip.",
    )
    sample_parser.add_argument(
        "--num-samples",
        type=int,
        default=60,
        help="Number of frames to sample.",
    )
    sample_parser.add_argument(
        "--output-npy",
        default="",
        help="Optional .npy output path for sampled frame tensor.",
    )
    sample_parser.add_argument(
        "--keep-bgr",
        action="store_true",
        help="Keep OpenCV BGR channel order instead of converting to RGB.",
    )
    sample_parser.set_defaults(handler=_sample_video_command)

    extract_parser = subparsers.add_parser(
        "extract-openface",
        help="Extract and cache OpenFace features for clips in an index file.",
    )
    extract_parser.add_argument(
        "--index-path",
        default="outputs/index/dataset_index.jsonl",
        help="Path to index JSONL generated by build-index.",
    )
    extract_parser.add_argument(
        "--openface-bin",
        default="FeatureExtraction",
        help="OpenFace FeatureExtraction executable path or command name.",
    )
    extract_parser.add_argument(
        "--cache-dir",
        default="outputs/features/openface_cache",
        help="Root directory for cached OpenFace features.",
    )
    extract_parser.add_argument(
        "--manifest-path",
        default="outputs/features/openface_cache/extraction_manifest.jsonl",
        help="Output JSONL path for per-clip extraction outcomes.",
    )
    extract_parser.add_argument(
        "--summary-path",
        default="outputs/features/openface_cache/extraction_summary.json",
        help="Output JSON path for extraction summary stats.",
    )
    extract_parser.add_argument(
        "--timeout-sec",
        type=int,
        default=900,
        help="Per-clip OpenFace process timeout in seconds.",
    )
    add_record_selection_arguments(extract_parser)
    extract_parser.add_argument(
        "--feature-flag",
        action="append",
        default=[],
        help="Override default OpenFace feature flags (repeatable).",
    )
    extract_parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument passed through to OpenFace (repeatable).",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-extraction even when cache key matches.",
    )
    extract_parser.add_argument(
        "--include-metadata-columns",
        action="store_true",
        help="Keep metadata columns like frame/timestamp in cached features.",
    )
    extract_parser.add_argument(
        "--disable-success-filter",
        action="store_true",
        help="Do not filter OpenFace rows by success > 0.5.",
    )
    extract_parser.add_argument(
        "--skip-raw-csv-copy",
        action="store_true",
        help="Do not copy OpenFace raw CSV into cache directories.",
    )
    extract_parser.set_defaults(handler=_extract_openface_command)

    extract_cnn_parser = subparsers.add_parser(
        "extract-cnn",
        help="Extract and cache CNN frame embeddings for clips in an index file.",
    )
    extract_cnn_parser.add_argument(
        "--index-path",
        default="outputs/index/dataset_index.jsonl",
        help="Path to index JSONL generated by build-index.",
    )
    extract_cnn_parser.add_argument(
        "--cache-dir",
        default="outputs/features/cnn_cache",
        help="Root directory for cached CNN features.",
    )
    extract_cnn_parser.add_argument(
        "--manifest-path",
        default="outputs/features/cnn_cache/extraction_manifest.jsonl",
        help="Output JSONL path for per-clip extraction outcomes.",
    )
    extract_cnn_parser.add_argument(
        "--summary-path",
        default="outputs/features/cnn_cache/extraction_summary.json",
        help="Output JSON path for extraction summary stats.",
    )
    extract_cnn_parser.add_argument(
        "--model-name",
        default="efficientnet_b0",
        help="Torchvision model name used as embedding backbone.",
    )
    extract_cnn_parser.add_argument(
        "--weights",
        default="DEFAULT",
        help="Torchvision weight enum name (DEFAULT by default).",
    )
    extract_cnn_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights and initialize model weights randomly.",
    )
    extract_cnn_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size for frame resizing before inference.",
    )
    extract_cnn_parser.add_argument(
        "--num-samples",
        type=int,
        default=60,
        help="Number of uniformly sampled frames per clip.",
    )
    extract_cnn_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size in frames.",
    )
    extract_cnn_parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda.",
    )
    add_record_selection_arguments(extract_cnn_parser)
    extract_cnn_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-extraction even when cache key matches.",
    )
    extract_cnn_parser.set_defaults(handler=_extract_cnn_command)

    fuse_parser = subparsers.add_parser(
        "fuse-features",
        help="Fuse cached OpenFace and CNN features clip-by-clip.",
    )
    fuse_parser.add_argument(
        "--index-path",
        default="outputs/index/dataset_index.jsonl",
        help="Path to index JSONL generated by build-index.",
    )
    fuse_parser.add_argument(
        "--openface-cache-dir",
        default="outputs/features/openface_cache",
        help="Root directory of cached OpenFace feature files.",
    )
    fuse_parser.add_argument(
        "--cnn-cache-dir",
        default="outputs/features/cnn_cache",
        help="Root directory of cached CNN feature files.",
    )
    fuse_parser.add_argument(
        "--output-dir",
        default="outputs/features/fused_cache",
        help="Output directory for fused features.",
    )
    fuse_parser.add_argument(
        "--manifest-path",
        default="outputs/features/fused_cache/fusion_manifest.jsonl",
        help="Output JSONL path for per-clip fusion outcomes.",
    )
    fuse_parser.add_argument(
        "--summary-path",
        default="outputs/features/fused_cache/fusion_summary.json",
        help="Output JSON path for fusion summary stats.",
    )
    fuse_parser.add_argument(
        "--alignment-mode",
        choices=list(ALIGNMENT_MODES),
        default="truncate",
        help="Temporal alignment strategy before feature fusion.",
    )
    fuse_parser.add_argument(
        "--fusion-method",
        choices=list(FUSION_METHODS),
        default="concat",
        help="Feature fusion method after temporal alignment.",
    )
    add_record_selection_arguments(fuse_parser)
    fuse_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-fusion even when cache key matches.",
    )
    fuse_parser.set_defaults(handler=_fuse_features_command)

    train_parser = subparsers.add_parser(
        "train-classifier",
        help="Train and evaluate a classifier from cached clip-level features.",
    )
    train_parser.add_argument(
        "--index-path",
        default="outputs/index/dataset_index.jsonl",
        help="Path to index JSONL generated by build-index.",
    )
    train_parser.add_argument(
        "--feature-cache-dir",
        default="outputs/features/fused_cache",
        help="Feature cache root used for training (fused/openface/cnn).",
    )
    train_parser.add_argument(
        "--output-dir",
        default="outputs/training",
        help="Output directory for trained model and metrics.",
    )
    train_parser.add_argument(
        "--manifest-path",
        default="outputs/training/feature_manifest.jsonl",
        help="Output JSONL path for loaded feature rows.",
    )
    train_parser.add_argument(
        "--summary-path",
        default="outputs/training/train_summary.json",
        help="Output JSON path for training summary.",
    )
    train_parser.add_argument(
        "--pooling-mode",
        choices=list(POOLING_MODES),
        default="mean",
        help="Clip-level pooling strategy for frame features.",
    )
    train_parser.add_argument(
        "--reduction-method",
        choices=list(REDUCTION_METHODS),
        default="none",
        help="Dimensionality reduction method before classifier fitting.",
    )
    train_parser.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="Requested PCA/SVD components when reduction is enabled.",
    )
    train_parser.add_argument(
        "--classifier-name",
        choices=list(CLASSIFIER_CHOICES),
        default="logistic_regression",
        help="Classifier used for engagement prediction.",
    )
    train_parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy passed to supported classifiers.",
    )
    train_parser.add_argument(
        "--enable-smote",
        action="store_true",
        help="Apply SMOTE on transformed training features.",
    )
    train_parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="k_neighbors parameter for SMOTE when enabled.",
    )
    train_parser.add_argument(
        "--disable-scaler",
        action="store_true",
        help="Disable StandardScaler preprocessing.",
    )
    train_parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Skip rows with missing/broken feature files instead of failing early.",
    )
    add_record_selection_arguments(train_parser)
    train_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible training transforms and models.",
    )
    train_parser.set_defaults(handler=_train_classifier_command)

    ablation_parser = subparsers.add_parser(
        "run-ablations",
        help="Run default feature/reduction/SMOTE ablation suite.",
    )
    ablation_parser.add_argument(
        "--index-path",
        default="outputs/index/dataset_index.jsonl",
        help="Path to index JSONL generated by build-index.",
    )
    ablation_parser.add_argument(
        "--openface-cache-dir",
        default="outputs/features/openface_cache",
        help="OpenFace feature cache root.",
    )
    ablation_parser.add_argument(
        "--cnn-cache-dir",
        default="outputs/features/cnn_cache",
        help="CNN feature cache root.",
    )
    ablation_parser.add_argument(
        "--fused-cache-dir",
        default="outputs/features/fused_cache",
        help="Fused feature cache root.",
    )
    ablation_parser.add_argument(
        "--output-dir",
        default="outputs/experiments",
        help="Output directory for ablation artifacts.",
    )
    ablation_parser.add_argument(
        "--pooling-mode",
        choices=list(POOLING_MODES),
        default="mean",
        help="Clip-level pooling strategy for frame features.",
    )
    ablation_parser.add_argument(
        "--classifier-name",
        choices=list(CLASSIFIER_CHOICES),
        default="logistic_regression",
        help="Classifier used across ablation runs.",
    )
    ablation_parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy passed to supported classifiers.",
    )
    ablation_parser.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="Requested PCA/SVD components for reduction ablations.",
    )
    ablation_parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="k_neighbors parameter for SMOTE-enabled ablations.",
    )
    ablation_parser.add_argument(
        "--disable-scaler",
        action="store_true",
        help="Disable StandardScaler preprocessing.",
    )
    ablation_parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Skip rows with missing/broken feature files instead of failing early.",
    )
    add_record_selection_arguments(ablation_parser)
    ablation_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible ablation runs.",
    )
    ablation_parser.set_defaults(handler=_run_ablations_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
