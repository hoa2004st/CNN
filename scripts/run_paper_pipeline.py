from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from engagement_pipeline.cnn import CNNExtractionConfig, extract_cnn_features_for_records
from engagement_pipeline.data_index import (
    SPLIT_ORDER,
    build_full_index,
    filter_records,
    write_json,
    write_records_jsonl,
)
from engagement_pipeline.experiments import default_ablation_specs, run_ablation_suite
from engagement_pipeline.fusion import FeatureFusionConfig, fuse_features_for_records
from engagement_pipeline.openface import (
    DEFAULT_FEATURE_FLAGS,
    OpenFaceExtractionConfig,
    extract_openface_features_for_records,
    write_manifest_jsonl,
)
from engagement_pipeline.training import TrainingConfig, train_classifier_from_feature_cache


def _run_build_index(daisee_root: Path, index_dir: Path, allow_missing_paths: bool) -> Path:
    records_by_split, diagnostics_by_split, leakage_report = build_full_index(
        daisee_root=daisee_root,
        strict_paths=not allow_missing_paths,
    )
    ordered_records = [
        record
        for split in SPLIT_ORDER
        for record in records_by_split.get(split, [])
    ]

    index_path = index_dir / "dataset_index.jsonl"
    summary_path = index_dir / "index_summary.json"
    write_records_jsonl(records=ordered_records, output_path=index_path)
    write_json(
        data={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "daisee_root": str(daisee_root),
            "strict_paths": not allow_missing_paths,
            "split_counts": {
                split: len(records_by_split.get(split, [])) for split in SPLIT_ORDER
            },
            "diagnostics": {
                split: diagnostics.to_dict()
                for split, diagnostics in diagnostics_by_split.items()
            },
            "subject_leakage": leakage_report,
        },
        output_path=summary_path,
    )
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot runner for the DAiSEE paper reproduction pipeline."
    )
    parser.add_argument("--daisee-root", required=True, help="Path to DAiSEE dataset root.")
    parser.add_argument(
        "--output-root",
        default="outputs/paper_run",
        help="Root directory for all generated artifacts.",
    )
    parser.add_argument(
        "--feature-root",
        default="",
        help=(
            "Optional root directory for extracted feature caches. "
            "Defaults to <output-root>/features so caches live in one tracked folder."
        ),
    )
    parser.add_argument(
        "--openface-bin",
        default="FeatureExtraction",
        help="OpenFace FeatureExtraction executable path or command name.",
    )
    parser.add_argument(
        "--skip-openface",
        action="store_true",
        help="Skip OpenFace extraction if cache already exists or you only want CNN-only runs.",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN extraction if cache already exists.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip feature fusion if fused cache already exists.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the main fused training run.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip the default ablation suite.",
    )
    parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        help="Keep index rows even if video paths are missing.",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Continue training and ablations when some caches are missing.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Restrict the run to one or more splits (repeatable).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Optional global cap on selected records after split filtering.",
    )
    parser.add_argument(
        "--max-clips-per-split",
        type=int,
        default=0,
        help="Optional per-split cap to use for smoke runs or sharded execution.",
    )
    parser.add_argument("--cnn-model-name", default="efficientnet_b0")
    parser.add_argument("--cnn-weights", default="DEFAULT")
    parser.add_argument("--cnn-device", default="auto")
    parser.add_argument("--cnn-image-size", type=int, default=224)
    parser.add_argument("--cnn-num-samples", type=int, default=60)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--no-cnn-pretrained", action="store_true")
    parser.add_argument(
        "--pooling-mode",
        default="mean",
        help="Clip-level pooling mode for training and ablations.",
    )
    parser.add_argument(
        "--reduction-method",
        default="pca",
        help="Reduction method for the main fused training run.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="Requested PCA/SVD components for training and ablations.",
    )
    parser.add_argument(
        "--classifier-name",
        default="logistic_regression",
        help="Classifier name for training and ablations.",
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        help="Class weighting strategy for supported classifiers.",
    )
    parser.add_argument(
        "--enable-smote",
        action="store_true",
        help="Enable SMOTE for the main fused training run.",
    )
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--disable-scaler", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    daisee_root = Path(args.daisee_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    index_dir = output_root / "index"
    feature_root = (
        Path(args.feature_root).expanduser().resolve()
        if args.feature_root
        else output_root / "features"
    )
    openface_cache_dir = feature_root / "openface_cache"
    cnn_cache_dir = feature_root / "cnn_cache"
    fused_cache_dir = feature_root / "fused_cache"
    training_dir = output_root / "training"
    experiments_dir = output_root / "experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    index_path = _run_build_index(
        daisee_root=daisee_root,
        index_dir=index_dir,
        allow_missing_paths=args.allow_missing_paths,
    )

    from engagement_pipeline.data_index import read_records_jsonl

    records = filter_records(
        records=read_records_jsonl(index_path=index_path),
        splits=args.split,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        max_clips_per_split=args.max_clips_per_split if args.max_clips_per_split > 0 else None,
    )

    run_summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "daisee_root": str(daisee_root),
        "output_root": str(output_root),
        "feature_root": str(feature_root),
        "selection": {
            "splits": args.split,
            "max_clips": args.max_clips,
            "max_clips_per_split": args.max_clips_per_split,
            "selected_records": len(records),
        },
        "steps": {},
    }

    if not args.skip_openface:
        openface_config = OpenFaceExtractionConfig(
            executable=args.openface_bin,
            feature_flags=DEFAULT_FEATURE_FLAGS,
        )
        manifest_rows, summary = extract_openface_features_for_records(
            records=records,
            cache_root=openface_cache_dir,
            config=openface_config,
            overwrite=args.overwrite,
        )
        write_manifest_jsonl(
            rows=manifest_rows,
            output_path=openface_cache_dir / "extraction_manifest.jsonl",
        )
        write_json(data=summary, output_path=openface_cache_dir / "extraction_summary.json")
        run_summary["steps"]["openface"] = summary

    if not args.skip_cnn:
        cnn_config = CNNExtractionConfig(
            model_name=args.cnn_model_name,
            pretrained=not args.no_cnn_pretrained,
            weights=args.cnn_weights,
            image_size=args.cnn_image_size,
            num_samples=args.cnn_num_samples,
            batch_size=args.cnn_batch_size,
            device=args.cnn_device,
        )
        manifest_rows, summary = extract_cnn_features_for_records(
            records=records,
            cache_root=cnn_cache_dir,
            config=cnn_config,
            overwrite=args.overwrite,
        )
        write_manifest_jsonl(
            rows=manifest_rows,
            output_path=cnn_cache_dir / "extraction_manifest.jsonl",
        )
        write_json(data=summary, output_path=cnn_cache_dir / "extraction_summary.json")
        run_summary["steps"]["cnn"] = summary

    if not args.skip_fusion:
        fusion_config = FeatureFusionConfig(alignment_mode="truncate", fusion_method="concat")
        manifest_rows, summary = fuse_features_for_records(
            records=records,
            openface_cache_root=openface_cache_dir,
            cnn_cache_root=cnn_cache_dir,
            fused_cache_root=fused_cache_dir,
            config=fusion_config,
            overwrite=args.overwrite,
        )
        write_manifest_jsonl(
            rows=manifest_rows,
            output_path=fused_cache_dir / "fusion_manifest.jsonl",
        )
        write_json(data=summary, output_path=fused_cache_dir / "fusion_summary.json")
        run_summary["steps"]["fusion"] = summary

    if not args.skip_training:
        _, summary = train_classifier_from_feature_cache(
            records=records,
            feature_cache_root=fused_cache_dir,
            output_dir=training_dir,
            config=TrainingConfig(
                pooling_mode=args.pooling_mode,
                reduction_method=args.reduction_method,
                n_components=args.n_components,
                use_smote=args.enable_smote,
                smote_k_neighbors=args.smote_k_neighbors,
                classifier_name=args.classifier_name,
                class_weight=args.class_weight,
                use_scaler=not args.disable_scaler,
                random_seed=args.random_seed,
            ),
            strict_features=not args.allow_missing_features,
        )
        run_summary["steps"]["training"] = summary

    if not args.skip_ablations:
        _, summary = run_ablation_suite(
            records=records,
            output_root=experiments_dir,
            specs=default_ablation_specs(
                openface_cache_root=openface_cache_dir,
                cnn_cache_root=cnn_cache_dir,
                fused_cache_root=fused_cache_dir,
            ),
            pooling_mode=args.pooling_mode,
            classifier_name=args.classifier_name,
            class_weight=args.class_weight,
            n_components=args.n_components,
            smote_k_neighbors=args.smote_k_neighbors,
            use_scaler=not args.disable_scaler,
            random_seed=args.random_seed,
            strict_features=not args.allow_missing_features,
        )
        run_summary["steps"]["ablations"] = summary

    write_json(data=run_summary, output_path=output_root / "run_summary.json")
    print(json.dumps(run_summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
