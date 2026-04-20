from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_selected_files(
    source_root: Path,
    destination_root: Path,
    patterns: Iterable[str],
) -> None:
    for pattern in patterns:
        for source_path in source_root.rglob(pattern):
            if source_path.is_file():
                relative_path = source_path.relative_to(source_root)
                _copy_file(source_path, destination_root / relative_path)


def export_reusable_artifacts(
    *,
    output_root: Path,
    feature_root: Path,
    export_root: Path,
) -> Path:
    output_root = output_root.expanduser().resolve()
    feature_root = feature_root.expanduser().resolve()
    export_root = export_root.expanduser().resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    for summary_name in ("run_summary.json",):
        source_path = output_root / summary_name
        if source_path.exists():
            _copy_file(source_path, export_root / summary_name)

    index_root = output_root / "index"
    if index_root.exists():
        _copy_selected_files(
            source_root=index_root,
            destination_root=export_root / "index",
            patterns=("*.json", "*.jsonl"),
        )

    if feature_root.exists():
        _copy_selected_files(
            source_root=feature_root,
            destination_root=export_root / "features",
            patterns=("features.npz", "meta.json", "*summary.json"),
        )

    for stage_dir_name in ("training", "experiments"):
        stage_root = output_root / stage_dir_name
        if stage_root.exists():
            _copy_selected_files(
                source_root=stage_root,
                destination_root=export_root / stage_dir_name,
                patterns=("*.json", "*.csv", "*.joblib", "*.png"),
            )

    return export_root
