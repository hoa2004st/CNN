from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SPLIT_ORDER = ("train", "validation", "test")
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


@dataclass(frozen=True)
class ClipRecord:
    split: str
    clip_id: str
    subject_id: str
    clip_stem: str
    clip_path: str
    engagement: int


@dataclass
class SplitDiagnostics:
    split: str
    rows_in_labels: int = 0
    rows_indexed: int = 0
    invalid_rows: List[str] = field(default_factory=list)
    missing_clip_paths: List[str] = field(default_factory=list)
    missing_in_manifest: List[str] = field(default_factory=list)
    missing_in_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SPLIT_TO_DIR:
        allowed = ", ".join(SPLIT_TO_DIR)
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {allowed}.")
    return normalized


def _get_row_field(row: Dict[str, str], target_field: str) -> str:
    target = target_field.strip().lower()
    for key, value in row.items():
        if key and key.strip().lower() == target:
            return "" if value is None else str(value).strip()
    return ""


def load_split_manifest(daisee_root: Path, split: str) -> List[str]:
    split_name = normalize_split(split)
    manifest_path = daisee_root / "DataSet" / SPLIT_TO_MANIFEST[split_name]
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        return [line.strip() for line in manifest_file if line.strip()]


def resolve_clip_path(daisee_root: Path, split: str, clip_id: str) -> Path | None:
    split_name = normalize_split(split)
    split_dir = daisee_root / "DataSet" / SPLIT_TO_DIR[split_name]

    clip_stem = Path(clip_id).stem
    subject_id = clip_stem[:6]

    canonical_path = split_dir / subject_id / clip_stem / clip_id
    if canonical_path.exists():
        return canonical_path

    clip_dir = split_dir / subject_id / clip_stem
    if clip_dir.exists():
        candidates = sorted(clip_dir.glob(f"{clip_stem}.*"))
        for candidate in candidates:
            if candidate.name.lower() == clip_id.lower():
                return candidate
        if candidates:
            return candidates[0]

    return None


def build_split_index(
    daisee_root: Path,
    split: str,
    target_label: str = "Engagement",
    strict_paths: bool = True,
) -> Tuple[List[ClipRecord], SplitDiagnostics]:
    split_name = normalize_split(split)
    labels_path = daisee_root / "Labels" / SPLIT_TO_LABELS[split_name]
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing label file: {labels_path}")

    diagnostics = SplitDiagnostics(split=split_name)
    records: List[ClipRecord] = []
    clip_ids_in_labels: List[str] = []

    with labels_path.open("r", encoding="utf-8", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        for row_number, row in enumerate(reader, start=2):
            diagnostics.rows_in_labels += 1

            clip_id = _get_row_field(row, "ClipID")
            if not clip_id:
                diagnostics.invalid_rows.append(f"line {row_number}: missing ClipID")
                continue

            clip_ids_in_labels.append(clip_id)
            target_raw = _get_row_field(row, target_label)
            if target_raw == "":
                diagnostics.invalid_rows.append(
                    f"line {row_number}: missing {target_label} for clip {clip_id}"
                )
                continue

            try:
                target_value = int(target_raw)
            except ValueError:
                diagnostics.invalid_rows.append(
                    f"line {row_number}: invalid {target_label} value '{target_raw}' for clip {clip_id}"
                )
                continue

            clip_path = resolve_clip_path(daisee_root, split_name, clip_id)
            if clip_path is None:
                diagnostics.missing_clip_paths.append(clip_id)
                if strict_paths:
                    continue

            clip_stem = Path(clip_id).stem
            subject_id = clip_stem[:6]
            records.append(
                ClipRecord(
                    split=split_name,
                    clip_id=clip_id,
                    subject_id=subject_id,
                    clip_stem=clip_stem,
                    clip_path="" if clip_path is None else str(clip_path),
                    engagement=target_value,
                )
            )

    manifest_clip_ids = set(load_split_manifest(daisee_root, split_name))
    label_clip_id_set = set(clip_ids_in_labels)

    if manifest_clip_ids:
        diagnostics.missing_in_manifest = sorted(label_clip_id_set - manifest_clip_ids)
        diagnostics.missing_in_labels = sorted(manifest_clip_ids - label_clip_id_set)

    diagnostics.rows_indexed = len(records)
    return records, diagnostics


def validate_subject_leakage(records_by_split: Dict[str, Sequence[ClipRecord]]) -> Dict[str, List[str]]:
    subjects_by_split: Dict[str, set[str]] = {
        split: {record.subject_id for record in records}
        for split, records in records_by_split.items()
    }

    overlaps: Dict[str, List[str]] = {}
    for split_a, split_b in combinations(SPLIT_ORDER, 2):
        overlap = subjects_by_split.get(split_a, set()) & subjects_by_split.get(split_b, set())
        if overlap:
            overlaps[f"{split_a}_vs_{split_b}"] = sorted(overlap)

    return overlaps


def build_full_index(
    daisee_root: Path,
    target_label: str = "Engagement",
    strict_paths: bool = True,
) -> Tuple[Dict[str, List[ClipRecord]], Dict[str, SplitDiagnostics], Dict[str, List[str]]]:
    records_by_split: Dict[str, List[ClipRecord]] = {}
    diagnostics_by_split: Dict[str, SplitDiagnostics] = {}

    for split in SPLIT_ORDER:
        records, diagnostics = build_split_index(
            daisee_root=daisee_root,
            split=split,
            target_label=target_label,
            strict_paths=strict_paths,
        )
        records_by_split[split] = records
        diagnostics_by_split[split] = diagnostics

    leakage_report = validate_subject_leakage(records_by_split)
    return records_by_split, diagnostics_by_split, leakage_report


def write_records_jsonl(records: Sequence[ClipRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def write_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=True)


def read_records_jsonl(index_path: Path) -> List[ClipRecord]:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    records: List[ClipRecord] = []
    with index_path.open("r", encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row at line {line_number} in {index_path}"
                ) from exc

            try:
                records.append(
                    ClipRecord(
                        split=str(row["split"]),
                        clip_id=str(row["clip_id"]),
                        subject_id=str(row["subject_id"]),
                        clip_stem=str(row["clip_stem"]),
                        clip_path=str(row.get("clip_path", "")),
                        engagement=int(row["engagement"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid index schema at line {line_number} in {index_path}"
                ) from exc

    return records
