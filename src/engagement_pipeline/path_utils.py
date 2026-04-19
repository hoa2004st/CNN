from __future__ import annotations

import os
from pathlib import Path


def resolve_user_path(raw_path: str | Path) -> Path:
    if isinstance(raw_path, Path):
        candidate = raw_path
    else:
        normalized = str(raw_path).strip()
        if os.sep == "/" and "\\" in normalized:
            normalized = normalized.replace("\\", "/")
        candidate = Path(normalized)
    return candidate.expanduser().resolve()
