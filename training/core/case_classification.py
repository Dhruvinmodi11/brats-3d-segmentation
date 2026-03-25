"""
Case-level classification labels (e.g. HGG vs LGG) keyed by BraTS case ID.

CSV format (header required):
  case_id,class_id
  BraTS-GLI-00006-000,0
  BraTS20_Training_001,1

- class_id: integer in [0, num_classes-1]. Use -1 in CSV for unknown (optional);
  patches for unlisted case_ids get label -1 and are ignored in cls loss.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


def case_id_from_patch_stem(stem: str) -> str:
    """e.g. BraTS-GLI-00006-000_patch_0001 -> BraTS-GLI-00006-000"""
    m = re.match(r"(.+)_patch_\d+", stem)
    return m.group(1) if m else stem


def load_case_class_csv(
    path: Path | str,
    *,
    id_column: str = "case_id",
    label_column: str = "class_id",
) -> dict[str, int]:
    """
    Load mapping case_id -> integer class index (or -1 for ignore).

    Accepts UTF-8 CSV with header. Extra columns are ignored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Classification CSV not found: {path}")

    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        fields = {h.strip() for h in reader.fieldnames}
        if id_column not in fields or label_column not in fields:
            raise ValueError(
                f"CSV {path} must contain columns {id_column!r} and {label_column!r}; "
                f"got {reader.fieldnames!r}"
            )
        for row in reader:
            cid = (row.get(id_column) or "").strip()
            if not cid:
                continue
            raw = (row.get(label_column) or "").strip()
            if raw == "" or raw.lower() in ("na", "n/a", "none"):
                out[cid] = -1
                continue
            try:
                v = int(raw)
            except ValueError as e:
                raise ValueError(f"Bad {label_column}={raw!r} for case_id={cid!r}") from e
            out[cid] = v
    return out
