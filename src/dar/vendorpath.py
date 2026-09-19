from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "toolmaze"


def add_vendor_to_syspath() -> Path:
    if not VENDOR_ROOT.is_dir():
        raise FileNotFoundError(
            "")
    p = str(VENDOR_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return VENDOR_ROOT
