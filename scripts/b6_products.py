#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]

PROVENANCE_PATH = LAB / "provenance" / "b6_products.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_idempotent(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == text:
            return
        raise SystemExit(
            "")
    path.write_text(text, encoding="utf-8")


class ProductEmitter:
    def __init__(self, tree_id: str, *, lab_stamp: str,
                 provenance_path: Path = PROVENANCE_PATH) -> None:
        self.tree_id = tree_id
        self.lab_stamp = lab_stamp
        self.provenance_path = provenance_path
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, path: Path, text: str) -> Path:
        path = Path(path)
        _write_idempotent(path, text)
        try:
            rel = str(path.relative_to(LAB))
        except ValueError:
            rel = str(path)
        self.records[rel] = {"path": rel, "sha256": sha256_text(text),
                             "bytes": len(text.encode("utf-8")),
                             "lab_stamp": self.lab_stamp}
        return path

    def write_json(self, path: Path, obj: Any) -> Path:
        return self.write(path, json.dumps(obj, ensure_ascii=False, indent=2,
                                           sort_keys=True) + "\n")

    def write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> Path:
        body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        return self.write(path, body)

    def payload(self) -> dict[str, Any]:
        return {"tree_id": self.tree_id, "lab_stamp": self.lab_stamp,
                "n_products": len(self.records),
                "products": [self.records[k] for k in sorted(self.records)]}

    def emit(self) -> Path:
        if not self.records:
            raise SystemExit("")
        prev: dict[str, Any] = {}
        if self.provenance_path.exists():
            prev = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        merged = dict(prev.get("trees") or {})
        old_products = {r["path"]: r for r in (merged.get(self.tree_id) or {}).get("products", [])}
        old_products.update(self.records)
        products = [old_products[k] for k in sorted(old_products)]
        merged[self.tree_id] = {**self.payload(),
                                "n_products": len(products),
                                "lab_stamps": sorted({r.get("lab_stamp", "") for r in products}),
                                "products": products}
        text = json.dumps({"schema": "b6_products/v1",
                           "trees": {k: merged[k] for k in sorted(merged)}},
                          ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        self.provenance_path.write_text(text, encoding="utf-8")
        return self.provenance_path
