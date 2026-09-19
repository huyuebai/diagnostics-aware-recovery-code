#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
VENDOR = LAB / "vendor" / "toolmaze"
PROVENANCE = LAB / "provenance"
BUNDLES = PROVENANCE / "bundles"
CODE_MANIFEST = PROVENANCE / "toolmaze_manifest.json"
DATASET_MANIFEST = PROVENANCE / "toolmaze_dataset_manifest.json"
DATASET_DEFAULT = Path("/Users/bai/Code/dissertation_tool/ToolMaze_dataset")

EXCLUDE_PARTS = {"__pycache__", ".DS_Store", ".git"}


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if set(p.parts) & EXCLUDE_PARTS:
            continue
        if p.is_symlink():
            raise SystemExit("")
        if p.is_file():
            yield p


def hash_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in _iter_files(root):
        out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def aggregate(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\n{files[rel]}\n".encode())
    return h.hexdigest()


def manifest_self_check(manifest: dict, path: Path) -> str | None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return (f"[stamp] REFUSING: {path}  files / dict/——"
                " manifest  = "
                "python lab/scripts/stamp_vendor.py")
    declared, actual = manifest.get("aggregate_sha256"), aggregate(files)
    if declared != actual:
        return (f"[stamp] REFUSING: {path} —— aggregate_sha256={declared!r}"
                f" files ={actual!r}/ "
                "run_dispatch._read_aggregate  config_snapshot  "
                "vendor_aggregate/dataset_aggregate ⇒ "
                " =  python lab/scripts/stamp_vendor.py  manifest")
    return None


def compare(manifest_files: dict[str, str], root: Path) -> dict[str, list[str]]:
    actual = hash_tree(root)
    return {
        "missing": sorted(set(manifest_files) - set(actual)),
        "extra": sorted(set(actual) - set(manifest_files)),
        "changed": sorted(r for r in set(manifest_files) & set(actual)
                          if manifest_files[r] != actual[r]),
    }


def write_manifest(path: Path, *, name: str, source: str, files: dict[str, str],
                   note: str) -> str:
    agg = aggregate(files)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "dar-vendor-manifest/1",
        "name": name,
        "source": source,
        "note": note,
        "aggregate_sha256": agg,
        "files": files,
    }, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return agg


def deterministic_bundle(root: Path, out_path: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in _iter_files(root):
            rel = p.relative_to(root).as_posix()
            info = tarfile.TarInfo(name=rel)
            data = p.read_bytes()
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o755 if (p.stat().st_mode & 0o111) else 0o644
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as f:
        f.write(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gz.getvalue())
    return hashlib.sha256(gz.getvalue()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    args = ap.parse_args()

    if not VENDOR.is_dir():
        return 2

    if args.check:
        manifest = json.loads(CODE_MANIFEST.read_text(encoding="utf-8"))
        if (why := manifest_self_check(manifest, CODE_MANIFEST)):
            print(why)
            return 1
        diff = compare(manifest["files"], VENDOR)
        if any(diff.values()):
            print(f"[stamp] DRIFT: {json.dumps(diff, ensure_ascii=False, indent=1)}")
            return 1
        ds = Path(args.dataset)
        if DATASET_MANIFEST.is_file():
            ds_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
            if (why := manifest_self_check(ds_manifest, DATASET_MANIFEST)):
                print(why)
                return 1
            if ds.is_dir():
                ds_diff = compare(ds_manifest["files"], ds)
                if any(ds_diff.values()):
                    print(f"[stamp] DATASET DRIFT: "
                          f"{json.dumps(ds_diff, ensure_ascii=False)[:2000]}")
                    return 1
            else:
                pass
        else:
            pass
        return 0

    code_files = hash_tree(VENDOR)
    code_agg = write_manifest(
        CODE_MANIFEST, name="toolmaze-code",
        source="/Users/bai/Code/dissertation_tool/ToolMaze-main git arXiv:2606.05806",
        files=code_files,
        note="vendor  gitignored#4 LICENSE  git "
             " =  aggregate  bundle  lab/vendor/toolmaze/")
    bundle = BUNDLES / f"toolmaze-code-{code_agg[:16]}.tar.gz"
    bundle_sha = deterministic_bundle(VENDOR, bundle)
    print(f"[stamp] code aggregate={code_agg}")
    print(f"[stamp] bundle={bundle.name} sha256={bundle_sha}")

    dataset = Path(args.dataset)
    if dataset.is_dir():
        data_files = hash_tree(dataset)
        data_agg = write_manifest(
            DATASET_MANIFEST, name="toolmaze-dataset",
            source=f"{dataset}HF dataset dongsheng/ToolMaze",
            files=data_files,
            note="HF manifest ")
        print(f"[stamp] dataset aggregate={data_agg}")
    else:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
