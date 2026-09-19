#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

def _force_utf8_stdio():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            buf = getattr(stream, "buffer", None)
            if buf is not None:
                setattr(sys, name, io.TextIOWrapper(
                    buf, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass


_force_utf8_stdio()

DEFAULT_LAB = Path(__file__).resolve().parents[1]
PROVENANCE_JSON_RELPATH = "provenance/lab_code_provenance.json"
SCHEMA = "dar-lab-code-provenance/1"
EXCLUDE_PARTS = {"__pycache__", ".DS_Store", ".git"}

SCOPE_DECLARATION = [
    "src/dar/**/*.py __pycache__/.DS_Store/.git",
    "scripts/*.pyaudit_probes_*/ /",
    "hpc/{b5_matrix,b6_llm_diagnose,bd8_ablation}.sbatch + hpc/preflight_lab_stamp.sh"
    "§8-4  .py ",
]

HPC_SCOPE_FILES = ("hpc/b5_matrix.sbatch", "hpc/b6_llm_diagnose.sbatch",
                   "hpc/bd8_ablation.sbatch", "hpc/preflight_lab_stamp.sh")


def _refuse(msg):
    print("[stamp-lab] REFUSING: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def iter_scope(lab_root):
    lab_root = Path(lab_root)
    src = lab_root / "src" / "dar"
    scripts = lab_root / "scripts"
    if not src.is_dir():
        _refuse("")
    if not scripts.is_dir():
        _refuse("")

    for root_dir in (src, scripts):
        for dirpath, dirnames, _ in os.walk(str(root_dir), followlinks=False):
            for d in dirnames:
                dp = Path(dirpath) / d
                if set(dp.parts) & EXCLUDE_PARTS:
                    continue
                if dp.is_symlink():
                    _refuse("")

    src_files = [p for p in src.rglob("*.py") if not (set(p.parts) & EXCLUDE_PARTS)]
    script_files = list(scripts.glob("*.py"))
    if not src_files:
        _refuse("")
    if not script_files:
        _refuse("")
    hpc_files = []
    for rel in HPC_SCOPE_FILES:
        p = lab_root / rel
        if not p.is_file():
            _refuse("")
        hpc_files.append(p)

    pairs = []
    for p in src_files + script_files + hpc_files:
        if p.is_symlink():
            _refuse("")
        pairs.append((p.relative_to(lab_root).as_posix(), p))
    pairs.sort(key=lambda t: t[0])
    return pairs


def hash_scope(lab_root):
    out = {}
    for rel, p in iter_scope(lab_root):
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def aggregate(files):
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(("%s\n%s\n" % (rel, files[rel])).encode())
    return h.hexdigest()


def compare_maps(expected, actual):
    return {
        "missing": sorted(set(expected) - set(actual)),
        "extra": sorted(set(actual) - set(expected)),
        "changed": sorted(r for r in set(expected) & set(actual)
                          if expected[r] != actual[r]),
    }


def bundle_bytes(lab_root):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, p in iter_scope(lab_root):
            data = p.read_bytes()
            info = tarfile.TarInfo(name=rel)
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
    return gz.getvalue()


def bundle_member_hashes(blob):
    out = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for info in tar.getmembers():
            if not info.isfile():
                continue
            handle = tar.extractfile(info)
            payload = handle.read() if handle is not None else b""
            out[info.name] = hashlib.sha256(payload).hexdigest()
    return out


def _load_stamp(lab_root):
    stamp_path = Path(lab_root) / PROVENANCE_JSON_RELPATH
    if not stamp_path.is_file():
        _refuse("")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    if stamp.get("schema") != SCHEMA:
        _refuse("")
    if not stamp.get("files"):
        _refuse("")
    return stamp


def do_stamp(lab_root):
    files = hash_scope(lab_root)
    fp = aggregate(files)
    bundle_rel = "provenance/bundles/lab_code_%s.tar.gz" % fp[:12]
    bundle_path = Path(lab_root) / bundle_rel
    blob = bundle_bytes(lab_root)

    if bundle_path.exists():
        existing = bundle_path.read_bytes()
        if existing != blob:
            diff = compare_maps(bundle_member_hashes(existing), bundle_member_hashes(blob))
            _refuse("")
    else:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(blob)

    bundle_sha = hashlib.sha256(blob).hexdigest()
    stamp = {
        "schema": SCHEMA,
        "scope": SCOPE_DECLARATION,
        "fingerprint_sha256": fp,
        "file_count": len(files),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle": bundle_rel,
        "bundle_sha256": bundle_sha,
        "files": files,
    }
    stamp_path = Path(lab_root) / PROVENANCE_JSON_RELPATH
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    print("[stamp-lab] bundle=%s sha256=%s" % (bundle_rel, bundle_sha))
    print("[stamp-lab] json → %s" % stamp_path)
    return 0


def do_check(lab_root):
    stamp = _load_stamp(lab_root)
    if aggregate(stamp["files"]) != stamp["fingerprint_sha256"]:
        return 1
    diff = compare_maps(stamp["files"], hash_scope(lab_root))
    if any(diff.values()):
        return 1
    return 0


def do_verify_bundle(lab_root):
    stamp = _load_stamp(lab_root)
    bundle_path = Path(lab_root) / stamp["bundle"]
    if not bundle_path.is_file():
        _refuse("")
    blob = bundle_path.read_bytes()
    actual_sha = hashlib.sha256(blob).hexdigest()
    if actual_sha != stamp["bundle_sha256"]:
        return 1
    members = bundle_member_hashes(blob)
    diff = compare_maps(stamp["files"], members)
    if any(diff.values()):
        return 1
    if aggregate(members) != stamp["fingerprint_sha256"]:
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-bundle", action="store_true")
    ap.add_argument("--lab-root", default=str(DEFAULT_LAB))
    args = ap.parse_args(argv)

    if args.check and args.verify_bundle:
        _refuse("")
    if args.check:
        return do_check(args.lab_root)
    if args.verify_bundle:
        return do_verify_bundle(args.lab_root)
    return do_stamp(args.lab_root)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        sys.exit(3)
