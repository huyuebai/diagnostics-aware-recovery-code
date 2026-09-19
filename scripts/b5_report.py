#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_items as I
import b5_judge as J
import b5_ledger as L
import matrix
import matrix_audit
import run_matrix
from dar import accounting as A
from dar import b5_stats as S
from dar import ga_disclosure as GD
from dar import headroom as H

B_BOOT = 10000
ALPHA_CI = 0.10
AXES = A.MATERIAL_AXES


def stage_gate0(data: Path, model: str) -> dict[str, Any]:
    I.assert_code_stamped()
    I.assert_corpus_identity_matches_local(data)
    man = sorted(data.glob(L.MANIFEST_GLOB))[0]
    res = matrix_audit.audit(data, man, model)
    if res["problems"]:
        raise SystemExit("")
    return {"counts": res["counts"], "notes": res.get("notes") or [],
            "manifest": man.name}


def stage_verify(data: Path) -> dict[str, Any]:
    items = J.load_b5_judge_items(data)
    vmap = J.load_verdicts_singlesource(
        data, expected_run_ids=sorted({i["run_id"] for i in items}),
        expected_items=J.items_universe(items))
    layer_a = {}
    reason = {}
    for line in (data / "ga_layerA.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            layer_a[r["run_id"]] = r
            reason[r["reason_code"]] = reason.get(r["reason_code"], 0) + 1
    eligible = {rid for rid, r in layer_a.items()
                if B4.is_item_eligible(r["verdict"], r.get("reason_code"))}
    if eligible != set(vmap):
        raise SystemExit(
            "")
    structural = {rid for rid, r in layer_a.items()
                  if r.get("reason_code") == "no_canonical_path"}
    if structural & set(vmap):
        raise SystemExit("")
    return {"n_runs_layer_a": len(layer_a), "n_judged": len(vmap),
            "reason_code_counts": dict(sorted(reason.items())),
            "n_structural_undecidable": len(structural),
            "structural_run_ids": sorted(structural),
            "_vmap": vmap, "_layer_a": layer_a, "_items": items}


def stage_join(data: Path, model: str, vmap: dict, layer_a: dict) -> list[dict[str, Any]]:
    from dar.b5_stats import run_verdict_from_item_map
    mat = B4._materiality(data)
    defects = B4._defects()
    rows = L.load_manifest_rows(data)
    ledger = {r["run_id"]: r for r in (
        json.loads(x) for x in
        (data / "analysis" / "ledger_b.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    out: list[dict[str, Any]] = []
    for row in rows:
        rid = row["run_id"]
        lb = ledger.get(rid)
        if lb is None:
            raise SystemExit("")
        im = vmap.get(rid)
        verdict = run_verdict_from_item_map(im) if im else layer_a[rid]["verdict"]
        if verdict != lb["ga_verdict"]:
            raise SystemExit("")
        base = {**lb, "arm": run_matrix.subtree_of(row), "verdict": verdict,
                "_row": row}
        axes = ([material_axis(row, a) for a in AXES] if row["truth"] == "NP"
                and row["label"] == "np" else [material_axis(row, None)])
        for axis in axes:
            e = defects.get(row["tid"])
            out.append({**base, "axis": axis,
                        "material": bool(mat[row["cat"]][row["tid"]][axis]["any_material"]),
                        "defect": A.defect_disposition_for_axis(e, axis)})
    return out


def material_axis(row: dict, pairing_axis: str | None) -> str:
    return A.material_axis_for_row(row, pairing_axis=pairing_axis)


def stage_pools(rows: list[dict], data: Path) -> dict[str, Any]:
    seen_rep = {r["run_id"] for r in rows if S.is_replicate_row(r["_row"])}
    man_rep = {r["run_id"] for r in L.load_manifest_rows(data) if S.is_replicate_row(r)}
    if seen_rep != man_rep:
        raise SystemExit(
            "")
    out: dict[str, Any] = {}
    for batch in ("main", "plain"):
        seen: set[str] = set()
        uniq = []
        for r in rows:
            if r["batch"] == batch and r["run_id"] not in seen:
                seen.add(r["run_id"])
                uniq.append(r)
        out[batch] = A.build_p0_pools(uniq, batch=batch)
    payload = {b: {k: (sorted("|".join(x) for x in v) if isinstance(v, (set, list)) else v)
                   for k, v in p.items()} for b, p in out.items()}
    (data / "analysis").mkdir(parents=True, exist_ok=True)
    _write_idempotent(data / "analysis" / "p0_pools.json",
                      json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return out


def _write_idempotent(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise SystemExit(
            "")
    path.write_text(text, encoding="utf-8")


def _ref_by_cat(rows: list[dict], pools: dict, unit, *, material: bool = True,
                cats: tuple[str, ...] | None = None) -> dict[str, set]:
    sieve = _sieve_for(rows, pools, unit.batch, unit.axis, material=material)
    return {c: {k for k in sieve if k[0] == c} for c in (cats or unit.cats)}


def _sieve_for(rows: list[dict], pools: dict, batch: str, axis: str,
               *, material: bool = True) -> set[tuple[str, str]]:
    sub = [r for r in rows if r["batch"] == batch and r["axis"] == axis]
    keys = {(r["cat"], r["tid"]) for r in sub}
    mat_by = {(r["cat"], r["tid"]): (r["material"] if material else not r["material"])
              for r in sub}
    dis_by = {(r["cat"], r["tid"]): r["defect"] for r in sub}
    return A.cell_level_sieve(keys, ga_p0_pool=pools[batch]["ga_p0_pool"],
                              material_by_task=mat_by, disposition_by_task=dis_by)


def stage_mu(rows: list[dict], pools: dict) -> list[dict[str, Any]]:
    out = []
    for batch in ("main", "plain"):
        cells = sorted({(r["truth"], r["label"]) for r in rows if r["batch"] == batch})
        for truth, label in cells:
            for axis in AXES:
                sieve = _sieve_for(rows, pools, batch, axis)
                sub = [r for r in rows
                       if r["batch"] == batch and r["truth"] == truth
                       and r["label"] == label and r["axis"] == axis
                       and (r["cat"], r["tid"]) in sieve
                       and not S.is_replicate_row(r["_row"])]
                if not sub:
                    continue
                by_cat: dict[str, list[dict]] = {}
                for r in sub:
                    by_cat.setdefault(r["cat"], []).append(r)
                out.append({
                    "batch": batch, "truth": truth, "label": label, "axis": axis,
                    "n": len(sub),
                    "mu_ga": sum(1 for r in sub if r["ga_achieved"]) / len(sub),
                    "mu_judge_pass": sum(1 for r in sub if r["judge_pass"]) / len(sub),
                    "n_by_cat": {c: len(v) for c, v in sorted(by_cat.items())},
                    "task_set_label": (f"batch={batch} / ={axis} / "
                                       f"cats={sorted(by_cat)}"),
                })
    return out


def stage_paired(rows: list[dict], pools: dict, q: float,
                 vmap: dict | None = None) -> list[dict[str, Any]]:
    out = []
    for u in S.B5_PAIRED_UNITS:
        sub = [r for r in rows if r["batch"] == u.batch and r["axis"] == u.axis
               and not S.is_replicate_row(r["_row"])]
        ref_by_cat = _ref_by_cat(rows, pools, u)
        diffs, meta = S.unit_paired_diffs(u, sub, ref_by_cat)
        ci = H.bootstrap_ci([d["diff"] for d in diffs], ALPHA_CI, B_BOOT,
                            S.B5_BOOTSTRAP_SEED)
        two = None
        if vmap is not None:
            items = S.unit_paired_item_maps(u, sub, vmap, ref_by_cat)
            if items:
                two = S.two_layer_ci_items(items, q, B=B_BOOT,
                                           seed=S.B5_BOOTSTRAP_SEED)
        degenerate = None
        for slot in ("none", "oracle"):
            vals = {d[slot] for d in diffs}
            if diffs and vals == {True}:
                degenerate = (f"{slot} {getattr(u, slot + '_arm')}** achieved**"
                              f"⇒  ≡ ±(1 − ) "
                              f"P0-GA=achieved outcome [LM13] "
                              f"")
        out.append({**meta, "ci": ci, "two_layer": two, "diffs": diffs, "q": q,
                    "degenerate_arm": degenerate})
    return out


def stage_holm(paired: list[dict]) -> dict[str, Any]:
    est = [{"name": p["unit"], "diffs_by_task": [d["diff"] for d in p["diffs"]]}
           for p in paired if p["tier"] == "confirmatory"]
    return S.holm_report(est, B=B_BOOT, seed=S.B5_BOOTSTRAP_SEED)


def stage_negctl(rows: list[dict], pools: dict) -> dict[str, Any]:
    rep_rows = [r for r in rows if r["batch"] == "rep" and r["axis"] == AXES[0]]
    orig = [r for r in rows if r["batch"] == "main" and r["truth"] == "NP"
            and r["label"] == "np" and r["axis"] == AXES[0]
            and not S.is_replicate_row(r["_row"])]
    floor = S.replicate_noise_floor([{**r, **r["_row"]} for r in rep_rows],
                                    [{**r, **r["_row"]} for r in orig])
    cells = []
    for u in S.B5_PAIRED_UNITS:
        if u.batch != "main" or u.mode not in H.AXIS_OF:
            continue
        for cat in u.cats:
            sub = [r for r in rows if r["batch"] == u.batch and r["axis"] == u.axis
                   and not S.is_replicate_row(r["_row"])]
            one = S.B5PairedUnit(u.name, u.none_arm, u.oracle_arm, u.mode, u.batch,
                                 u.axis, (cat,), u.tier)
            d, _m = S.unit_paired_diffs(
                one, sub, _ref_by_cat(rows, pools, one, material=False))
            if d:
                cells.append({"name": f"{u.name}×{cat}", "cat": cat,
                              "diffs": [x["diff"] for x in d]})
    calib = (S.calibration_by_cell(floor, cells, B=B_BOOT, seed=S.B5_BOOTSTRAP_SEED)
             if cells else None)
    verd = S.negctl_verdicts(cells, floor, alpha=ALPHA_CI, B=B_BOOT,
                             seed=S.B5_BOOTSTRAP_SEED, calibration=calib)
    return {"floor": floor, "calibration": calib, "verdicts": verd}


def stage_qc(data: Path, rows: list[dict], pools: dict, vmap: dict,
             items: list[dict]) -> dict[str, Any]:
    _stg_paths = sorted(data.glob("staging_manifest.chunk*of*.json"))
    if not _stg_paths:
        raise SystemExit("")
    stg = json.loads(_stg_paths[0].read_text(encoding="utf-8"))["runs"]
    for _p in _stg_paths[1:]:
        if json.loads(_p.read_text(encoding="utf-8"))["runs"] != stg:
            raise SystemExit(
                "")
    ids = S.staging_task_ids(stg)
    pos = H.position_index(ids)
    pairs, signs = [], {}
    for u in S.B5_PAIRED_UNITS:
        if u.batch != "main":
            continue
        sub = [r for r in rows if r["batch"] == u.batch and r["axis"] == u.axis
               and not S.is_replicate_row(r["_row"])]
        ref_by_cat = _ref_by_cat(rows, pools, u)
        d, _ = S.unit_paired_diffs(u, sub, ref_by_cat)
        sgn = S.injected_minus_np_sign(u)
        pairs += [{"cell": u.name, "cat": x["cat"], "tid": x["tid"],
                   "diff": sgn * x["diff"]} for x in d]
        signs[u.name] = sgn
    n0 = S.position_drift_regression_b5(pairs, pos, list(matrix.CATS))
    n0 = {"per_cell": {k: {"full": v["full"]} for k, v in n0["per_cell"].items()},
          "pooled": {"full": n0["pooled"]["full"]},
          "direction": " − np §5-7  sign_by_unit",
          "sign_by_unit": signs,
          "warmup_note": ("C4  N_CHUNKS=4 cat  server"
                          "** cat ** ⇒  cat  warmup "
                          "`excl_warmup` ** full **"
                          "2026-08-11  A-6 =  §7quinquies ⑭-D")}
    return {"n0": n0, "judge_q": J.recon_channel_report(data, vmap, items=items)}


def stage_disclosure(rows: list[dict], pools: dict | None = None,
                     items: list[dict] | None = None) -> dict[str, Any]:
    uniq = {r["run_id"]: r for r in rows}.values()
    c_counts = GD.c_counts_by_arm_cell([{**r, "arm": r["arm"], "cat": r["cat"],
                                         "mode": r["mode"], "verdict": r["verdict"]}
                                        for r in uniq])
    flagged = []
    n_skipped_empty_arm = 0
    n_undefined = 0
    for u in S.B5_PAIRED_UNITS:
        for cat in u.cats:
            o = [r["verdict"] for r in uniq if r["arm"] == u.oracle_arm
                 and r["cat"] == cat and r["mode"] == u.mode and r["batch"] == u.batch]
            n = [r["verdict"] for r in uniq if r["arm"] == u.none_arm
                 and r["cat"] == cat and r["mode"] == u.mode and r["batch"] == u.batch]
            if not (o and n):
                n_skipped_empty_arm += 1
                continue
            mb = GD.manski_bounds(o, n)
            if mb.get("flagged") is None:
                n_undefined += 1
            elif mb["flagged"]:
                flagged.append({"cell": f"{u.name}×{cat}", **mb})
    out = {"c_counts_by_arm_cell": {"|".join(map(str, k)): v
                                    for k, v in sorted(c_counts.items())},
           "manski_flagged": flagged, "n_manski_flagged": len(flagged),
           "population": (" P0-GA  n=100 —— "
                          "`ga_semantic_judge.md` (C) ×"
                          "2026-08-11  A-6 "),
           "n_cells_evaluated": sum(1 for u in S.B5_PAIRED_UNITS for _ in u.cats),
           "n_cells_skipped_empty_arm": n_skipped_empty_arm,
           "n_flagged_undefined": n_undefined,
           "sieved_n_manski_flagged": None, "sieved_population": None,
           "split_decision_rates": (GD.split_decision_rates(items) if items else None)}
    if pools is not None:
        sflag, n_eval = [], 0
        for u in S.B5_PAIRED_UNITS:
            ref = _sieve_for(rows, pools, u.batch, u.axis)
            for cat in u.cats:
                keys = {k for k in ref if k[0] == cat}
                o = [r["verdict"] for r in uniq if r["arm"] == u.oracle_arm
                     and (r["cat"], r["tid"]) in keys and r["mode"] == u.mode
                     and r["batch"] == u.batch]
                n = [r["verdict"] for r in uniq if r["arm"] == u.none_arm
                     and (r["cat"], r["tid"]) in keys and r["mode"] == u.mode
                     and r["batch"] == u.batch]
                if o and n:
                    n_eval += 1
                    mb = GD.manski_bounds(o, n)
                    if mb.get("flagged"):
                        sflag.append({"cell": f"{u.name}×{cat}", **mb})
        out["sieved_n_manski_flagged"] = len(sflag)
        out["sieved_manski_flagged"] = sflag
        out["sieved_n_cells_evaluated"] = n_eval
        out["sieved_population"] = " μ̂/r P0-GA ∩  ∩ disposition——****"
    return out


def stage_cost(rows: list[dict]) -> dict[str, Any]:
    uniq = list({r["run_id"]: r for r in rows}.values())
    np_np = [r for r in uniq if r["batch"] == "main" and r["truth"] == "NP"
             and r["label"] == "np" and not S.is_replicate_row(r["_row"])]
    inj = [r for r in uniq if r["batch"] == "main" and r["truth"] == "NP"
           and r["label"] in AXES]
    vclass = {}
    for r in inj:
        sib = r["_row"].get("np_trigger_source_sibling")
        if sib:
            vclass[(r["cat"], r["tid"], sib)] = A.victim_risk_class(r["cat"], r["tid"], sib)
    cost: dict[str, Any] = {
        "np_waste": A.np_waste_calls(inj, np_np),
        "np_waste_by_risk_class": A.np_waste_by_risk_class(inj, np_np, vclass),
        "np_waste_class_basis": ("**** = cat/tid/sibling——2026-08-11  C3"
                                 " DECISIONS §13#5(a)`is_action` "
                                 "run  `dar_dispatch.fields_rendered."
                                 "tool` "),
        "prr_per_run": A.prr_per_run(uniq),
        "prr_per_hit_pooled": A.prr_per_hit_pooled(uniq),
    }
    aw = []
    for cat in (A.COST_ONLY_CAT,):
        for truth in A.COST_ONLY_TRUTHS:
            disp = [r for r in uniq if r["cat"] == cat and r["truth"] == truth
                    and r["label"] == "persistent" and r["batch"] == "main"]
            nod = [r for r in uniq if r["cat"] == cat and r["truth"] == truth
                   and r["label"] == "np" and r["batch"] == "main"]
            if disp and nod:
                aw.append({"cell": f"{cat}×{truth}×persistent", **A.avoided_waste(disp, nod)})
    cost["avoided_waste"] = aw
    return cost


_LAB = Path(__file__).resolve().parents[1]


def _portable(p: Path | None, placeholder: str) -> str:
    if p is None:
        return placeholder
    q = Path(p)
    if q.is_absolute():
        try:
            return str(q.resolve().relative_to(_LAB))
        except ValueError:
            return str(q)
    return str(q)


def render(res: dict[str, Any], data: Path | None = None, model: str | None = None,
           out: Path | None = None) -> str:
    L_ = []
    a = L_.append
    a("# B5 ——\n")
    _d = _portable(data, "data/<batch>")
    _m = model or "<model>"
    _o = _portable(out, "docs/<out>.md")
    a(">  = `lab/scripts/b5_report.py`\n>\n"
      "> ```bash\n> cd lab && uv run python scripts/b5_report.py \\\n"
      f"> --data {_d} --model {_m} --out {_o}\n> ```\n")
    a(f"> B={B_BOOT}CI={int((1 - ALPHA_CI) * 100)}%"
      f"bootstrap seed={S.B5_BOOTSTRAP_SEED}Holm  α={res['holm']['alpha']}"
      f" C1 q={res['qc']['judge_q']['q_recon']:.4f}\n")
    v = res["verify"]
    a("\n## §0 \n")
    a(f"-  A run  {v['n_runs_layer_a']} run  {v['n_judged']}"
      f"**** {v['n_structural_undecidable']}"
      f"`no_canonical_path`GT  disposition=exclude ⇒  judge item"
      f"****§2.3 \n")
    a(f"- reason_code `{v['reason_code_counts']}`\n")
    a(f"-  `batch_chunks.json` GPU·h  `analysis/ledger_a.json`  "
      f"`meta.wall_seconds_by_chunk`\n")
    _notes = res["gate0"].get("notes") or []
    for note in _notes:
        a(f"- `matrix_audit` ℹ️{note}\n")
    if _notes:
        a("- **[LE4]  =  §7ter ⑫-B1** chunk1 ****"
          " chunk1  **1,897 run @  + 3 run @ EL9 **⑫-B1  run "
          " `quarantine-20260804/pre-rerun-backup/`** chunk "
          " ⑫-B1  1897+3** 3  run  7,597  run "
          "EL7→EL9⑫-B1 \n")
    a("\n## §2 15  μ̂headlineNP×np \n\n")
    a("> ⚠️ **[LM13]**`NP×np`  μ̂(GA)\n"
      "> ** 1.000**—— P0-GA=achieved§2.2 outcome \n"
      "> P0  GA ⇒ ****`r(t)`  np \n"
      "> P1–P4  × np judge.pass  1\n\n")
    a("> **B-9**——****15 "
      " = ** (batch, , , )**`NP×np` "
      "****§2.2 ⇒ ** 20  = main 1615  + NP×np "
      "+  4** ≠ 15**§4** 30= ** × cat** P1–P4 "
      " A-5  `r(P2)×c1`/`r(P4)×c1`**§5**(C)  = X / 41"
      "= ** × cat ** 11 ****"
      "⚠️ ** 30  41 **\n\n")
    a("| batch |  |  |  | n | μ̂(GA) | μ̂(judge.pass) |  cat n "
      "|  |\n")
    a("|---|---|---|---|---|---|---|---|---|\n")
    for m in res["mu"]:
        a(f"| {m['batch']} | {m['truth']} | {m['label']} | {m['axis']} | {m['n']} | "
          f"{m['mu_ga']:.3f} | {m['mu_judge_pass']:.3f} | `{m['n_by_cat']}` | "
          f"{m.get('task_set_label', '—')} |\n")
    a("\n> ** μ̂ **B-21[LM5] `μ̂(GA)`  `μ̂(judge.pass)`"
      "  **GA-P0 **§2.2 GA-P0 **** 15  μ̂ ——"
      "judge.pass  outcome****⚠️  P0 `p0_pools.json` "
      " judge.pass-P0 \n")
    a("> **B-6 task_set_label ** `stage_mu` "
      " JSONMarkdown  ⇒ \n")
    a("\n## §3 r(t)  d(t,ℓ) bootstrap 90% CId  exploratory\n\n")
    a("> **×CI** = §5-3 [BD-5] \n"
      "> judge item  q  `apply_semantic_verdicts`  run **\n"
      "> ** >0  ≤0 \n\n")
    a("> ****B-6`r(P2)`/`r(P4)`/`r_plain(P2)` ** c1**"
      "cost-only§2.3 `d(·)`  §2  15  μ̂** c1******"
      "—— §2  μ̂  r\n\n")
    a("|  |  | (cats) | none  | oracle  | n |  "
      "| 90% CI | 90% CI·item |\n")
    a("|---|---|---|---|---|---|---|---|---|\n")
    for p in res["paired"]:
        ci = p["ci"]
        rng = (f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if ci["lo"] is not None else "—")
        mean = f"{ci['mean']:+.3f}" if ci["mean"] is not None else "—"
        t2 = p.get("two_layer")
        rng2 = (f"[{t2['lo']:+.3f}, {t2['hi']:+.3f}]"
                if t2 and t2.get("lo") is not None else "")
        deg = " ⚠️" if p.get("degenerate_arm") else ""
        a(f"| {p['unit']}{deg} | {p['tier']} | {','.join(p.get('cats') or [])} | "
          f"{p['none_arm']} | {p['oracle_arm']} | {ci['n']} | {mean} | {rng} | {rng2} |\n")
    degs = [(p["unit"], p["degenerate_arm"]) for p in res["paired"] if p.get("degenerate_arm")]
    if degs:
        a("\n**⚠️ [LM13]**——"
          "\n")
        for name, why in degs:
            a(f"- `{name}`{why}\n")
    a("\n## §3b Holm r(t)>0 × 4 P  α=0.05\n\n")
    a("|  | n | p | p_adj | reject |  90% CI |\n|---|---|---|---|---|---|\n")
    for h in res["holm"]["per_hypothesis"]:
        ci = h["ci"] or {}
        f4 = (lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "—")
        rng = ("—" if ci.get("lo") is None
               else f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
        a(f"| {h['name']} | {h['n']} | {f4(h.get('p'))} | {f4(h.get('p_adj'))} | "
          f"{'✅' if h.get('reject') else ('—' if h.get('reject') is not None else '')}"
          f" | {rng} |\n")
    ng = res["negctl"]
    a("\n## §4 §5-4 + \n\n")
    a(f"- floorrep-mean  C5`{ng['floor'].get('floor_by_cat')}`"
      f" n={ng['floor'].get('n_pairs')}={ng['floor'].get('flip_rate')}\n")
    a(f"- `expected_ci_false_positive_cells`**CI **"
      f"= {ng['verdicts'].get('expected_ci_false_positive_cells')}\n")
    a(f"-  {ng['verdicts'].get('n_cells')} "
      f"`{_verdict_hist(ng['verdicts'])}`\n")
    if ng.get("calibration"):
        a(f"- ** H0 **`noise_floor_calibration`⑪-B "
          f" CI ****——⚠️  cat "
          f"**** cat ** n** A-2 ****"
          f"`{_calib_short(ng['calibration'])}`** n **"
          f" cat  n  c4  n=8  n=2\n")
    a("- §5-4****"
      " H0 ****"
      "——\n")
    a(f"- {S.CROSS_CELL_CAVEAT}\n")
    a("- **=  × cat**B-9 ****§2 = §5 "
      "= ×cat  41 ⚠️ ****"
      " (mode, cat)  **r  d **"
      "** `fa/none` ** ⇒ H0 "
      "** n**[LM17]\n\n"
      "  ```bash\n"
      "  python3 -c \"import json,collections,re;"
      "v=json.load(open('<data>/analysis/b5_analysis.json'))['negctl']['verdicts'];"
      "c=collections.Counter((re.match(r'[rd]\\((P\\d)',x['name']).group(1),x['cat'])"
      " for x in v['cells']);"
      "print(len(v['cells']),collections.Counter(c.values()))\"\n"
      "  ```\n")
    a("- **2026-08-11  A-5 =  §7quinquies ⑭-F**"
      "** r  cost-only ** ⇒ "
      "`r(P2)×c1`  `r(P4)×c1` c1×persistent  = abort"
      " Action Δ ⑨-1 "
      "§5-4  cat  ⇒ ****\n")
    hist = _verdict_hist_by_n(ng["verdicts"])
    if hist:
        a("- ** n 2026-08-11  A-7 =  §7quinquies ⑭-F**"
          "—— H0 A-2 "
          "⇒ ****\n\n")
        a("|  n |  |  | / |  n  H0  |\n")
        a("|---|---|---|---|---|\n")
        for (_cat, n), row in sorted(hist.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            rate = row["h0"]
            a(f"| {n} | {row['']} | {row['']} | {row['']} | "
              f"{('%.4f' % rate) if rate is not None else '—'} |\n")
        a("\n")
    cells = ng["verdicts"].get("cells") or []
    if cells:
        a("- ****§5-4  n  [LD7]③ rn  0 "
          " ⇒  n \n\n")
        a("|  | cat | n |  | 90% CI | floor |  |  H0  |\n")
        a("|---|---|---|---|---|---|---|---|\n")
        for c in sorted(cells, key=lambda x: (x.get("cat") or "", x.get("name") or "")):
            ci = c.get("ci") or {}
            rng = ("—" if ci.get("lo") is None
                   else f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
            mean = f"{c['mean']:+.3f}" if isinstance(c.get("mean"), float) else "—"
            fl = f"{c['floor']:.3f}" if isinstance(c.get("floor"), float) else "—"
            h0 = (f"{c['floor_rule_h0_rate']:.4f}"
                  if isinstance(c.get("floor_rule_h0_rate"), float) else "—")
            a(f"| {c['name']} | {c.get('cat')} | {ci.get('n')} | {mean} | {rng} | {fl} | "
              f"{c.get('verdict') or ('' + str(c.get('reason')))} | {h0} |\n")
        a("\n")

    a("\n## §5  QC\n")
    a(f"-  q= {res['qc']['judge_q']['q_recon']:.4f}"
      f"n={res['qc']['judge_q']['n_recon']} kind "
      f"`{ {k: round(x['q'], 4) for k, x in res['qc']['judge_q']['by_kind'].items()} }`\n")
    a(f"-  q_promptB4 = {res['qc']['judge_q']['q_prompt']}\n")
    a(f"- (N0)-B5  = **{res['qc']['n0'].get('direction', '')}**"
      f"2026-08-11  A-1r  d  ⇒ "
      f" JSON  `qc.n0.sign_by_unit`\n")
    a(f"- (N0)-B5{res['qc']['n0']['warmup_note']}\n")
    a(f"  pooled(full) = `{res['qc']['n0']['pooled']['full']}`\n")
    dsc = res["disclosure"]
    a(f"- (C) Manski  = **{dsc['n_manski_flagged']}** / "
      f"**{dsc.get('n_cells_evaluated')} =  Σ|u.cats| **"
      f"**** {dsc.get('n_cells_skipped_empty_arm')} "
      f"**`manski_bounds` ** {dsc.get('n_flagged_undefined')} "
      f"B-23 [LE9]⚠️ **[LM5]******"
      f"=  −  0 "
      f"** 0 **⚠️ ** 0**"
      f" `stage_disclosure` ——\n")
    a(f"  - **2026-08-11  A-6  =  §7quinquies ⑭-F**"
      f"⚠️ **** ⑭-B  A-6 = (N0)  full  §5 "
      f" (N0) ——V3B-14{dsc.get('population')}\n")
    if dsc.get("sieved_n_manski_flagged") is not None:
        a(f"  - **·**{dsc.get('sieved_population')} "
          f"**{dsc['sieved_n_manski_flagged']}** / {dsc.get('sieved_n_cells_evaluated')} "
          f"——⇒ **≈ undetermined **"
          f" headline  (C) ****\n")
    sdr = dsc.get("split_decision_rates")
    if sdr:
        a("- **(C)  item **`ga_semantic_judge.md` §5 "
          " =  §1.5.6V1 B-5 **** §5-5substitution "
          " arg ****B-5  Markdown \n\n")
        a("| item  | n | EQUIVALENT | NOT_EQUIVALENT | CANNOT_DETERMINE |\n")
        a("|---|---|---|---|---|\n")
        for k in sorted(sdr):
            b = sdr[k] or {}
            rt = b.get("rates") or {}
            f3 = (lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—")
            a(f"| {k} | {b.get('n')} | {f3(rt.get('EQUIVALENT'))} | "
              f"{f3(rt.get('NOT_EQUIVALENT'))} | {f3(rt.get('CANNOT_DETERMINE'))} |\n")
        a("\n  ⚠️ ****[LM5] `action_substitution` "
          "**(canonical path ×  Action )** `arg`  **(path × )** "
          "——** run** run  run "
          "`python3 -c \"import json,collections;c=collections.Counter();"
          "[c.update([json.loads(l)['run_id']]) for l in open('<data>/judge_items.jsonl') "
          "if json.loads(l)['kind']=='action_substitution'];print(len(c),max(c.values()))\"`"
          " = `dar.ga_semantic.semantic_items` JSON  "
          "`disclosure.split_decision_rates`\n")
    a("\n## §6 \n")
    a("- **PRR §5-82026-08-11  A-3 ⑭-G C-3 **"
      "**** = ** 7600 runmain/plain/rep**"
      "⚠️ ****——per-run  **hit-conditioned** = "
      " run `accounting.prr_per_run` docstringper-hit  = hit "
      "7600 ⑭-E A-3  =  7600 run⑭-G C-3 "
      " render "
      "⚠️ PRR  **4.2 **"
      "`fa/none` ≈0.128 ↔ `fa/fixed-transient` ≈0.541****"
      "⑭-E A-3  `res` ——V3B-19"
      "⇒ **pooled **"
      " agent "
      "DECISIONS §14.5②  estimand mean-of-ratios"
      "****—— JSON "
      " `cost`  + `analysis/ledger_b.jsonl`  `prr_hits`/`prr_resolved` \n")
    a(f"- PRRestimand = per-run`{_short(res['cost']['prr_per_run'])}`"
      f" per-hit pooled `{_short(res['cost']['prr_per_hit_pooled'])}`\n")
    nw = res["cost"]["np_waste"]
    _tot = sum(r.get("np_waste_calls") or 0 for r in (nw.get("per_run") or []))
    _np = nw.get("n_pairs") or 0
    _mean = f"{_tot / _np:.4f}" if _np else "—"
    a(f"- **np_waste ** NP×np ** {_tot} "
      f"** {_np} **{_mean}** `cost.np_waste.per_run`"
      f" `{_short(nw)}`** = np  hit **"
      f"`accounting.np_waste_calls` ——** r/d **"
      f"⚠️  **n_pairs**/ `n_tasks`  n\n")
    a(f"- np_waste {res['cost']['np_waste_class_basis']}\n")
    bc = (res["cost"].get("np_waste_by_risk_class") or {}).get("by_class") or {}
    if bc:
        a("- ****§5-7(a) RQ4 ****\n\n")
        a("| victim  | n_tasks | n_pairs |  |  np_waste  "
          "|  np_waste |\n")
        a("|---|---|---|---|---|---|\n")
        for k in sorted(bc):
            v = bc[k] or {}
            mean = v.get("mean_np_waste_calls")
            a(f"| {k} | {v.get('n_tasks')} | {v.get('n_pairs')} | "
              f"{v.get('n_shared_baseline_tasks')} | {v.get('total_np_waste_calls')} | "
              f"{('%.4f' % mean) if isinstance(mean, float) else '—'} |\n")
        a("\n  ⚠️ ** `n_tasks`  n `n_pairs`**——transient  "
          "persistent ** NP×np ** ⇒ n_pairs "
          "`accounting` docstring \n")
        a("  ⚠️ **§5-7(b) **NP×transient  P1 victim NP×persistent "
          " P2 victim U6 first-touchDECISIONS §13#5(b)⇒ ****"
          "armed-not-touchedrun ****\n")
        _cen = None
        if data is not None:
            try:
                import np_provenance_census as _NC
                _cen = _NC.census(data)["summary"]
            except (Exception, SystemExit) as exc:
                _cen = f"__FAILED__{type(exc).__name__}"
        if _cen is None:
            a(f"   `uv run python scripts/np_provenance_census.py "
              f"--data {_d}`****[LM17]\n")
        elif isinstance(_cen, str) and _cen.startswith("__FAILED__"):
            a(f"  🔴 **** `{_cen[len('__FAILED__'):]}` stderr"
              f"—— `uv run python scripts/np_provenance_census.py --data {_d}`\n")
        else:
            a(f"  ** `np_provenance_census.census()` **"
              f"`{_cen}`—— = ⑫-B2 +"
              f" `uv run python scripts/np_provenance_census.py --data {_d}`\n")
    a(f"- avoided_wastecost-only  = c1×{{P2,P4}}×persistent vs  np "
      f"****`{res['cost']['avoided_waste']}`\n")
    return "".join(L_)


def _verdict_hist_by_n(v: dict) -> dict:
    out: dict[tuple[str, int], dict] = {}
    for c in v["cells"]:
        n = c.get("n")
        if n is None:
            continue
        key = (c.get("cat"), n)
        row = out.setdefault(key, {"": 0, "": 0, "": 0, "h0": None})
        verdict = c.get("verdict")
        row[verdict if verdict in ("", "") else ""] += 1
        rate = c.get("floor_rule_h0_rate")
        if row["h0"] is None:
            row["h0"] = rate
        elif rate is not None and rate != row["h0"]:
            raise ValueError(
                "")
    return out


def _verdict_hist(v: dict) -> dict:
    h: dict[str, int] = {}
    for c in v["cells"]:
        k = str(c["verdict"])
        h[k] = h.get(k, 0) + 1
    return h


def _calib_short(c: dict) -> dict:
    per = c["per_cat"]
    for k, v in sorted(per.items()):
        if isinstance(v, dict) and v and "h0_exceed_rate" not in v:
            raise KeyError(
                "")
    return {k: round(v["h0_exceed_rate"], 4)
            for k, v in sorted(per.items())
            if isinstance(v, dict) and isinstance(v.get("h0_exceed_rate"), float)}


def _short(d: dict) -> dict:
    return {k: (round(x, 4) if isinstance(x, float) else x)
            for k, x in d.items() if not isinstance(x, (list, dict))}


def _strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _strip(v)
                for k, v in o.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    if isinstance(o, tuple):
        return list(o)
    return o


def frozen_block(holm_alpha: float) -> dict[str, Any]:
    return {"B": B_BOOT, "alpha_ci": ALPHA_CI,
            "bootstrap_seed": S.B5_BOOTSTRAP_SEED,
            "holm_alpha_one_sided": holm_alpha,
            "adjudications_2026_08_11": {
                "C1": "Holm  α=0.05", "C3": "np_waste ",
                "C5": "floor = rep-mean", "C6": "seed ⑭-C",
                "BD-28①": " B7",
                "A-6": "(N0)  full ", "A-7": "r_plain(P2)  c1"}}


def run(data: Path, model: str, out: Path) -> int:
    res: dict[str, Any] = {"gate0": stage_gate0(data, model)}
    res["verify"] = stage_verify(data)
    vmap, layer_a, items = (res["verify"].pop("_vmap"), res["verify"].pop("_layer_a"),
                            res["verify"].pop("_items"))
    rows = stage_join(data, model, vmap, layer_a)
    pools = stage_pools(rows, data)
    res["mu"] = stage_mu(rows, pools)
    res["qc"] = stage_qc(data, rows, pools, vmap, items)
    q = res["qc"]["judge_q"]["q_recon"]
    res["paired"] = stage_paired(rows, pools, q, vmap)
    res["holm"] = stage_holm(res["paired"])
    res["negctl"] = stage_negctl(rows, pools)
    items_with_verdict = [
        {**it, "verdict": vmap[it["run_id"]][(it["path_id"], it["idx"])]}
        for it in items
        if (it["path_id"], it["idx"]) in vmap.get(it["run_id"], {})]
    res["disclosure"] = stage_disclosure(rows, pools, items_with_verdict)
    res["cost"] = stage_cost(rows)
    res["frozen"] = frozen_block(res["holm"]["alpha"])
    payload = json.dumps(_strip(res), ensure_ascii=False, indent=2,
                         sort_keys=True, default=str) + "\n"
    md = render(res, data=data, model=model, out=out)
    jp = data / "analysis" / "b5_analysis.json"
    _write_idempotent(jp, payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_idempotent(out, md)
    print(f"           {jp}\n           {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", default="docs/b5_analysis_report.md")
    args = ap.parse_args()
    return run(Path(args.data), args.model, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
