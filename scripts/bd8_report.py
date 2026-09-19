#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_judge as J
import bd8_items as I
import bd8_matrix
import run_matrix

from dar.analysis import bd8_ablation as BD
from dar.b5_stats import run_verdict_from_item_map
from dar import ga_disclosure as GD
from dar.ga_semantic import main_estimator


class Bd8ReportError(SystemExit):
    pass


def load_layer_a(data: Path) -> dict[str, dict]:
    rows = {}
    for line in (data / "ga_layerA.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["run_id"] in rows:
                raise Bd8ReportError("")
            rows[r["run_id"]] = r
    if len(rows) != bd8_matrix.EXPECTED_TOTAL:
        raise Bd8ReportError("")
    return rows


def compose_verdicts(data: Path, layer_a: dict[str, dict]) -> dict[str, str]:
    items = J.load_b5_judge_items(data)
    vmap = J.load_verdicts_singlesource(
        data, expected_run_ids=sorted({i["run_id"] for i in items}),
        expected_items=J.items_universe(items))
    eligible = {rid for rid, r in layer_a.items()
                if B4.is_item_eligible(r["verdict"], r.get("reason_code"))}
    if eligible != set(vmap):
        raise Bd8ReportError(
            "")
    structural = {rid for rid, r in layer_a.items()
                  if r.get("reason_code") == "no_canonical_path"}
    if structural & set(vmap):
        raise Bd8ReportError("")
    out = {}
    for rid, r in layer_a.items():
        im = vmap.get(rid)
        out[rid] = run_verdict_from_item_map(im) if im else r["verdict"]
    return out


def achieved_map(layer_a: dict[str, dict], verdicts: dict[str, str]
                 ) -> dict[tuple[str, str, str], bool]:
    out: dict[tuple[str, str, str], bool] = {}
    for rid, r in layer_a.items():
        k = (r["arm"], r["cat"], r["tid"])
        if k in out:
            raise Bd8ReportError("")
        out[k] = main_estimator(verdicts[rid])
    return out


def judge_pass_map(data: Path, layer_a: dict[str, dict], model: str
                   ) -> dict[tuple[str, str, str], bool]:
    inv = {v: k for k, v in I.ARM_LABEL.items()}
    out: dict[tuple[str, str, str], bool] = {}
    missing: list[str] = []
    for rid, r in layer_a.items():
        sub = inv.get(r["arm"])
        if sub is None:
            raise Bd8ReportError("")
        p = (data / sub / model.replace("/", "_").replace(":", "_") / "fc" / r["cat"]
             / "evaluations" / f"{r['tid']}_{r['mode']}_eval.json")
        if not p.is_file():
            missing.append(rid)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("task_id") != r["tid"] or d.get("mode") != r["mode"]:
            raise Bd8ReportError("")
        if not isinstance(d.get("pass"), bool):
            raise Bd8ReportError("")
        out[(r["arm"], r["cat"], r["tid"])] = d["pass"]
    if missing:
        raise Bd8ReportError("")
    return out


def b5_ga_p0_pool(b5_data: Path) -> set[tuple[str, str]]:
    p = b5_data / "analysis" / "p0_pools.json"
    if not p.is_file():
        raise Bd8ReportError("")
    pool = {tuple(s.split("|", 1)) for s in
            json.loads(p.read_text(encoding="utf-8"))["main"]["ga_p0_pool"]}
    if not pool:
        raise Bd8ReportError("")
    return pool


def paired_composition(ach: dict[tuple[str, str, str], bool],
                       sup: dict[str, dict[str, set[tuple[str, str]]]]) -> dict:
    out: dict[str, dict] = {
        "note": " =  Δ [LE1] "
                " cat ",
        "support_label": "headline c1", "by_delta": {},
    }
    for name, truth, pair in (("delta1", "P1", BD.DELTA1), ("delta2", "P2", BD.DELTA2)):
        diffs = BD.paired_diffs(ach, pair, sup["headline"][truth])
        cats = sorted({d["cat"] for d in diffs})
        if set(cats) != set(BD.HEADLINE_CATS):
            raise Bd8ReportError(
                "")
        per: dict[str, dict[str, Any]] = {}
        for cat in cats:
            sub = [d for d in diffs if d["cat"] == cat]
            ds = [d["diff"] for d in sub]
            per[cat] = {"n": len(ds),
                        "pos": sum(1 for v in ds if v > 0),
                        "neg": sum(1 for v in ds if v < 0),
                        "zero": sum(1 for v in ds if v == 0),
                        "ci": BD.delta_ci(sub)}
        tot = {k: sum(c[k] for c in per.values()) for k in ("n", "pos", "neg", "zero")}
        if tot["n"] != len(diffs):
            raise Bd8ReportError("")
        out["by_delta"][name] = {"pair": list(pair), "truth": truth,
                                 "total": tot, "by_cat": per}
    return out


def c_sensitivity(layer_a: dict[str, dict], verdicts: dict[str, str],
                  sup: dict[str, dict[str, set[tuple[str, str]]]]) -> dict:
    tok = {(r["arm"], r["cat"], r["tid"]): verdicts[rid] for rid, r in layer_a.items()}
    la = {(r["arm"], r["cat"], r["tid"]): r["verdict"] for r in layer_a.values()}
    want_cells = {c for pair in BD._TRUTH_CELLS.values() for c in pair}

    on_disk = {arm for arm, _, _ in la}
    if on_disk != want_cells:
        raise Bd8ReportError(
            "")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for use in ("headline", "crossbatch", "c1_only"):
        counts[use] = {}
        for truth, pair in BD._TRUTH_CELLS.items():
            s = sorted(sup[use][truth])
            if not s:
                raise Bd8ReportError("")
            for cid in pair:
                lack = [k for k in s if (cid,) + k not in la]
                if lack:
                    raise Bd8ReportError(
                        "")
                counts[use][cid] = {
                    "n": len(s),
                    "c_layer_a": sum(1 for k in s if la[(cid,) + k] == "undetermined"),
                    "c_post_b": sum(1 for k in s if tok[(cid,) + k] == "undetermined"),
                }

    bounds: dict[str, dict] = {}
    for name, truth, pair in (("delta1", "P1", BD.DELTA1), ("delta2", "P2", BD.DELTA2)):
        s = sorted(sup["headline"][truth])
        va = [tok[(pair[0],) + k] for k in s]
        vb = [tok[(pair[1],) + k] for k in s]
        m = GD.manski_bounds(va, vb)
        if m["main"] is None:
            raise Bd8ReportError("")
        bounds[name] = {"pair": list(pair), "truth": truth, "n": len(s),
                        "lo": m["lo"], "hi": m["hi"], "main": m["main"],
                        "c_sensitive": m["flagged"]}
    return {
        "source": "b5_lm22_extension_table.md  3 ③ prereg §2.4 "
                  " = dar.ga_disclosure.manski_bounds",
        "note": "(C) = GA `undetermined` not_achieved"
                "`lo` = (A)  (C)  not_achieved × (B)  (C)  achieved"
                "`hi` = 🔴 **`lo`  B  (C) == 0 **"
                " ==B  (C)  lo < main⇒ "
                " `lo == main`  B  `c_post_b`"
                "`c_sensitive` = ",
        "counts": counts,
        "manski_headline": bounds,
    }


def build(data: Path, b5_data: Path, model: str) -> dict:
    layer_a = load_layer_a(data)
    verdicts = compose_verdicts(data, layer_a)
    ach = achieved_map(layer_a, verdicts)
    jp = judge_pass_map(data, layer_a, model)
    sup = BD.support_sets(layer_a.values(), ga_p0_pool=b5_ga_p0_pool(b5_data))

    cells = {lbl: BD._TRUTH_CELLS[t] for t in BD._TRUTH_CELLS for lbl in [t]}
    out: dict = {
        "schema": "bd8_analysis/v1",
        "prereg": "lab/docs/bd8_ablation_prereg.md",
        "n_runs": len(layer_a),
        "support_sizes": {u: {t: len(s) for t, s in by_t.items()}
                          for u, by_t in sup.items()},
        "mu": {}, "judge_pass": {}, "headline": None,
    }
    for use in ("headline", "crossbatch", "c1_only"):
        out["mu"][use] = {}
        out["judge_pass"][use] = {}
        for truth, pair in cells.items():
            s = sup[use][truth]
            for cid in pair:
                out["mu"][use][cid] = BD.cell_mu(ach, cid, s)
                out["judge_pass"][use][cid] = BD.cell_mu(jp, cid, s)
    out["headline"] = BD.run_discriminant(ach, sup["headline"])
    out["c1_stratum"] = {
        "note": "c1 =  ⇒ persistent  instruction  graceful_abort"
                "**** §6§9-7",
        "delta_like": {t: {"pair": list(p), "n": len(sup["c1_only"][t]),
                           "mu_a": out["mu"]["c1_only"][p[0]]["mu"],
                           "mu_b": out["mu"]["c1_only"][p[1]]["mu"]}
                       for t, p in cells.items()},
    }
    out["paired_composition"] = paired_composition(ach, sup)
    for name in ("delta1", "delta2"):
        t = out["paired_composition"]["by_delta"][name]["total"]
        ci = out["headline"][name]["ci"]
        if (t["pos"], t["neg"], t["zero"]) != (ci["n_pos"], ci["n_neg"], ci["n_zero"]):
            raise Bd8ReportError(
                "")
    out["c_sensitivity"] = c_sensitivity(layer_a, verdicts, sup)
    for name in ("delta1", "delta2"):
        got = out["c_sensitivity"]["manski_headline"][name]["main"]
        exp = out["headline"][name]["ci"]["mean"]
        if abs(got - exp) > 1e-12:
            raise Bd8ReportError(
                "")
    out["disclosure"] = [
        "headline  c1crossbatch  c1c1_only****§4-b",
        "judge.pass  b5_lm22_extension_table  † "
        "",
        "①  B5  NP×np §5 ",
        " exploratory §6 H",
        " cat  Δ  90% CI  **post-hoc / exploratory**2026-09-02 "
        " §7.1  §4-b  cat ",
        "**** cat [LE1] 2026-09-01 "
        "—— `paired_composition`",
        "(C)undetermined× + Manski lm22 ③ ——"
        " `c_sensitivity`⚠️  A  (C)  post-B  (C) [LM5]",
    ]
    return out


def render(res: dict) -> str:
    a = ["# [BD-8] `bd8_report.py`  `docs/bd8_ablation_findings.md`", ""]
    a.append(f"- runs: {res['n_runs']} = `{json.dumps(res['support_sizes'])}`")
    a.append("")
    a.append("##  μ̂**⛔ **")
    a.append("|  |  | n | μ̂(GA) | judge.pass† |")
    a.append("|---|---|---|---|---|")
    for use in ("headline", "crossbatch", "c1_only"):
        for cid, m in res["mu"][use].items():
            jp = res["judge_pass"][use][cid]["mu"]
            a.append(f"| {use} | `{cid}` | {m['n']} | {m['mu']:.3f} | {jp:.3f} |")
    h = res["headline"]
    a.append("")
    a.append(f"##  = {h['support_label']}cats = {h['cats']}")
    a.append(f"- bootstrap: `{json.dumps(h['bootstrap'], ensure_ascii=False)}`")
    for name in ("delta1", "delta2"):
        d = h[name]
        ci = d["ci"]
        a.append(f"- **Δ{name[-1]}** {d['truth']}n={d['n']}= "
                 f"{ci['mean']:+.4f}90% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
                 f"μ̂ {d['mu_a']:.3f} − {d['mu_b']:.3f}")
    a.append(f"- ****{h['discriminant']['verdict']}`{h['discriminant']['code']}`"
             f"{h['discriminant']['why']}")
    a.append("")
    pc = res["paired_composition"]
    a.append("##  cat ****[LE1]")
    a.append("| Δ | cat | n | +1 | −1 |  |  | Δ(cat) | 90% CI |  0 |")
    a.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in ("delta1", "delta2"):
        b = pc["by_delta"][name]
        for cat, c in b["by_cat"].items():
            ci = c["ci"]
            excl = "" if (ci["lo"] > 0 or ci["hi"] < 0) else "** 0**"
            a.append(f"| Δ{name[-1]} | {cat} | {c['n']} | {c['pos']} | {c['neg']} | "
                     f"{c['zero']} | {c['zero'] / c['n']:.3f} | {ci['mean']:+.4f} | "
                     f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}] | {excl} |")
        t = b["total"]
        a.append(f"| Δ{name[-1]} | **** | {t['n']} | {t['pos']} | {t['neg']} | "
                 f"{t['zero']} | {t['zero'] / t['n']:.3f} | — | — §2.2 | — |")
    a.append("")
    cs = res["c_sensitivity"]
    a.append("## (C) lm22 ③ ****")
    a.append("|  |  | n | (C) A | (C) post-B |")
    a.append("|---|---|---|---|---|")
    for use in ("headline", "crossbatch", "c1_only"):
        for cid, c in cs["counts"][use].items():
            a.append(f"| {use} | `{cid}` | {c['n']} | {c['c_layer_a']} | {c['c_post_b']} |")
    for name in ("delta1", "delta2"):
        b = cs["manski_headline"][name]
        a.append(f"- **Δ{name[-1]}  Manski **headlinen={b['n']}= "
                 f"[{b['lo']:+.4f}, {b['hi']:+.4f}](C)- = {b['c_sensitive']}")
    a.append("")
    a.append("## ")
    a += [f"- {x}" for x in res["disclosure"]]
    return "\n".join(a) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--b5-data", required=True, type=Path)
    ap.add_argument("--model", default=I.DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.out or (args.data / "analysis" / "bd8_analysis.json")
    shared = (args.data / "analysis" / "bd8_analysis.json")
    _same = out.resolve(strict=False) == shared.resolve(strict=False)
    if _same and out.exists() and not args.overwrite:
        raise Bd8ReportError(
            "")
    res = build(args.data, args.b5_data, args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = render(res)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
