#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]

DEFAULT_TREE = "b7-20260824-d"
PRODUCT_FILES = (
    "identity_selfcheck.json", "synthesis.json", "pstar.json", "cost_lambda.json",
    "pi_np_scan.json", "pareto.json", "np_waste.json", "manifest.json",
)
COUNTS_FILES = ("rule", "ml", "llm", "api")
TRUTHS = ("P1", "P2", "P3", "P4", "NP")

LAMBDA_SHOW = (0.01, 1.0, 100.0)
PI_NP_SHOW = (0.0, 0.2, 0.95)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_tree(tree_id: str) -> dict[str, dict]:
    tree = LAB / "data" / tree_id
    if not tree.is_dir():
        raise SystemExit("")
    prov = json.loads((LAB / "provenance" / "b7_products.json").read_text())
    rec = {Path(f["path"]).name: f["sha256"] for f in prov["trees"][tree_id]["files"]}
    docs: dict[str, dict] = {}
    for name in PRODUCT_FILES:
        p = tree / name
        actual = _sha256(p)
        if actual != rec[name]:
            raise SystemExit("")
        docs[name.removesuffix('.json')] = json.loads(p.read_text())
    return docs


def _f(x, nd=4) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _pstar_cell(pt: dict) -> tuple[str, str]:
    reason = pt.get("reason")
    raw = pt.get("raw") or {}
    if pt["value"] is None and reason and reason.startswith("zero_observed_column"):
        wb = raw.get("value_if_zero_weight")
        return ("", f"`{reason}`[BD-71]would-be = {_f(wb)}")
    closed_reason = reason or raw.get("closed_form_reason")
    if closed_reason == "d_bar_negative":
        return (_f(pt["value"]), "`d_bar_negative`")
    if closed_reason == "d_bar_zero":
        return (_f(pt["value"]), " = 0`d_bar_zero`d̄ = 0 ⇒ B(p) = p·r")
    if closed_reason == "denominator_zero_no_breakeven":
        return ("", "`denominator_zero_no_breakeven`r + d̄ = 0")
    return (_f(pt["value"]), "")


def _block(tag: str, lines: list[str]) -> str:
    return "\n".join([f"<!-- b7t:{tag} -->"] + lines + [f"<!-- b7t:/{tag} -->"])


def t1_mu(docs) -> str:
    syn = docs["synthesis"]
    lines = ["|  | μ̂(t, transient) | μ̂(t, persistent) | μ̂(t, np) |",
             "|---|---|---|---|"]
    for t in TRUTHS:
        row = syn["mu_headline_ex_c1"][t]
        lines.append(f"| {t} | {_f(row['transient'])} | {_f(row['persistent'])} | {_f(row['np'])} |")
    lines.append("")
    lines.append("16  = 15  + NP×np axis = ")
    lines.append("| axis |  |  | μ̂ | n |")
    lines.append("|---|---|---|---|---|")
    for r in syn["mu_headline_rows_ex_c1"]:
        lines.append(f"| {r['axis']} | {r['truth']} | {r['label']} | {_f(r['mu_ga'])} | {r['n']} |")
    return _block("T1", lines)


def t2_pstar(docs) -> str:
    ps = docs["pstar"]
    lines = ["|  |  | p\\* | U14①[BD-71] | r | d̄ | p |",
             "|---|---|---|---|---|---|---|"]
    for name in COUNTS_FILES:
        for t in TRUTHS:
            pt = ps["per_file"][name][t]
            raw = pt.get("raw") or {}
            val, mark = _pstar_cell(pt)
            r_v = pt.get("r", raw.get("r"))
            d_v = pt.get("d_bar", raw.get("d_bar"))
            p_v = pt.get("p", raw.get("p"))
            lines.append(f"| {name} | {t} | {val} | {mark} | {_f(r_v)} | {_f(d_v)} | {_f(p_v)} |")
    return _block("T2", lines)


def t3_outcome(docs) -> str:
    syn = docs["synthesis"]
    lines = ["|  |  | Y(t; p̂, T̂) |  | would-be | p̂ |",
             "|---|---|---|---|---|---|"]
    for name in COUNTS_FILES:
        for t in TRUTHS:
            pt = syn["empirical"][name]["outcome_by_truth"][t]
            raw = pt.get("raw") or {}
            reason = pt.get("reason")
            mark = f"`{reason}`[BD-71]" if pt["value"] is None and reason else ""
            lines.append(f"| {name} | {t} | {_f(pt['value'])} | {mark} | "
                         f"{_f(raw.get('value_if_zero_weight'))} | "
                         f"{_f(pt.get('p', raw.get('p')))} |")
    return _block("T3", lines)


def t4_components(docs) -> str:
    ps = docs["pstar"]
    uc = ps["components_ex_c1"]["unit_ci"]
    lines = ["|  c1 | mean | 90% CI | n | n_pos/n_neg/n_zero |",
             "|---|---|---|---|---|"]
    for u in sorted(uc):
        c = uc[u]
        lines.append(f"| {u} | {_f(c['mean'])} | [{_f(c['lo'])}, {_f(c['hi'])}] | {c['n']} | "
                     f"{c['n_pos']}/{c['n_neg']}/{c['n_zero']} |")
    return _block("T4", lines)


def t5_holm(docs) -> str:
    h = docs["pstar"]["holm_ex_c1"]
    lines = [f" = {h['family']}",
             f"m = {h['m']}n_boot = {h['n_boot']}seed = {h['seed']}"
             f"p  = {h['p_smoothing']}α = {h['alpha']}{h['alpha_sided']}"
             f"CI α = {h['ci_alpha']}", "",
             "|  | mean | 90% CI | p | p_adjHolm | reject |",
             "|---|---|---|---|---|---|"]
    for hy in h["per_hypothesis"]:
        c = hy["ci"]
        lines.append(f"| {hy['name']} > 0 | {_f(c['mean'])} | [{_f(c['lo'])}, {_f(c['hi'])}] | "
                     f"{hy['p']:.6g} | {hy['p_adj']:.6g} | {hy['reject']} |")
    return _block("T5", lines)


def t6_pi(docs) -> str:
    syn, scan = docs["synthesis"], docs["pi_np_scan"]
    lines = ["π headline  π = 0.2×5[BD-71] None ",
             "|  | Σ_t π_t·Y(t) |  | would-be |",
             "|---|---|---|---|"]
    for name in COUNTS_FILES:
        ag = syn["empirical"][name]["aggregate_uniform_pi"]
        raw = ag.get("raw") or {}
        mark = f"`{ag.get('reason')}`[BD-71] = {','.join(ag.get('affected_truths', []))}" \
            if ag["value"] is None else ""
        lines.append(f"| {name} | {_f(ag['value'])} | {mark} | {_f(raw.get('value_if_zero_weight'))} |")
    lines.append("")
    lines.append(f"π(NP)  = `{scan['grid_expr']}`headline  π_NP = "
                 f"{scan['headline_pi_np']}  = {scan['headline_on_grid']}"
                 f" = {{{', '.join(str(x) for x in PI_NP_SHOW)}}}")
    lines.append("|  | " + " | ".join(f"π_NP = {x}" for x in PI_NP_SHOW) + " |")
    lines.append("|---|" + "---|" * len(PI_NP_SHOW))
    for name in COUNTS_FILES:
        pts = {p["pi_np"]: p for p in scan["per_file"][name]["points"]}
        cells = []
        for x in PI_NP_SHOW:
            p = pts[x]
            cells.append(_f(p["value"]) if p["value"] is not None else f"`{p['reason']}`")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return _block("T6", lines)


def t7_cost(docs) -> str:
    cl, npw = docs["cost_lambda"], docs["np_waste"]
    lines = [f" A  `{cl['diagnoser_cost_column']['column']}` "
             "",
             "|  | calls | prompt_tokens | completion_tokens | usd_ledger_sum | n_rows |",
             "|---|---|---|---|---|---|"]
    for name in sorted(cl["ledger_a_native"]):
        nat = cl["ledger_a_native"][name]["native"]
        usd = cl["ledger_a_native"][name]["usd_ledger_sum"]
        lines.append(f"| {name} | {nat['calls']} | {nat['prompt_tokens']} | "
                     f"{nat['completion_tokens']} | {usd:.6g} | {nat['n_rows']} |")
    lines.append("")
    lines.append(f"λ combined =  A calls + λ·np_waste π "
                 f" = `{cl['lambda_grid_expr']}` = "
                 f"{{{', '.join(str(x) for x in LAMBDA_SHOW)}}}")
    lines.append("|  | " + " | ".join(f"λ = {x:g}" for x in LAMBDA_SHOW) + " |")
    lines.append("|---|" + "---|" * len(LAMBDA_SHOW))
    for name in sorted(cl["per_file"]):
        pts = {p["lam"]: p["total"] for p in cl["per_file"][name]["lambda_scan"]}
        lines.append(f"| {name} | " + " | ".join(f"{pts[x]:.6g}" for x in LAMBDA_SHOW) + " |")
    lines.append("")
    lines.append("np_waste§6bis estimand")
    lines.append("|  |  |  | n |")
    lines.append("|---|---|---|---|")
    for lab in sorted(npw["per_label"]):
        v = npw["per_label"][lab]
        lines.append(f"| per-labelwaste({lab}) | {_f(v['mean_waste_calls'], 4)} | "
                     f"{v['denominator']} | {v['n_tasks']} |")
    for rc in sorted(npw["risk_class_disclosure"]):
        v = npw["risk_class_disclosure"][rc]
        lines.append(f"| {rc} | {_f(v['mean_waste_calls'], 4)} | "
                     f"n_rows n  n_tasks —— denominator  | "
                     f"n_rows={v['n_rows']}n_tasks={v['n_tasks']} |")
    lines.append("")
    lines.append("π  headlineT[NP][ℓ] ")
    lines.append("|  | Σ_ℓ T[NP][ℓ]·waste(ℓ) |")
    lines.append("|---|---|")
    for name in sorted(npw["pi_weighted_per_file"]):
        lines.append(f"| {name} | {npw['pi_weighted_per_file'][name]['value']:.6g} |")
    return _block("T7", lines)


def t8_pareto(docs) -> str:
    pa = docs["pareto"]
    lines = [f"Δp π  no-dispatch = {_f(pa['delta_p_baseline_no_dispatch'])}"
             f" = {pa['cost_axis']['column']} ",
             "|  | Δp |  | would-be |",
             "|---|---|---|---|"]
    for name in sorted(pa["delta_p"]):
        d = pa["delta_p"][name]
        raw = d.get("raw") or {}
        mark = f"`{d.get('reason')}`[BD-71]" if d["value"] is None else ""
        lines.append(f"| {name} | {_f(d['value'])} | {mark} | {_f(raw.get('value_if_zero_weight'))} |")
    lines.append("")
    fronts = sorted({tuple(p["name"] for p in pl["pareto"]["frontier"]) for pl in pa["per_lambda"]})
    lines.append(f" §8ter λ  {len(pa['per_lambda'])} "
                 f"{fronts}")
    v_expr = pa["per_lambda"][0]["v_scan"]["grid_expr"]
    v_opts = sorted({str(pt["optimal"]) for pl in pa["per_lambda"] for pt in pl["v_scan"]["scan"]})
    v_none = [(pl["lam"], pt["v"]) for pl in pa["per_lambda"]
              for pt in pl["v_scan"]["scan"] if pt["optimal"] is None]
    lines.append("")
    lines.append(f"V V- optimal_for_valueV  = `{v_expr}`"
                 f" λ×V  {len(pa['per_lambda'])}×{len(pa['per_lambda'][0]['v_scan']['scan'])} "
                 f" optimal = {v_opts}"
                 f"optimal = None  = {[(f'λ={l:g}', f'V={v:g}') for l, v in v_none]} optimal")
    pl1 = next(pl for pl in pa["per_lambda"] if pl["lam"] == 1.0)
    lines.append("")
    lines.append("λ = 1  λ excluded ")
    lines.append("|  | cost | Δp |")
    lines.append("|---|---|---|")
    for p in pl1["pareto"]["points"]:
        lines.append(f"| {p['name']} | {p['cost']:.6g} | {_f(p['delta_p'])} |")
    for e in pl1["excluded"]:
        lines.append(f"| excluded{e['file']} | — | `{e['reason'][:28]}…` |"
                     if len(e["reason"]) > 28 else
                     f"| excluded{e['file']} | — | `{e['reason']}` |")
    return _block("T8", lines)


def t9_c1_plain(docs) -> str:
    syn, ps = docs["synthesis"], docs["pstar"]
    c1 = syn["c1_disclosure_parallel"]
    lines = [f"c1  = {c1['tag']} seed = cat == \"c1\"",
             "|  | c1-only mean | c1-only 90% CI | n | headline c1mean |",
             "|---|---|---|---|---|"]
    hd = ps["components_ex_c1"]["unit_ci"]
    for u in sorted(c1["unit_ci"]):
        c = c1["unit_ci"][u]
        lines.append(f"| {u} | {_f(c['mean'])} | [{_f(c['lo'])}, {_f(c['hi'])}] | {c['n']} | "
                     f"{_f(hd[u]['mean'])} |")
    plain = syn["plain_input_face"]
    lines.append("")
    lines.append(f"plain = {plain['tag']}")
    lines.append("| axis |  |  | μ̂ | n |")
    lines.append("|---|---|---|---|---|")
    for r in plain["mu_rows"]:
        lines.append(f"| {r['axis']} | {r['truth']} | {r['label']} | {_f(r['mu_ga'])} | {r['n']} |")
    return _block("T9", lines)


def t0_warnings(docs) -> str:
    seen: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("warnings", "qualifiers", "statements") and isinstance(v, list):
                    seen.update(s for s in v if isinstance(s, str))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for name in (n.removesuffix(".json") for n in PRODUCT_FILES):
        walk(docs[name])
    lines = ["warningsqualifiersstatements "]
    for s in sorted(seen):
        lines.append(f"> {s}")
    return _block("T0", lines)


TABLES = (t0_warnings, t1_mu, t2_pstar, t3_outcome, t4_components,
          t5_holm, t6_pi, t7_cost, t8_pareto, t9_c1_plain)


def render(tree_id: str) -> str:
    docs = load_tree(tree_id)
    parts = [f"<!-- b7_findings_tables = {tree_id} =  sha  "
             f"lab/provenance/b7_products.json  -->"]
    parts += [fn(docs) for fn in TABLES]
    return "\n\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=DEFAULT_TREE)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    text = render(a.tree)
    if a.out:
        a.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
