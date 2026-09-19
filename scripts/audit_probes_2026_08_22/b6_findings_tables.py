#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "scripts"))
sys.path.insert(0, str(LAB / "src"))

import b6_subset as SUB
import b6_t1_collect as COL
import measure_diagnosers as MD
from dar.diagnoser import llm as LLM
from dar.labels import Label, TrueType, correct_label, true_type_from_mode

if not __debug__:
    raise SystemExit("")


DEFAULT_CANON = LAB / "data" / "b6-20260814"
DEFAULT_CORPUS = LAB / "data" / "b5-11090655"
DEFAULT_T1 = LAB / "data" / "b6-11309161"
DEFAULT_T2 = LAB / "data" / "b6-t2api-20260820"

HEAD_TO_HEAD = ("rule", "ml", "llm", "api", "oracle")
DISCLOSURE = ("rule", "ml")

STRATUM_ROWS = ("P1", "P2", "P3", "P4", "P0×transient", "P0×persistent")

_MODE_TO_TRUE_LABEL = {
    m: correct_label(true_type_from_mode(m)).value for m in ("P0", "P1", "P2", "P3", "P4")
}


def stratum_key(unit: dict[str, Any]) -> str:
    tt = true_type_from_mode(unit["mode"])
    if tt is TrueType.NP:
        return f"P0×{unit['axis']}"
    return str(unit["mode"])


def empty_stratum() -> dict[str, dict[str, int]]:
    return {r: {c: 0 for c in MD.AXIS_ORDER} for r in STRATUM_ROWS}


def aggregate_mode_split(strat: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    out = MD.empty_matrix()
    for row_key, row in strat.items():
        mode = row_key.split("×", 1)[0]
        if mode not in _MODE_TO_TRUE_LABEL:
            raise SystemExit("")
        true_label = _MODE_TO_TRUE_LABEL[mode]
        for col in MD.AXIS_ORDER:
            if col not in row:
                raise SystemExit("")
            out[true_label][col] += int(row[col])
    return out


def payload_axis_order(payload: dict[str, Any]) -> list[str]:
    ao = payload.get("axis_order")
    if not ao:
        raise SystemExit("")
    if set(ao) != set(MD.AXIS_ORDER):
        raise SystemExit("")
    return list(ao)


def counts_rows(payload: dict[str, Any]) -> list[tuple[str, list[int], int]]:
    ao = payload_axis_order(payload)
    matrix = payload["matrix"]
    out: list[tuple[str, list[int], int]] = []
    for row in ao:
        cells = [int(matrix[row][col]) for col in ao]
        out.append((row, cells, sum(cells)))
    return out


def diagonal_shares(payload: dict[str, Any]) -> dict[str, tuple[int, int]]:
    ao = payload_axis_order(payload)
    matrix = payload["matrix"]
    out: dict[str, tuple[int, int]] = {}
    for row in ao:
        n = sum(int(matrix[row][c]) for c in ao)
        if n == 0:
            raise SystemExit("")
        out[row] = (int(matrix[row][row]), n)
    return out


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(_cell(h) for h in headers) + " |"
    rule = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join([head, rule, *body])


QUANTILE_CALIBER = ("p50 / p90 = nearest-above`sorted[int(n·q)]`"
                    " = b6_preregistration.md §6.2"
                    "⚠️ 2026-08-22  statistics.median + "
                    " b6_v1_review_2026-08-22.md V1-5")


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise SystemExit("")
    s = sorted(values)
    n = len(s)
    q = lambda p: s[min(n - 1, int(n * p))]
    return {"n": n, "min": s[0], "p50": q(0.5), "p90": q(0.9), "max": s[-1]}


def load_counts(canon: Path, name: str, face: str = "") -> dict[str, Any]:
    stem = f"{name}.{face}." if face else f"{name}."
    return json.loads((canon / "matrix" / f"{stem}counts.json").read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def assert_call_site_wiring(canon: Path) -> dict[str, Any]:
    man = json.loads((canon / "manifest.json").read_text())
    cfg = SUB.load_config()
    import inspect
    sig = inspect.signature(SUB.enumerate_units).parameters
    assert sig["with_task_description"].default is True, ""
    assert sig["with_tool_schemas"].default is True, ""
    assert list(MD.AXIS_ORDER) == [l.value for l in Label], ""
    import ast as _ast
    _tree = _ast.parse(inspect.getsource(MD.run))
    _ml_calls = [n for n in _ast.walk(_tree)
                 if isinstance(n, _ast.Call)
                 and isinstance(n.func, _ast.Attribute)
                 and n.func.attr == "enumerate_units"
                 and any(k.arg == "with_tool_schemas" for k in n.keywords)]
    assert _ml_calls, ("")
    for _c in _ml_calls:
        _kw = {k.arg: getattr(k.value, "value", None) for k in _c.keywords}
        assert _kw.get("with_task_description") is False and \
            _kw.get("with_tool_schemas") is False, (
                "")

    seed = man["sampling"]["subset_seed"]
    assert seed == int(cfg["measurement"]["subset_seed"]), (
        "")
    assert man["sampling"]["main_arm"] == cfg["measurement"]["main_arm"], (
        "")
    return {"manifest": man, "cfg": cfg}


def head_to_head_units(canon: Path, corpus: Path) -> list[dict[str, Any]]:
    ctx = assert_call_site_wiring(canon)
    man, cfg = ctx["manifest"], ctx["cfg"]
    units = SUB.enumerate_units(corpus, cfg["measurement"]["main_arm"], man["subset_ids"])
    if len(units) != man["n_diagnoses"]:
        raise SystemExit("")
    return units


def labels_by_run_id(canon: Path, corpus: Path, t1: Path, t2: Path,
                     units: list[dict[str, Any]], *, with_ml: bool,
                     ) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for name, fn in (("rule", MD.predict_rule), ("oracle", MD.predict_oracle)):
        labels, _ledger_ignored = fn(units)
        out[name] = {u["run_id"]: l for u, l in zip(units, labels)}
    t1_main, _ext = COL.arm_split(load_jsonl(t1 / "responses.jsonl"))
    for name, rows in (("llm", t1_main), ("api", load_jsonl(t2 / "responses.jsonl"))):
        ok, _err = COL.split_request_errors(rows)
        parses, _reasons = COL.parse_arm(ok)
        out[name] = {rid: p.label for rid, p in parses.items()}
    if with_ml:
        train: list[dict[str, Any]] = []
        for arm in SUB.MAIN_BATCH_ARMS:
            train.extend(SUB.enumerate_units(corpus, arm, None,
                                             with_task_description=False,
                                             with_tool_schemas=False))
        ml_labels, _report = MD.ml_out_of_fold(train, SUB.load_config())
        labels, _ledger_ignored = MD.predict_ml(units, ml_labels)
        out["ml"] = {u["run_id"]: l for u, l in zip(units, labels)}
    return out


def mode_split(units: list[dict[str, Any]],
               label_of: dict[str, str | None]) -> dict[str, dict[str, int]]:
    table = empty_stratum()
    for u in units:
        if u["obs"] is None:
            continue
        lab = label_of.get(u["run_id"])
        if lab is None:
            continue
        table[stratum_key(u)][lab] += 1
    return table


IDENTITY_SCOPE = ("matrix", "n_parsed", "n_unparseable", "excluded_counts")


def identity_check(canon: Path, units: list[dict[str, Any]],
                   labels: dict[str, dict[str, str | None]],
                   names: tuple[str, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name in names:
        counts, excluded, n_unp, _by_cell = MD.tally(
            units, [labels[name].get(u["run_id"]) for u in units])
        frozen = load_counts(canon, name)
        got = {"matrix": counts, "excluded_counts": excluded, "n_unparseable": n_unp,
               "n_parsed": sum(counts[r][c] for r in counts for c in counts[r])}
        diffs = {k: {"got": got[k], "frozen": frozen[k]}
                 for k in IDENTITY_SCOPE if got[k] != frozen[k]}
        if name in ("llm", "api"):
            n_err = len([u for u in units
                         if u["obs"] is not None and u["run_id"] not in labels[name]])
            if n_err != frozen["n_request_error"]:
                diffs["n_request_error"] = {"got": n_err,
                                            "frozen": frozen["n_request_error"]}
        agg_ok = aggregate_mode_split(mode_split(units, labels[name])) == frozen["matrix"]
        results.append({"diagnoser": name, "bit_identical": not diffs, "diffs": diffs,
                        "mode_split_aggregates_back": agg_ok})
    return results


def render(canon: Path, corpus: Path, t1: Path, t2: Path, *, with_ml: bool) -> str:
    man = json.loads((canon / "manifest.json").read_text())
    man_t1 = json.loads((canon / "manifest_t1.json").read_text())
    man_t2 = json.loads((canon / "manifest_t2.json").read_text())
    units = head_to_head_units(canon, corpus)
    labels = labels_by_run_id(canon, corpus, t1, t2, units, with_ml=with_ml)
    ao = list(MD.AXIS_ORDER)
    P: list[str] = []
    add = P.append

    add("<!--  scripts/audit_probes_2026_08_22/b6_findings_tables.py "
        " ⇒  -->")
    add(f"# B6  = {man['tree_id']}\n")
    if not with_ml:
        add("> ⚠️ **ML **`--skip-ml` ****"
            "ML ML `n_parsed`"
            "`excluded_counts`  `ml_training.per_fold` "
            "`--with-ml`≈69 min CPU \n")

    add("## T1. head-to-head 3×3500 \n")
    add(f" = `head_to_head` =  `axis_order` = `{ao}`"
        f" α = 0\n")
    for name in HEAD_TO_HEAD:
        pl = load_counts(canon, name)
        add(f"### T1-{name} · counts\n")
        rows = [[r, *[str(v) for v in cells], str(tot)]
                for r, cells, tot in counts_rows(pl)]
        add(md_table(["\\", *ao, ""], rows))
        rn = MD.rownorm(pl["matrix"])
        add("\n `*.rownorm.json`——B7 `b7_preregistration.md` §2.3\n")
        add(md_table(["\\", *ao],
                     [[r, *[f"{rn[r][c]:.4f}" for c in ao]] for r in ao]))
        sh = diagonal_shares(pl)
        add("\n** CI**/\n")
        add(md_table(["", "/", ""],
                     [[r, f"{sh[r][0]}/{sh[r][1]}", f"{sh[r][0] / sh[r][1]:.4f}"]
                      for r in ao]))
        add(f"\n`n_parsed` = {pl['n_parsed']}`n_unparseable` = {pl['n_unparseable']}"
            f" {pl['unparseable_rate']}`excluded_counts` = "
            f"{json.dumps(pl['excluded_counts'], ensure_ascii=False, sort_keys=True)}"
            + (f"`n_request_error` = {pl['n_request_error']}"
               if "n_request_error" in pl else "") + "\n")

    add("## T2.  mode \n")
    add(f" = {list(STRATUM_ROWS)}P0  `axis` §2ter"
        "** 3×3 == **\n")
    for name in HEAD_TO_HEAD:
        if name == "ml" and not with_ml:
            add(f"### T2-{name}\n\n⚠️ ****——ML  mode "
                "`--skip-ml` \n")
            continue
        st = mode_split(units, labels[name])
        add(f"### T2-{name}\n")
        add(md_table(["mode", *ao, ""],
                     [[r, *[str(st[r][c]) for c in ao],
                       str(sum(st[r][c] for c in ao))] for r in STRATUM_ROWS]))
        same = aggregate_mode_split(st) == load_counts(canon, name)["matrix"]
        add(f"\n 3×3 == **{same}**\n")

    add("## T3.  cat \n")
    for name in HEAD_TO_HEAD:
        if name == "ml" and not with_ml:
            add(f"### T3-{name}\n\n⚠️ **** T2-ml\n")
            continue
        cats = sorted({u["cat"] for u in units})
        tab = {c: {k: 0 for k in ao} for c in cats}
        for u in units:
            if u["obs"] is None:
                continue
            lab = labels[name].get(u["run_id"])
            if lab is None:
                continue
            tab[u["cat"]][lab] += 1
        add(f"### T3-{name}\n")
        add(md_table(["cat", *ao, ""],
                     [[c, *[str(tab[c][k]) for k in ao],
                       str(sum(tab[c].values()))] for c in cats]))
        add("")

    add("## T4. ** head-to-head **\n")
    for name in DISCLOSURE:
        pl = load_counts(canon, name, "disclosure")
        add(f"### T4-{name} · \n")
        add(md_table(["\\", *ao, ""],
                     [[r, *[str(v) for v in cells], str(tot)]
                      for r, cells, tot in counts_rows(pl)]))
        add(f"\n`face_note` = {pl['face_note']}`n_parsed` = {pl['n_parsed']}"
            f"`excluded_counts` = "
            f"{json.dumps(pl['excluded_counts'], ensure_ascii=False, sort_keys=True)}\n")
        for arm in sorted(pl["per_arm_counts"]):
            mm = pl["per_arm_counts"][arm]
            add(f"#### T4-{name} · `{arm}`\n")
            add(md_table(["\\", *ao, ""],
                         [[r, *[str(mm[r][c]) for c in ao],
                           str(sum(mm[r][c] for c in ao))] for r in ao]))
            add("")
        add(" `per_arm_excluded` = "
            f"{json.dumps(pl['per_arm_excluded'], ensure_ascii=False, sort_keys=True)}\n")

    add("## T5.  A#12\n")
    add(f"****{QUANTILE_CALIBER}\n")
    off = load_jsonl(canon / "ledger_a.jsonl")
    add("### T5-a rule / ml / oracle· `ledger_a.jsonl`\n")
    add("`basis` `DiagnoserCostRecord.basis` [BD-11]\n")
    for b in sorted({r["basis"] for r in off}):
        add(f"- `{b}`")
    add("")
    rows = []
    for d in sorted({r["diagnoser"] for r in off}):
        sub = [r for r in off if r["diagnoser"] == d]
        q = quantiles([r["latency_ms"] for r in sub])
        rows.append([d, str(len(sub)),
                     str(sum(r["calls"] for r in sub)),
                     str(sum(r["prompt_tokens"] for r in sub)),
                     str(sum(r["completion_tokens"] for r in sub)),
                     f"{q['min']:.6f}", f"{q['p50']:.6f}", f"{q['p90']:.6f}",
                     f"{q['max']:.6f}",
                     str(len({r["latency_ms"] for r in sub}))])
    add(md_table(["", "", "Σcalls", "Σprompt_tok", "Σcompletion_tok",
                  "latency_ms min", "p50", "p90", "max", ""], rows))
    add("\n⚠️ **`latency_ms` **——`measure_diagnosers`"
        " docstring ****"
        "`ledger_a.jsonl` \n")

    add("### T5-b  LLM T1· `ledger_a_t1.jsonl`\n")
    t1_rows = load_jsonl(canon / "ledger_a_t1.jsonl")
    add(f"`basis` = `{sorted({r['basis'] for r in t1_rows})[0]}`\n")
    rows = []
    for axis in sorted({r["axis"] for r in t1_rows}):
        sub = [r for r in t1_rows if r["axis"] == axis]
        qp = quantiles([r["prompt_tokens"] for r in sub])
        qc = quantiles([r["completion_tokens"] for r in sub])
        rows.append([axis, str(len(sub)), str(sum(r["calls"] for r in sub)),
                     str(sum(r["prompt_tokens"] for r in sub)),
                     f"{qp['p50']:.0f}", f"{qp['max']:.0f}",
                     str(sum(r["completion_tokens"] for r in sub)),
                     f"{qc['p50']:.0f}", f"{qc['max']:.0f}"])
    add(md_table(["axis", "", "Σcalls", "Σprompt", "prompt p50", "prompt max",
                  "Σcompletion", "compl p50", "compl max"], rows))
    add("\n`latency_ms`**** `concurrency`  T5 \n")
    lrows = []
    for axis in sorted({r["axis"] for r in t1_rows}):
        sub = [r for r in t1_rows if r["axis"] == axis]
        ql = quantiles([r["latency_ms"] for r in sub])
        lrows.append([axis, str(len(sub)),
                      f"{ql['min']:.3f}", f"{ql['p50']:.3f}", f"{ql['p90']:.3f}",
                      f"{ql['max']:.3f}",
                      str(sorted({r["concurrency"] for r in sub}))])
    main_axes = {"main", "transient", "persistent"}
    msub = [r for r in t1_rows if r["axis"] in main_axes]
    qm = quantiles([r["latency_ms"] for r in msub])
    lrows.append(["****main+transient+persistent", str(len(msub)),
                  f"{qm['min']:.3f}", f"{qm['p50']:.3f}", f"{qm['p90']:.3f}",
                  f"{qm['max']:.3f}",
                  str(sorted({r["concurrency"] for r in msub}))])
    add(md_table(["axis", "", "min", "p50", "p90", "max", "concurrency "],
                 lrows))
    add("\n****\n")
    lc = man_t1["latency_calibers"]
    add(f"-  `latency_ms` `concurrency` "
        f" = {sorted({r['concurrency'] for r in t1_rows})}")
    add(f"- T1  n = {lc['t1_serial_prefix']['n']}"
        f"`usable_as_pure_latency` = **{lc['t1_serial_prefix']['usable_as_pure_latency']}**"
        f" = {lc['t1_serial_prefix']['ruling_pointer']}")
    mas = man_t1["main_arm_serial_latency"]
    add(f"- T1b  n = {mas['n']} {len(mas['per_cell'])} "
        f"`destination` = **{mas['destination']}**[BD-34]  = disclosure-only")
    add(f"- {lc['serial_row_key']}\n")

    add("### T5-c API T2· `ledger_a_t2.jsonl`\n")
    t2_rows = load_jsonl(canon / "ledger_a_t2.jsonl")
    add(f"`basis` = `{sorted({r['basis'] for r in t2_rows})[0]}`\n")
    qp = quantiles([r["prompt_tokens"] for r in t2_rows])
    qc = quantiles([r["completion_tokens"] for r in t2_rows])
    qu = quantiles([r["usd"] for r in t2_rows])
    add(md_table(["", "n", "Σ", "min", "p50", "p90", "max"], [
        ["prompt_tokens", str(qp["n"]), str(sum(r["prompt_tokens"] for r in t2_rows)),
         f"{qp['min']:.0f}", f"{qp['p50']:.0f}", f"{qp['p90']:.0f}", f"{qp['max']:.0f}"],
        ["completion_tokens", str(qc["n"]), str(sum(r["completion_tokens"] for r in t2_rows)),
         f"{qc['min']:.0f}", f"{qc['p50']:.0f}", f"{qc['p90']:.0f}", f"{qc['max']:.0f}"],
        ["usd", str(qu["n"]), f"{sum(r['usd'] for r in t2_rows):.6f}",
         f"{qu['min']:.6f}", f"{qu['p50']:.6f}", f"{qu['p90']:.6f}", f"{qu['max']:.6f}"],
    ]))
    tot = man_t2["batch_usage_reconciliation"]["totals"]
    add(f"\n`cached_tokens`  input = {tot['cached']}/{tot['api_in']} = "
        f"{tot['cached'] / tot['api_in']:.6f}`reasoning_tokens`  output = "
        f"{tot['reasoning']}/{tot['api_out']} = {tot['reasoning'] / tot['api_out']:.6f}\n")
    add(f"`actual_usd_caliber`{man_t2['pricing']['actual_usd_caliber']}\n")
    add(f"`bill_level_reconciliation`"
        f"{man_t2['pricing']['bill_level_reconciliation']}\n")
    add(f"`power_note`{man_t2['batch_usage_reconciliation']['power_note']}\n")

    add("### T5-d `llm.extended_arm.cost_bound.json`\n")
    cb = json.loads((canon / "matrix" / "llm.extended_arm.cost_bound.json").read_text())
    add(f" = `{cb['face']}``n_requests` = {cb['n_requests']}`n_ok` = {cb['n_ok']}"
        f"`n_request_error` = {cb['n_request_error']}"
        f" {json.dumps(cb['request_error_by_cell'], ensure_ascii=False, sort_keys=True)}\n")
    add(md_table(["", ""], [[k, str(v)] for k, v in sorted(cb["totals"].items())]))
    add(f"\n`coverage_note`{cb['coverage_note']}\n")
    add(f"`caliber_rule`{cb['caliber_rule']}\n")
    add("20 `t1b_latency`  T1b token/calls  T1 \n")
    add(md_table(["", "calls", "prompt_tok", "completion_tok", "n_request_error",
                  "t1b n", "t1b mean_ms"],
                 [[k, str(v["calls"]), str(v["prompt_tokens"]),
                   str(v["completion_tokens"]), str(v["n_request_error"]),
                   str(v["t1b_latency"]["n"]) if v.get("t1b_latency") else "—",
                   f"{v['t1b_latency']['mean_ms']:.1f}" if v.get("t1b_latency") else "—"]
                  for k, v in sorted(cb["per_cell"].items())]))
    add("")

    add("## T6.  QC \n")
    add(md_table(["", ""], [
        ["ML `k_folds`", str(load_counts(canon, "ml")["ml_training"]["k_folds"])],
        ["ML `n_train_units`", str(load_counts(canon, "ml")["ml_training"]["n_train_units"])],
        ["ML `fold_sizes`", json.dumps(load_counts(canon, "ml")["ml_training"]["fold_sizes"],
                                       sort_keys=True)],
        ["T1  `n_truncated`", str(man_t1["counts"]["main_arm"]["n_truncated"])],
        ["T1  `n_truncated`", str(man_t1["counts"]["extended_arm"]["n_truncated"])],
        ["T2 `finish_reason_histogram`",
         json.dumps(man_t2["counts"]["main_arm"]["finish_reason_histogram"], sort_keys=True)],
        ["T2 `model_echo_histogram`",
         json.dumps(man_t2["counts"]["main_arm"]["model_echo_histogram"], sort_keys=True)],
        ["T2 `n_text_none_error_free`",
         str(man_t2["counts"]["main_arm"]["n_text_none_error_free"])],
        ["T1 `driver_parse_retries`", str(man_t1["parse"]["driver_parse_retries"])],
        ["T1 `collect_reparse_retries`", str(man_t1["parse"]["collect_reparse_retries"])],
        ["Σ  `usd``ledger_a_t2.jsonl` ", f"{sum(r['usd'] for r in t2_rows):.6f}"],
        ["config `llm.api.max_budget_usd`U12  $50",
         str(SUB.load_config()["llm"]["api"]["max_budget_usd"])],
    ]))
    add("\nML  fold `stopped_by_tolerance`  True\n")
    pf = load_counts(canon, "ml")["ml_training"]["per_fold"]
    add(md_table(["fold", "n_samples", "epochs_ran", "initial_objective",
                  "final_objective", "final ≤ initial", "stopped_by_tolerance",
                  "max|W|", "isfinite(max|W|)"],
                 [[k, str(v["n_samples"]), str(v["epochs_ran"]),
                   f"{v['initial_objective']:.10f}", f"{v['final_objective']:.10f}",
                   str(v["final_objective"] <= v["initial_objective"]),
                   str(v["stopped_by_tolerance"]),
                   f"{v['max_abs_weight']:.6f}",
                   str(v["max_abs_weight"] == v["max_abs_weight"]
                       and abs(v["max_abs_weight"]) != float("inf"))]
                  for k, v in sorted(pf.items())]))
    add(f"\nΣ`epochs_ran`= **{sum(v['epochs_ran'] for v in pf.values())}**"
        f" min = {min(v['epochs_ran'] for v in pf.values())}"
        f"max = {max(v['epochs_ran'] for v in pf.values())}")
    add(f"max|W|  = [{min(v['max_abs_weight'] for v in pf.values()):.6f}, "
        f"{max(v['max_abs_weight'] for v in pf.values()):.6f}]")
    add("")

    add("## T7. §5.2\n")
    names = tuple(n for n in HEAD_TO_HEAD if n != "ml" or with_ml)
    res = identity_check(canon, units, labels, names)
    add(f"= `{list(IDENTITY_SCOPE)}`"
        "api/llm  `n_request_error`** payload **"
        "—— `c5577b4` U16④  `unparseable_by_cell`\n")
    add(md_table(["", "", " 3×3", ""],
                 [[r["diagnoser"], str(r["bit_identical"]),
                   str(r["mode_split_aggregates_back"]),
                   "—" if not r["diffs"] else json.dumps(r["diffs"], ensure_ascii=False)]
                  for r in res]))
    if not with_ml:
        add("\n⚠️ **ML **`--skip-ml`")
    add("")
    return "\n".join(P) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_CANON)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--t1", type=Path, default=DEFAULT_T1)
    ap.add_argument("--t2", type=Path, default=DEFAULT_T2)
    ap.add_argument("--with-ml", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.out is not None and str(a.out.resolve()).startswith(str((LAB / "data").resolve())):
        raise SystemExit("")
    text = render(a.data, a.corpus, a.t1, a.t2, with_ml=a.with_ml)
    if a.out:
        a.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
