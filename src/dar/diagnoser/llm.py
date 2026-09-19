from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from dar.diagnoser.base import DiagnoserCostRecord, DiagnosisInput, DiagnosisOutput
from dar.labels import Label

_DOMAIN: dict[str, Label] = {label.value.upper(): label for label in Label}

DEFINITIONS: dict[Label, dict[str, str]] = {
    Label.TRANSIENT: {
        "definition": "A tool returned an explicit error signal, and the trace gives no "
                      "indication that the same tool would keep failing on a retry.",
        "example": "One call to `book_flight` returns {\"error\": \"Service Temporarily "
                   "Unavailable\"} and that tool has not failed before in this trace.",
    },
    Label.PERSISTENT: {
        "definition": "The same tool returned an explicit error signal on two or more "
                      "consecutive calls, so retrying that tool is not expected to help.",
        "example": "`book_flight` returns {\"error\": ...} twice in a row with no "
                   "successful call in between.",
    },
    Label.NP: {
        "definition": "The trace prefix carries no failure signal at all: every tool "
                      "result so far looks like a normal payload.",
        "example": "Every tool result is a regular object with no `error` key.",
    },
}


def render_definitions(definitions: dict[Label, dict[str, str]] | None = None) -> str:
    table = DEFINITIONS if definitions is None else definitions
    lines = []
    for label, body in table.items():
        lines.append(f"- {label.value.upper()}: {body['definition']}")
        lines.append(f"  Example: {body['example']}")
    return "\n".join(lines)


def domain_text() -> str:
    return "|".join(_DOMAIN)


BASE_RUBRIC = """\
You are a diagnostic classifier for a tool-using agent. You are shown a prefix of one
agent trajectory, truncated at the moment a diagnosis is requested, together with the
task the agent was given and the schemas of the tools it may call.

Decide which single label describes the situation at the end of the prefix:

{definitions}

Output contract — return exactly one JSON object and nothing else:
{{"label": "<{domain}>", "reasoning": "<one or two sentences>"}}
No markdown code fences, no preamble, no trailing commentary.
"""

VARIANT_OVERLAY = """\
Procedure (follow it in order; do not restate the label definitions or the output
contract above — they are fixed):

1. Walk the trace prefix from the start and note, for every tool result, the tool name
   and whether the result carries an explicit error signal.
2. If there is no explicit error signal anywhere in the prefix, stop and answer with the
   no-problem label.
3. Otherwise, check whether the same tool produced an explicit error on two or more
   consecutive calls, with no successful call of that tool in between.
4. Answer with the persistent label if it did, and with the transient label if it did not.
5. Keep the reasoning field to at most two sentences that cite the tool names you used.
"""

USER_TEMPLATE = """\
## Task given to the agent
{task_description}

## Tool schemas
{tool_schemas}

## Trace prefix (truncated at the diagnosis point)
{trace_prefix}
"""


def system_prompt() -> str:
    base = BASE_RUBRIC.format(definitions=render_definitions(), domain=domain_text())
    return base.rstrip() + "\n\n" + VARIANT_OVERLAY.lstrip()


def user_prompt(obs: DiagnosisInput) -> str:
    return USER_TEMPLATE.format(
        task_description=obs.task_description or "(not provided)",
        tool_schemas=json.dumps(list(obs.tool_schemas), ensure_ascii=False, indent=2)
        if obs.tool_schemas else "(not provided)",
        trace_prefix=json.dumps(list(obs.trace_prefix), ensure_ascii=False, indent=2),
    )


def skeleton_fingerprint() -> str:
    blob = "\x00".join([BASE_RUBRIC, VARIANT_OVERLAY, USER_TEMPLATE,
                        render_definitions(), domain_text(), system_prompt()])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_tokens_chars4(text: str) -> float:
    return len(text) / 4.0


def skeleton_token_estimate() -> dict[str, float]:
    fixed_user = USER_TEMPLATE.format(task_description="", tool_schemas="", trace_prefix="")
    sysp = system_prompt()
    return {"system_chars": float(len(sysp)),
            "system_tokens_chars4": estimate_tokens_chars4(sysp),
            "user_template_chars": float(len(fixed_user)),
            "user_template_tokens_chars4": estimate_tokens_chars4(fixed_user),
            "skeleton_tokens_chars4": estimate_tokens_chars4(sysp + fixed_user)}


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_BRACES = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParseResult:
    label: Label | None
    reason: str
    raw_label: str | None = None


def parse_response(text: Any) -> ParseResult:
    if not isinstance(text, str):
        return ParseResult(None, "no_json", raw_label=None if text is None else repr(text)[:80])
    stripped = _FENCE.sub("", text.strip())
    obj: Any = None
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        m = _BRACES.search(stripped)
        if m is None:
            return ParseResult(None, "no_json")
        try:
            obj = json.loads(m.group(0))
        except (ValueError, TypeError):
            return ParseResult(None, "no_json")
    if not isinstance(obj, dict):
        return ParseResult(None, "not_object")
    if "label" not in obj:
        return ParseResult(None, "no_label_key")
    raw = obj["label"]
    if not isinstance(raw, str):
        return ParseResult(None, "label_out_of_domain", raw_label=repr(raw))
    key = raw.strip().upper()
    if key not in _DOMAIN:
        return ParseResult(None, "label_out_of_domain", raw_label=raw)
    return ParseResult(_DOMAIN[key], "ok", raw_label=raw)


CompleteFn = Callable[[str, str], dict[str, Any]]


class LlmDiagnoser:
    name = "llm"

    def __init__(self, complete_fn: CompleteFn, *, name: str = "llm",
                 parse_retries: int = 0, usd_per_mtok: float | None = None,
                 basis: str | None = None, require_usage: bool = True) -> None:
        if parse_retries < 0:
            raise ValueError("")
        self.complete_fn = complete_fn
        self.name = name
        self.parse_retries = int(parse_retries)
        self.usd_per_mtok = usd_per_mtok
        self.basis = basis
        self.require_usage = bool(require_usage)
        self.n_unparseable = 0
        self.unparseable_reasons: dict[str, int] = {}

    def reset(self) -> None:
        self.n_unparseable = 0
        self.unparseable_reasons = {}

    def _cost(self, prompt_tokens: int, completion_tokens: int, calls: int,
              latency_ms: float) -> DiagnoserCostRecord:
        if self.usd_per_mtok is None:
            return DiagnoserCostRecord(prompt_tokens=prompt_tokens,
                                       completion_tokens=completion_tokens,
                                       calls=calls, latency_ms=latency_ms)
        usd = (prompt_tokens + completion_tokens) / 1e6 * float(self.usd_per_mtok)
        if not self.basis:
            raise ValueError("")
        if prompt_tokens + completion_tokens <= 0:
            raise ValueError(
                "")
        return DiagnoserCostRecord(prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens, calls=calls,
                                   latency_ms=latency_ms, usd=usd, basis=self.basis)

    def diagnose_or_none(self, obs: DiagnosisInput) -> tuple[DiagnosisOutput | None, dict[str, Any]]:
        sysp, usr = system_prompt(), user_prompt(obs)
        attempts = 0
        p_tok = c_tok = 0
        t0 = time.monotonic()
        last: ParseResult | None = None
        for _ in range(self.parse_retries + 1):
            attempts += 1
            resp = self.complete_fn(sysp, usr)
            p_tok += int(resp.get("prompt_tokens") or 0)
            c_tok += int(resp.get("completion_tokens") or 0)
            last = parse_response(resp.get("text"))
            if last.label is not None:
                break
        latency = (time.monotonic() - t0) * 1000.0
        if self.require_usage and p_tok + c_tok <= 0:
            raise ValueError(
                "")
        cost = self._cost(p_tok, c_tok, attempts, latency)
        assert last is not None
        if last.label is None:
            self.n_unparseable += 1
            self.unparseable_reasons[last.reason] = self.unparseable_reasons.get(last.reason, 0) + 1
            return None, {"reason": last.reason, "attempts": attempts, "raw_label": last.raw_label,
                          "cost": cost}
        return DiagnosisOutput(
            label=last.label, cost=cost,
            detail={"source": self.name, "attempts": attempts, "raw_label": last.raw_label},
        ), {"reason": "ok", "attempts": attempts, "cost": cost}

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        out, meta = self.diagnose_or_none(obs)
        if out is None:
            raise ValueError(
                "")
        return out


def make_local_vllm_complete_fn(client: Any, model: str, *, temperature: float,
                                top_p: float, top_k: int, min_p: float,
                                max_tokens: int) -> CompleteFn:
    def _complete(system: str, user: str) -> dict[str, Any]:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            extra_body={"top_k": top_k, "min_p": min_p},
        )
        usage = getattr(resp, "usage", None)
        choices = getattr(resp, "choices", None) or []
        choice = choices[0] if choices else None
        msg = getattr(choice, "message", None) if choice is not None else None
        reasoning = getattr(msg, "reasoning_content", None) if msg is not None else None
        return {"text": getattr(msg, "content", None) if msg is not None else None,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "finish_reason": getattr(choice, "finish_reason", None)
                if choice is not None else None,
                "reasoning_len": len(reasoning) if isinstance(reasoning, str) else 0,
                "n_choices": len(choices)}
    return _complete


def batch_request_body(obs: DiagnosisInput, *, model: str, max_tokens: int,
                       reasoning_effort: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt()},
                     {"role": "user", "content": user_prompt(obs)}],
        "max_completion_tokens": max_tokens}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def prompt_pair(obs: DiagnosisInput) -> Sequence[str]:
    return (system_prompt(), user_prompt(obs))
