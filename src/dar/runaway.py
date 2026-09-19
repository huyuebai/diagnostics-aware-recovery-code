from __future__ import annotations

import re

LOOP_TAIL_CHARS = 3000
LOOP_NGRAM = 12
LOOP_DUP_THRESHOLD = 0.40


def dup_rate(raw: str) -> float:
    if not isinstance(raw, str):
        return 0.0
    toks = re.findall(r"\S+", raw[-LOOP_TAIL_CHARS:])
    if len(toks) < LOOP_NGRAM + 1:
        return 0.0
    grams = [tuple(toks[i:i + LOOP_NGRAM]) for i in range(len(toks) - LOOP_NGRAM + 1)]
    return 1 - len(set(grams)) / len(grams)


def is_runaway(raw: str) -> bool:
    return dup_rate(raw) >= LOOP_DUP_THRESHOLD
