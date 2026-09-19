#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dar.agents import CallLog, ClientProxy, StripStats


class _Rec:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = type("M", (), {"content": "ok", "reasoning_content": None,
                             "tool_calls": None})()
        return type("R", (), {
            "choices": [type("C", (), {"finish_reason": "stop", "message": msg})()],
            "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})(),
        })()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Rec()


THINK = {"role": "assistant",
         "content": "<think>\ndeliberation\n</think>ANSWER",
         "tool_calls": [{"id": "c0"}]}


def _fail(msg: str) -> None:
    print(f"[preflight-wiring] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    StripStats.reset()
    fake = _Client()
    proxy = ClientProxy(fake, timeout_s=60.0, chat_template_kwargs=None,
                        strip_history_thinking=True)
    if getattr(proxy.chat.completions, "_strip", None) is not True:
        _fail("")
    proxy.chat.completions.create(model="m", messages=[dict(THINK)])
    sent = fake.chat.completions.kwargs["messages"][0]["content"]
    if sent != "ANSWER":
        _fail("")
    if StripStats.chars_removed <= 0:
        _fail("")

    fake2 = _Client()
    proxy2 = ClientProxy(fake2, timeout_s=60.0, chat_template_kwargs=None)
    if getattr(proxy2.chat.completions, "_strip", None) is not False:
        _fail("")
    proxy2.chat.completions.create(model="m", messages=[dict(THINK)])
    if fake2.chat.completions.kwargs["messages"][0]["content"] != THINK["content"]:
        _fail("")

    CallLog.configure(None)
    if CallLog.enabled():
        _fail("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
