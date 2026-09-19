from __future__ import annotations

import hashlib
import itertools
import json
import threading
from pathlib import Path
from typing import Any


def derive_crn_seed(task_id: str, turn: int) -> int:
    h = hashlib.sha256(f"{task_id}|{turn}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31)

_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"


def strip_thinking(content: Any) -> tuple[Any, bool]:
    if not isinstance(content, str) or _THINK_OPEN not in content:
        return content, False
    idx = content.rfind(_THINK_CLOSE)
    stripped = content[idx + len(_THINK_CLOSE):] if idx != -1 else ""
    return (stripped, True) if stripped != content else (content, False)


def strip_history_thinking_from(messages: list[Any]) -> tuple[list[Any], int]:
    out: list[Any] = []
    n = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            out.append(m)
            continue
        new_content, changed = strip_thinking(m.get("content"))
        has_reasoning = "reasoning_content" in m
        if changed and not str(new_content).strip() and not m.get("tool_calls"):
            if has_reasoning:
                m2 = dict(m)
                m2.pop("reasoning_content", None)
                out.append(m2)
                n += 1
            else:
                out.append(m)
            continue
        if not changed and not has_reasoning:
            out.append(m)
            continue
        m2 = dict(m)
        if changed:
            m2["content"] = new_content
        m2.pop("reasoning_content", None)
        out.append(m2)
        n += 1
    return out, n


def writeback_reasoning_to_thought(action: Any, messages: list[Any]) -> Any:
    if not messages:
        return action
    last = messages[-1]
    reasoning = last.get("reasoning_content") if isinstance(last, dict) else None
    if not reasoning:
        return action
    existing = getattr(action, "thought", None) or ""
    action.thought = reasoning if not str(existing).strip() else f"{reasoning}\n{existing}"
    return action


class StripStats:
    calls_with_strip = 0
    messages_stripped = 0
    chars_removed = 0
    _lock = threading.Lock()

    @classmethod
    def add(cls, n_messages: int, chars: int) -> None:
        with cls._lock:
            cls.calls_with_strip += 1
            cls.messages_stripped += n_messages
            cls.chars_removed += chars

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.calls_with_strip = 0
            cls.messages_stripped = 0
            cls.chars_removed = 0


class CallLog:
    _path: str | None = None
    _lock = threading.Lock()
    n_written = 0
    n_write_errors = 0

    @classmethod
    def configure(cls, path: str | None) -> None:
        cls._path = path
        cls.n_written = 0
        cls.n_write_errors = 0
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            open(path, "w", encoding="utf-8").close()

    @classmethod
    def enabled(cls) -> bool:
        return bool(cls._path)

    @classmethod
    def record(cls, response: Any, n_messages_sent: int, origin: str | None = None,
               seed: int | None = None) -> None:
        if not cls._path:
            return
        rec: dict[str, Any] = {"n_messages_sent": n_messages_sent, "origin": origin, "seed": seed}
        try:
            u = getattr(response, "usage", None)
            if u is not None:
                rec["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                rec["completion_tokens"] = getattr(u, "completion_tokens", None)
            choices = getattr(response, "choices", None) or []
            if choices:
                ch = choices[0]
                rec["finish_reason"] = getattr(ch, "finish_reason", None)
                msg = getattr(ch, "message", None)
                content = getattr(msg, "content", None) if msg else None
                reasoning = getattr(msg, "reasoning_content", None) if msg else None
                rec["content_chars"] = len(content) if isinstance(content, str) else 0
                rec["reasoning_chars"] = len(reasoning) if isinstance(reasoning, str) else 0
                from dar.runaway import dup_rate
                rec["reasoning_dup_rate"] = (dup_rate(reasoning)
                                             if isinstance(reasoning, str) else 0.0)
                rec["has_tool_calls"] = bool(getattr(msg, "tool_calls", None)) if msg else False
        except Exception as e:
            rec["record_error"] = repr(e)
        try:
            line = json.dumps(rec, ensure_ascii=False, default=repr)
            with cls._lock:
                with open(cls._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                cls.n_written += 1
        except Exception:
            with cls._lock:
                cls.n_write_errors += 1


class _CompletionsProxy:
    _origin_counter = itertools.count()

    def __init__(self, real_completions: Any, timeout_s: float,
                 chat_template_kwargs: dict | None, strip_history_thinking: bool = False,
                 sampling: dict | None = None, seed_mode: str | None = None):
        self._real = real_completions
        self._timeout_s = timeout_s
        self._chat_template_kwargs = chat_template_kwargs
        self._strip = strip_history_thinking
        self._sampling = sampling
        self._seed_mode = seed_mode
        self._crn_task_id: str | None = None
        self._crn_turn = 0
        self._origin = f"o{next(self._origin_counter)}"

    def set_crn_task(self, task_id: str) -> None:
        self._crn_task_id = task_id
        self._crn_turn = 0

    def create(self, **kwargs: Any) -> Any:
        kwargs["timeout"] = self._timeout_s
        if self._chat_template_kwargs is not None:
            extra = dict(kwargs.get("extra_body") or {})
            merged = dict(extra.get("chat_template_kwargs") or {})
            merged.update(self._chat_template_kwargs)
            extra["chat_template_kwargs"] = merged
            kwargs["extra_body"] = extra
        if self._sampling:
            if self._sampling.get("top_p") is not None:
                kwargs["top_p"] = self._sampling["top_p"]
            eb = dict(kwargs.get("extra_body") or {})
            for k in ("top_k", "min_p"):
                if self._sampling.get(k) is not None:
                    eb[k] = self._sampling[k]
            if eb:
                kwargs["extra_body"] = eb
        seed = None
        if self._seed_mode == "crn" and self._crn_task_id is not None:
            seed = derive_crn_seed(self._crn_task_id, self._crn_turn)
            kwargs["seed"] = seed
            self._crn_turn += 1
        if self._strip and kwargs.get("messages"):
            before = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                         for m in kwargs["messages"]
                         if isinstance(m, dict) and m.get("role") == "assistant")
            new_msgs, n = strip_history_thinking_from(kwargs["messages"])
            after = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                        for m in new_msgs
                        if isinstance(m, dict) and m.get("role") == "assistant")
            kwargs["messages"] = new_msgs
            if n:
                StripStats.add(n, max(0, before - after))
        resp = self._real.create(**kwargs)
        CallLog.record(resp, len(kwargs.get("messages") or []), origin=self._origin, seed=seed)
        return resp

    def __getattr__(self, name: str) -> Any:
        real = self.__dict__.get("_real")
        if real is None:
            raise AttributeError(name)
        return getattr(real, name)


class _ChatProxy:
    def __init__(self, real_chat: Any, timeout_s: float,
                 chat_template_kwargs: dict | None, strip_history_thinking: bool = False,
                 sampling: dict | None = None, seed_mode: str | None = None):
        self.completions = _CompletionsProxy(real_chat.completions, timeout_s,
                                             chat_template_kwargs, strip_history_thinking,
                                             sampling, seed_mode)
        self._real = real_chat

    def __getattr__(self, name: str) -> Any:
        real = self.__dict__.get("_real")
        if real is None:
            raise AttributeError(name)
        return getattr(real, name)


class ClientProxy:
    def __init__(self, real_client: Any, timeout_s: float,
                 chat_template_kwargs: dict | None, strip_history_thinking: bool = False,
                 sampling: dict | None = None, seed_mode: str | None = None):
        self.chat = _ChatProxy(real_client.chat, timeout_s, chat_template_kwargs,
                               strip_history_thinking, sampling, seed_mode)
        self._real = real_client
        self.dar_timeout_s = timeout_s
        self.dar_chat_template_kwargs = chat_template_kwargs
        self.dar_strip_history_thinking = strip_history_thinking
        self.dar_sampling = sampling
        self.dar_seed_mode = seed_mode

    def __getattr__(self, name: str) -> Any:
        real = self.__dict__.get("_real")
        if real is None:
            raise AttributeError(name)
        return getattr(real, name)


def _make_dar_vllm_agent_class():
    from evaluation.agents.vllm_agent import VLLMAgent

    class DarVLLMAgent(VLLMAgent):
        dar_timeout_s: float = 300.0
        dar_enable_thinking: bool | None = None
        dar_strip_history_thinking: bool = False
        dar_writeback_reasoning: bool = False
        dar_force_prompt: str | None = None
        dar_sampling: dict | None = None
        dar_seed_mode: str | None = None

        @classmethod
        def configure(cls, *, timeout_s: float | None = None,
                      enable_thinking: bool | None = ...,
                      strip_history_thinking: bool | None = None,
                      writeback_reasoning: bool | None = None,
                      force_prompt: str | None = ...,
                      sampling: dict | None = ...,
                      seed_mode: str | None = ...) -> None:
            if timeout_s is not None:
                cls.dar_timeout_s = float(timeout_s)
            if enable_thinking is not ...:
                cls.dar_enable_thinking = enable_thinking
            if strip_history_thinking is not None:
                cls.dar_strip_history_thinking = bool(strip_history_thinking)
            if writeback_reasoning is not None:
                cls.dar_writeback_reasoning = bool(writeback_reasoning)
            if force_prompt is not ...:
                if force_prompt not in (None, "fault_aware", "p0"):
                    raise ValueError("")
                cls.dar_force_prompt = force_prompt
            if sampling is not ...:
                cls.dar_sampling = sampling
            if seed_mode is not ...:
                if seed_mode not in (None, "crn"):
                    raise ValueError("")
                cls.dar_seed_mode = seed_mode

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            ctk = (None if self.dar_enable_thinking is None
                   else {"enable_thinking": self.dar_enable_thinking})
            self.client = ClientProxy(self.client, self.dar_timeout_s, ctk,
                                      self.dar_strip_history_thinking,
                                      self.dar_sampling, self.dar_seed_mode)

        def set_crn_task_id(self, task_id: str) -> None:
            self.client.chat.completions.set_crn_task(task_id)

        def step(self, user_message: Any = None) -> Any:
            action = super().step(user_message)
            if self.dar_writeback_reasoning:
                return writeback_reasoning_to_thought(action, self.messages)
            return action

        def _build_system_message(self) -> str:
            if self.dar_force_prompt == "fault_aware":
                return self._build_perturbation_system_message()
            if self.dar_force_prompt == "p0":
                return self._build_p0_system_message()
            return super()._build_system_message()

    return DarVLLMAgent


_agent_class = None


def get_dar_vllm_agent_class():
    global _agent_class
    if _agent_class is None:
        _agent_class = _make_dar_vllm_agent_class()
    return _agent_class
