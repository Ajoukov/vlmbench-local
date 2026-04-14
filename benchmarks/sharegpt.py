"""ShareGPT conversation benchmark."""

import json

from dataloaders import LocalDataset
from dataloaders.sharegpt_dataset import ShareGPTDataset
from src.benchmark import Benchmark
from tasks.chatbot import ChatBot


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _parse_conv(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _normalize_messages(conv):
    messages = []

    for m in conv:
        if not isinstance(m, dict):
            continue

        if "human" in m:
            content = _to_text(m.get("human")).strip()
            if content:
                messages.append({"role": "user", "content": content})

        if "assistant" in m:
            content = _to_text(m.get("assistant")).strip()
            if content:
                messages.append({"role": "assistant", "content": content})

        if "human" in m or "assistant" in m:
            continue

        role = _to_text(m.get("role") or m.get("from")).strip().lower()
        if role in ("human", "user"):
            role = "user"
        elif role in ("assistant", "gpt", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue

        content = _to_text(m.get("content") or m.get("value")).strip()
        if content:
            messages.append({"role": role, "content": content})

    return messages


def _chat_input(entry, *conversation_keys):
    conv = []
    for key in conversation_keys:
        conv = _parse_conv(entry.get(key))
        if conv:
            break

    if not conv:
        return "", {}

    messages = _normalize_messages(conv)
    if not messages:
        return "", {}

    opts = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.95,
    }
    return "", opts


class ShareGPTBenchmark(Benchmark):
    """ShareGPT: conversational prompt/response pairs."""

    def build_input(self, entry):
        return _chat_input(entry, "conversations", "messages")

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "ShareGPTBenchmark":
        dataset = ShareGPTDataset(cache_dir)
        task = ChatBot(model=model)
        return cls(dataset, task)


class LocalShareGPTBenchmark(Benchmark):
    """ShareGPT: conversational prompt/response pairs."""

    def build_input(self, entry):
        return _chat_input(entry, "conversation", "messages")

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LocalShareGPTBenchmark":
        dataset = LocalDataset("sharegpt.csv", cache_dir)
        task = ChatBot(model=model)
        return cls(dataset, task)
