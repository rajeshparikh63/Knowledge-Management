"""
Thin LLM + embedding helpers for the KG revamp package.

Self-contained so the new `kg` package doesn't depend on the old
graphrag_client. Embeddings via OpenAI, chat/JSON via OpenRouter — same config
the rest of the app uses.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, List

import numpy as np
from openai import AsyncOpenAI

from app.logger import logger
from app.settings import settings

_EMBED_BATCH = 1024  # OpenAI caps input arrays; batch defensively


def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _openrouter() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings, preserving order. One request per <=1024 items."""
    if not texts:
        return []
    client = _openai()
    if len(texts) <= _EMBED_BATCH:
        resp = await client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    batches = [texts[i:i + _EMBED_BATCH] for i in range(0, len(texts), _EMBED_BATCH)]
    responses = await asyncio.gather(*[
        client.embeddings.create(model=settings.EMBEDDING_MODEL, input=b) for b in batches
    ])
    out: List[List[float]] = []
    for resp in responses:
        out.extend(d.embedding for d in resp.data)
    return out


def cosine_distance(a: List[float], b: List[float]) -> float:
    """1 - cosine similarity. 0 = identical, higher = more different."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def parse_json(text: str) -> Any:
    """Best-effort parse of an LLM JSON reply (tolerates fences / surrounding prose)."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Grab the first {...} or [...] block.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM reply: {text[:200]!r}")


async def llm_json(prompt: str, model: str | None = None, max_tokens: int = 2048) -> Any:
    """Call the LLM and parse its reply as JSON."""
    client = _openrouter()
    resp = await client.chat.completions.create(
        model=model or settings.ONTOLOGY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return parse_json(content)


async def llm_yes_no(prompt: str, model: str | None = None) -> bool:
    """Ask a yes/no question; return True for yes."""
    client = _openrouter()
    resp = await client.chat.completions.create(
        model=model or settings.ONTOLOGY_MODEL,
        messages=[{"role": "user", "content": prompt + "\n\nAnswer with only 'yes' or 'no'."}],
        temperature=0,
        max_tokens=5,
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    return ans.startswith("y")
