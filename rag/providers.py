"""
rag/providers.py — the ONLY file that talks to a model provider.
================================================================

This is the keystone of the whole repo's design. RAG is an *architecture*, not a
provider feature: chunking, vector search, reranking, and evaluation don't care
who serves the model. So we hide the one provider-specific part — turning text
into vectors (`embed`) and turning a prompt into an answer (`generate`) — behind
two tiny functions. Everything else in `rag/` and `examples/` is pure RAG logic.

Pick your stack with `PROVIDER` in `.env`:

  PROVIDER=openai  ->  OpenAI embeddings + OpenAI chat
                       needs OPENAI_API_KEY
  PROVIDER=claude  ->  Voyage AI embeddings + Claude messages
                       needs ANTHROPIC_API_KEY and VOYAGE_API_KEY

If you've worked through the sibling repos (openai-api-deep-dive /
claude-api-deep-dive), this is exactly the embeddings + chat calls you already
know — just wrapped so the rest of the code can stay provider-agnostic.
"""

import os
from functools import lru_cache

# Default models per stack. These mirror the ones the sibling repos use.
_OPENAI_EMBED = "text-embedding-3-small"
_OPENAI_CHAT = "gpt-4o-mini"
_VOYAGE_EMBED = "voyage-3.5"
_CLAUDE_CHAT = "claude-haiku-4-5"

_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"],
}


def provider_name() -> str:
    """The active stack: 'openai' (default) or 'claude'. Set via PROVIDER in .env."""
    return os.getenv("PROVIDER", "openai").strip().lower()


def required_keys() -> list[str]:
    """Environment variables the active stack needs."""
    return _KEYS.get(provider_name(), [])


def describe() -> str:
    """One-line summary of the active stack — handy for examples to print."""
    p = provider_name()
    if p == "openai":
        return f"openai  (embed={_OPENAI_EMBED}, chat={_OPENAI_CHAT})"
    if p == "claude":
        return f"claude  (embed={_VOYAGE_EMBED} via Voyage, chat={_CLAUDE_CHAT})"
    return f"unknown provider {p!r}"


def ensure_ready() -> None:
    """Fail fast with a friendly message if the stack isn't configured.

    Call this at the top of any script *after* `load_dotenv()`. It's the
    provider-aware version of the `if not os.getenv(...)` guard in the sibling
    repos' examples.
    """
    import sys

    p = provider_name()
    if p not in _KEYS:
        sys.exit(
            f"PROVIDER={p!r} is not recognized. Set PROVIDER=openai or "
            f"PROVIDER=claude in .env."
        )
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in the environment. "
            f"Provide them via secrun (see SECRETS.md), or run `secrun python check_setup.py`."
        )


# --- Clients are created lazily and cached, so importing this module never
#     forces an SDK import or a network call. The SDK is only touched the first
#     time you actually embed or generate. ---


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


@lru_cache(maxsize=1)
def _voyage_client():
    import voyageai  # type: ignore[import-untyped]

    return voyageai.Client()  # pyright: ignore[reportPrivateImportUsage]


def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Turn a list of strings into a list of embedding vectors (one per string).

    `input_type` is "document" (the default, for things you're storing) or
    "query" (for a search query). Voyage uses this hint to optimize each side of
    retrieval and improve match quality; OpenAI ignores it. Using the same
    parameter for both keeps the rest of the code provider-agnostic.
    """
    if not texts:
        return []
    p = provider_name()
    if p == "openai":
        resp = _openai_client().embeddings.create(model=_OPENAI_EMBED, input=list(texts))
        return [item.embedding for item in resp.data]
    if p == "claude":
        result = _voyage_client().embed(
            list(texts), model=_VOYAGE_EMBED, input_type=input_type
        )
        return result.embeddings
    raise ValueError(f"Unknown PROVIDER={p!r} (expected 'openai' or 'claude').")


def generate(system: str, user: str, max_tokens: int = 512) -> str:
    """Turn a (system, user) prompt into a text answer.

    The two providers differ in shape — OpenAI puts the system prompt in the
    messages list and returns `choices[0].message.content`; Claude takes a
    top-level `system=` and returns a list of content blocks — so we normalize
    both to a plain string here.
    """
    p = provider_name()
    if p == "openai":
        resp = _openai_client().chat.completions.create(
            model=_OPENAI_CHAT,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    raise ValueError(f"Unknown PROVIDER={p!r} (expected 'openai' or 'claude').")
