"""Deterministic source-evidence compaction for LLM prompts."""

from __future__ import annotations

import re
from math import ceil

from src.models.schemas import Source


# Each LLM request gets its own evidence budget.
PROMPT_EVIDENCE_TOKEN_BUDGET = 4_000

# Conservative hard-guardrail estimate.
# Deliberately pessimistic so the selected evidence stays well below
# the observed provider-side 8,000 TPM ceiling.
CHARS_PER_TOKEN = 3


def _estimate_tokens(text: str) -> int:
    """Conservatively estimate token usage from text length."""
    if not text:
        return 0

    return ceil(len(text) / CHARS_PER_TOKEN)


def _normalize_tokens(text: str) -> list[str]:
    """Return normalized word-like tokens for lexical relevance scoring."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _paragraphs(content: str) -> list[str]:
    """Split content into non-empty structural paragraphs."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", content)
        if paragraph.strip()
    ]


def _paragraph_score(paragraph: str, queries: list[str]) -> int:
    """
    Score a paragraph by lexical overlap with query terms.

    The score is deterministic and does not use an LLM.
    """
    paragraph_tokens = set(_normalize_tokens(paragraph))

    query_tokens: set[str] = set()
    for query in queries:
        query_tokens.update(_normalize_tokens(query))

    return len(paragraph_tokens & query_tokens)


def _compact_source(
    source: Source,
    fallback_queries: list[str],
    remaining_tokens: int,
) -> tuple[str, int]:
    """Select the most relevant complete paragraphs within the remaining budget."""
    paragraphs = _paragraphs(source.content)

    if not paragraphs or remaining_tokens <= 0:
        return "", 0

    queries = source.search_queries or fallback_queries

    scored = [
        (
            _paragraph_score(paragraph, queries),
            index,
            paragraph,
            _estimate_tokens(paragraph),
        )
        for index, paragraph in enumerate(paragraphs)
    ]

    # Highest relevance first; original document order breaks ties.
    ranked = sorted(
        scored,
        key=lambda item: (-item[0], item[1]),
    )

    selected: list[tuple[int, str]] = []
    used_tokens = 0

    for _, index, paragraph, paragraph_tokens in ranked:
        if paragraph_tokens > remaining_tokens - used_tokens:
            continue

        selected.append((index, paragraph))
        used_tokens += paragraph_tokens

        if used_tokens >= remaining_tokens:
            break

    # Restore the source's original paragraph order.
    selected.sort(key=lambda item: item[0])

    return "\n\n".join(paragraph for _, paragraph in selected), used_tokens


def compact_sources_for_prompt(
    sources: list[Source],
    queries: list[str],
    token_budget: int = PROMPT_EVIDENCE_TOKEN_BUDGET,
) -> list[Source]:
    """
    Return prompt-facing copies of sources within a global evidence budget.

    Canonical Source objects are never mutated.

    Source metadata remains unchanged. Only the content field is compacted.

    Paragraphs are selected deterministically using lexical overlap between
    each paragraph and the source's search queries, falling back to the
    supplied queries when a source has no search-query metadata.
    """
    if not sources or token_budget <= 0:
        return []

    remaining_tokens = token_budget
    compacted: list[Source] = []

    for source in sources:
        if remaining_tokens <= 0:
            break

        compacted_content, used_tokens = _compact_source(
            source,
            queries,
            remaining_tokens,
        )

        if not compacted_content:
            continue

        compacted.append(
            source.model_copy(
                update={"content": compacted_content},
                deep=True,
            )
        )

        remaining_tokens -= used_tokens

    return compacted