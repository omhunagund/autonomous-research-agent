from datetime import datetime, timezone

from src.core.evidence_compaction import (
    PROMPT_EVIDENCE_TOKEN_BUDGET,
    compact_sources_for_prompt,
)
from src.models.schemas import Source


def _source(
    citation_id: int,
    content: str,
    search_queries: list[str],
) -> Source:
    return Source(
        citation_id=citation_id,
        title=f"Source {citation_id}",
        url=f"https://example.com/{citation_id}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Example snippet.",
        search_queries=search_queries,
        content=content,
    )


def test_compaction_preserves_original_source_content():
    original = (
        "Generative AI improves software engineering productivity.\n\n"
        "It can assist developers with code generation and review."
    )

    source = _source(
        citation_id=1,
        content=original,
        search_queries=["generative AI software engineering"],
    )

    compacted = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
    )

    assert source.content == original
    assert compacted[0].content
    assert compacted[0].citation_id == source.citation_id


def test_compaction_preserves_source_metadata():
    source = _source(
        citation_id=7,
        content=(
            "Generative AI software engineering tools can assist with code generation."
        ),
        search_queries=["generative AI software engineering"],
    )

    compacted = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
    )

    result = compacted[0]

    assert result.citation_id == source.citation_id
    assert result.title == source.title
    assert result.url == source.url
    assert result.snippet == source.snippet
    assert result.search_queries == source.search_queries


def test_compaction_preserves_paragraph_boundaries_and_relevance():
    source = _source(
        citation_id=1,
        content=(
            "This paragraph discusses unrelated weather observations.\n\n"
            "Generative AI software engineering tools can assist with code generation.\n\n"
            "Another unrelated paragraph discusses cooking recipes."
        ),
        search_queries=["generative AI software engineering"],
    )

    compacted = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
    )

    assert (
        "Generative AI software engineering tools can assist with code generation."
        in compacted[0].content
    )


def test_compaction_is_deterministic():
    source = _source(
        citation_id=1,
        content=(
            "Generative AI helps software developers generate code.\n\n"
            "Software engineering teams also use AI for testing.\n\n"
            "Unrelated paragraph about gardening."
        ),
        search_queries=["generative AI software engineering"],
    )

    first = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
    )

    second = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
    )

    assert first == second


def test_compaction_respects_global_budget():
    sources = [
        _source(
            citation_id=index,
            content=("generative AI software engineering " * 20_000),
            search_queries=["generative AI software engineering"],
        )
        for index in range(1, 5)
    ]

    compacted = compact_sources_for_prompt(
        sources,
        ["generative AI software engineering"],
    )

    total_estimated_tokens = sum(
        (len(source.content) + 2) // 3
        for source in compacted
    )

    assert total_estimated_tokens <= PROMPT_EVIDENCE_TOKEN_BUDGET


def test_compaction_returns_no_sources_when_budget_is_zero():
    source = _source(
        citation_id=1,
        content="Generative AI software engineering evidence.",
        search_queries=["generative AI software engineering"],
    )

    compacted = compact_sources_for_prompt(
        [source],
        ["generative AI software engineering"],
        token_budget=0,
    )

    assert compacted == []