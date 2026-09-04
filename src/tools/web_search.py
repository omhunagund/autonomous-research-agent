from ddgs import DDGS

from src.models.schemas import SearchResult


class SearchError(RuntimeError):
    """Raised when a web search cannot be completed."""


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Search the web and return normalized search results.

    A successful search with no matches returns an empty list.

    Malformed individual results are skipped.

    A search-level failure raises SearchError so that the caller
    can distinguish a failed search from a valid zero-result search.
    """

    if not query.strip():
        return []

    try:
        raw_results = DDGS().text(
            query=query,
            max_results=max_results,
        )
    except Exception as exc:
        raise SearchError(
            f"Web search failed for query: {query!r}"
        ) from exc

    results: list[SearchResult] = []

    for item in raw_results:
        title = item.get("title")
        url = item.get("href")
        snippet = item.get("body")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (title, url, snippet)
        ):
            continue

        try:
            results.append(
                SearchResult(
                    title=title.strip(),
                    url=url.strip(),
                    snippet=snippet.strip(),
                )
            )
        except Exception:
            continue

    return results