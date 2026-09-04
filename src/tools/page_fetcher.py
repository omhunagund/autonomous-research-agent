import re

import httpx
import trafilatura
from bs4 import BeautifulSoup


class PageFetchError(RuntimeError):
    """Raised when a webpage cannot be fetched or extracted successfully."""


REQUEST_TIMEOUT = 15.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/142.0.0.0 Safari/537.36"
)

MIN_CONTENT_WORDS = 200


def _count_words(text: str) -> int:
    """Count whitespace-separated words after normalizing whitespace."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return len(normalized.split())


def _extract_with_trafilatura(html: str) -> str | None:
    """Extract main textual content using Trafilatura."""
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )

    if extracted is None:
        return None

    extracted = extracted.strip()

    if not extracted:
        return None

    return extracted


def _extract_with_beautifulsoup(html: str) -> str | None:
    """Fallback extraction using visible page text from BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]
    ):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text or None


def fetch_page_content(url: str) -> str:
    """
    Fetch a webpage and return validated meaningful textual content.

    Extraction order:
    1. Trafilatura
    2. BeautifulSoup only when Trafilatura returns no text

    A successful extraction must contain at least 200 words.

    Raises:
        PageFetchError: If the request fails, extraction fails, or the
            resulting content contains fewer than 200 words.
    """

    if not url.strip():
        raise PageFetchError("Page URL is empty.")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PageFetchError(
            f"HTTP error while fetching page: {url} "
            f"(status {exc.response.status_code})."
        ) from exc
    except httpx.RequestError as exc:
        raise PageFetchError(
            f"Request failed while fetching page: {url}."
        ) from exc

    content_type = response.headers.get("content-type", "").lower()

    if content_type and not (
        "text/html" in content_type
        or "application/xhtml+xml" in content_type
    ):
        raise PageFetchError(
            f"Unsupported content type for page: {content_type}."
        )

    html = response.text

    extracted = _extract_with_trafilatura(html)

    if extracted is not None:
        if _count_words(extracted) < MIN_CONTENT_WORDS:
            raise PageFetchError(
                f"Trafilatura extracted fewer than "
                f"{MIN_CONTENT_WORDS} words."
            )

        return extracted

    extracted = _extract_with_beautifulsoup(html)

    if extracted is None or _count_words(extracted) < MIN_CONTENT_WORDS:
        raise PageFetchError(
            f"Page content contains fewer than "
            f"{MIN_CONTENT_WORDS} meaningful words."
        )

    return extracted