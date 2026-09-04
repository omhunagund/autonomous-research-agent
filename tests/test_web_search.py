from unittest.mock import patch

import pytest

from src.models.schemas import SearchResult
from src.tools.web_search import SearchError, web_search


def test_web_search_maps_ddgs_results():
    fake_results = [
        {
            "title": "Example Title",
            "href": "https://example.com",
            "body": "Example snippet",
        }
    ]

    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = fake_results

        results = web_search("example query", max_results=5)

    assert results == [
        SearchResult(
            title="Example Title",
            url="https://example.com",
            snippet="Example snippet",
        )
    ]

    mock_ddgs.return_value.text.assert_called_once_with(
        query="example query",
        max_results=5,
    )


def test_web_search_skips_malformed_results():
    fake_results = [
        {
            "title": "Valid",
            "href": "https://valid.example",
            "body": "Valid snippet",
        },
        {
            "title": "Missing URL",
            "body": "Invalid result",
        },
        {
            "href": "https://missing-title.example",
            "body": "Invalid result",
        },
    ]

    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = fake_results

        results = web_search("example query")

    assert len(results) == 1
    assert results[0].title == "Valid"


def test_web_search_returns_empty_list_for_zero_results():
    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = []

        results = web_search("example query")

    assert results == []


def test_web_search_raises_search_error_on_failure():
    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.side_effect = RuntimeError("search failed")

        with pytest.raises(SearchError, match="Web search failed"):
            web_search("example query")


def test_web_search_returns_empty_list_for_blank_query():
    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        results = web_search("   ")

    assert results == []
    mock_ddgs.assert_not_called()