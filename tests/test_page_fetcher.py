from unittest.mock import Mock, patch

import pytest

from src.tools.page_fetcher import (
    PageFetchError,
    fetch_page_content,
)


VALID_TEXT = " ".join(["meaningful content"] * 250)


def test_fetch_page_content_uses_trafilatura_when_extraction_is_sufficient():
    fake_response = Mock()
    fake_response.headers = {"content-type": "text/html"}
    fake_response.text = "<html>example</html>"
    fake_response.raise_for_status.return_value = None

    with (
        patch("src.tools.page_fetcher.httpx.Client") as mock_client,
        patch(
            "src.tools.page_fetcher.trafilatura.extract",
            return_value=VALID_TEXT,
        ),
        patch(
            "src.tools.page_fetcher._extract_with_beautifulsoup"
        ) as mock_bs,
    ):
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        result = fetch_page_content("https://example.com")

    assert result == VALID_TEXT
    mock_bs.assert_not_called()


def test_short_trafilatura_result_is_not_rescued_by_beautifulsoup():
    short_text = " ".join(["short"] * 199)

    fake_response = Mock()
    fake_response.headers = {"content-type": "text/html"}
    fake_response.text = "<html>example</html>"
    fake_response.raise_for_status.return_value = None

    with (
        patch("src.tools.page_fetcher.httpx.Client") as mock_client,
        patch(
            "src.tools.page_fetcher.trafilatura.extract",
            return_value=short_text,
        ),
        patch(
            "src.tools.page_fetcher._extract_with_beautifulsoup",
            return_value=VALID_TEXT,
        ) as mock_bs,
    ):
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        with pytest.raises(PageFetchError, match="Trafilatura extracted fewer"):
            fetch_page_content("https://example.com")

    mock_bs.assert_not_called()


def test_beautifulsoup_is_used_when_trafilatura_returns_nothing():
    fake_response = Mock()
    fake_response.headers = {"content-type": "text/html"}
    fake_response.text = "<html>example</html>"
    fake_response.raise_for_status.return_value = None

    with (
        patch("src.tools.page_fetcher.httpx.Client") as mock_client,
        patch(
            "src.tools.page_fetcher.trafilatura.extract",
            return_value=None,
        ),
        patch(
            "src.tools.page_fetcher._extract_with_beautifulsoup",
            return_value=VALID_TEXT,
        ),
    ):
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        result = fetch_page_content("https://example.com")

    assert result == VALID_TEXT


def test_beautifulsoup_short_result_raises_error():
    short_text = " ".join(["short"] * 199)

    fake_response = Mock()
    fake_response.headers = {"content-type": "text/html"}
    fake_response.text = "<html>example</html>"
    fake_response.raise_for_status.return_value = None

    with (
        patch("src.tools.page_fetcher.httpx.Client") as mock_client,
        patch(
            "src.tools.page_fetcher.trafilatura.extract",
            return_value=None,
        ),
        patch(
            "src.tools.page_fetcher._extract_with_beautifulsoup",
            return_value=short_text,
        ),
    ):
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        with pytest.raises(PageFetchError, match="fewer than 200"):
            fetch_page_content("https://example.com")


def test_http_status_error_is_converted_to_page_fetch_error():
    fake_response = Mock()
    fake_response.raise_for_status.side_effect = (
        __import__("httpx").HTTPStatusError(
            "404",
            request=Mock(),
            response=Mock(status_code=404),
        )
    )

    with patch("src.tools.page_fetcher.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        with pytest.raises(PageFetchError, match="HTTP error"):
            fetch_page_content("https://example.com")


def test_request_error_is_converted_to_page_fetch_error():
    import httpx

    with patch("src.tools.page_fetcher.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = (
            httpx.ConnectError("connection failed")
        )

        with pytest.raises(PageFetchError, match="Request failed"):
            fetch_page_content("https://example.com")


def test_non_html_content_is_rejected():
    fake_response = Mock()
    fake_response.headers = {"content-type": "application/pdf"}
    fake_response.text = "not html"
    fake_response.raise_for_status.return_value = None

    with patch("src.tools.page_fetcher.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = (
            fake_response
        )

        with pytest.raises(PageFetchError, match="Unsupported content type"):
            fetch_page_content("https://example.com")