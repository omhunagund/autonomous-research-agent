from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from src.core.llm import (
    LLMInvocationError,
    LLMService,
    _is_retryable_error,
)


class FakeNotFoundError(Exception):
    status_code = 404
    code = "model_not_found"


class FakeAuthenticationError(Exception):
    status_code = 401
    code = "authentication_error"


def test_retryable_model_not_found_error():
    assert _is_retryable_error(FakeNotFoundError("model unavailable")) is True


def test_non_retryable_authentication_error():
    assert _is_retryable_error(FakeAuthenticationError("invalid API key")) is False


def test_primary_success_does_not_call_fallback():
    primary = Mock()
    fallback = Mock()

    expected_response = Mock()
    primary.invoke.return_value = expected_response

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
    )

    messages = [HumanMessage(content="test")]
    response = service.invoke(messages)

    assert response is expected_response
    primary.invoke.assert_called_once_with(messages)
    fallback.invoke.assert_not_called()


def test_retryable_primary_failure_uses_fallback():
    primary = Mock()
    fallback = Mock()

    primary_error = RuntimeError("service temporarily unavailable")
    primary.invoke.side_effect = primary_error

    expected_response = Mock()
    fallback.invoke.return_value = expected_response

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
    )

    messages = [HumanMessage(content="test")]
    response = service.invoke(messages)

    assert response is expected_response
    primary.invoke.assert_called_once_with(messages)
    fallback.invoke.assert_called_once_with(messages)


def test_non_retryable_primary_failure_does_not_use_fallback():
    primary = Mock()
    fallback = Mock()

    primary.invoke.side_effect = RuntimeError("invalid request")

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
    )

    messages = [HumanMessage(content="test")]

    with pytest.raises(RuntimeError, match="invalid request"):
        service.invoke(messages)

    fallback.invoke.assert_not_called()


def test_fallback_failure_raises_llm_invocation_error():
    primary = Mock()
    fallback = Mock()

    primary.invoke.side_effect = RuntimeError("timeout")
    fallback.invoke.side_effect = RuntimeError("fallback unavailable")

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
    )

    messages = [HumanMessage(content="test")]

    with pytest.raises(LLMInvocationError, match="Both primary and fallback"):
        service.invoke(messages)

    primary.invoke.assert_called_once_with(messages)
    fallback.invoke.assert_called_once_with(messages)