from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.core.llm import (
    LLMConfigurationError,
    LLMInvocationError,
    LLMService,
    _get_required_env,
    _is_retryable_error,
)


class ExampleStructuredOutput(BaseModel):
    answer: str


class FakeNotFoundError(Exception):
    status_code = 404
    code = "model_not_found"


class FakeAuthenticationError(Exception):
    status_code = 401
    code = "authentication_error"

class FakeTimeoutError(Exception):
    status_code = 408

class FakeConflictError(Exception):
    status_code = 409

class FakeStructuredOutputError(Exception):
    status_code = 400

    def __init__(self, code: str, failed_generation: str):
        super().__init__(f"Groq structured output error: {code}")
        self.code = code
        self.body = {
            "error": {
                "code": code,
                "failed_generation": failed_generation,
            }
        }


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------


def test_retryable_model_not_found_error():
    assert _is_retryable_error(
        FakeNotFoundError("model unavailable")
    ) is True


def test_non_retryable_authentication_error():
    assert _is_retryable_error(
        FakeAuthenticationError("invalid API key")
    ) is False

def test_timeout_error_is_retryable():
    assert _is_retryable_error(
        FakeTimeoutError("request timed out")
    ) is True


def test_conflict_error_is_not_retryable():
    assert _is_retryable_error(
        FakeConflictError("request conflict")
    ) is False


def test_required_env_rejects_missing_or_empty_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_REQUIRED_ENV", raising=False)

    with pytest.raises(LLMConfigurationError, match="missing or empty"):
        _get_required_env("TEST_REQUIRED_ENV")

    monkeypatch.setenv("TEST_REQUIRED_ENV", "   ")

    with pytest.raises(LLMConfigurationError, match="missing or empty"):
        _get_required_env("TEST_REQUIRED_ENV")


def test_required_env_strips_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REQUIRED_ENV", "  configured-value  ")

    assert _get_required_env("TEST_REQUIRED_ENV") == "configured-value"


# ---------------------------------------------------------------------------
# Normal invocation tests
# ---------------------------------------------------------------------------


def test_primary_success_does_not_call_fallback():
    primary = Mock()
    fallback = Mock()

    expected_response = Mock()
    primary.invoke.return_value = expected_response

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
        groq_client=Mock(),
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]
    response = service.invoke(messages)

    assert response is expected_response
    primary.invoke.assert_called_once_with(messages)
    fallback.invoke.assert_not_called()


def test_retryable_primary_failure_uses_fallback():
    primary = Mock()
    fallback = Mock()

    primary.invoke.side_effect = RuntimeError(
        "service temporarily unavailable"
    )

    expected_response = Mock()
    fallback.invoke.return_value = expected_response

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
        groq_client=Mock(),
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
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
        groq_client=Mock(),
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
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
        groq_client=Mock(),
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    with pytest.raises(
        LLMInvocationError,
        match="Both primary and fallback",
    ):
        service.invoke(messages)

    primary.invoke.assert_called_once_with(messages)
    fallback.invoke.assert_called_once_with(messages)

def test_normal_invocation_does_not_set_structured_reasoning_effort():
    primary = Mock()
    fallback = Mock()

    primary.invoke.return_value = Mock()

    service = LLMService(
        primary_model=primary,
        fallback_model=fallback,
        groq_client=Mock(),
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]
    service.invoke(messages)

    primary.invoke.assert_called_once_with(messages)


# ---------------------------------------------------------------------------
# Native structured-output tests
# ---------------------------------------------------------------------------


def test_structured_primary_success():
    groq_client = Mock()

    response = Mock()
    response.choices = [
        Mock(
            message=Mock(
                content='{"answer": "primary response"}'
            )
        )
    ]

    groq_client.chat.completions.create.return_value = response

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    result = service.invoke_structured(
        messages,
        ExampleStructuredOutput,
    )

    assert result == ExampleStructuredOutput(
        answer="primary response"
    )

    groq_client.chat.completions.create.assert_called_once()

    call_kwargs = groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"


def test_structured_retryable_primary_failure_uses_fallback():
    groq_client = Mock()

    primary_error = RuntimeError(
        "service temporarily unavailable"
    )

    fallback_response = Mock()
    fallback_response.choices = [
        Mock(
            message=Mock(
                content='{"answer": "fallback response"}'
            )
        )
    ]

    groq_client.chat.completions.create.side_effect = [
        primary_error,
        fallback_response,
    ]

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    result = service.invoke_structured(
        messages,
        ExampleStructuredOutput,
    )

    assert result == ExampleStructuredOutput(
        answer="fallback response"
    )

    assert groq_client.chat.completions.create.call_count == 2


def test_structured_non_retryable_failure_does_not_use_fallback():
    groq_client = Mock()

    groq_client.chat.completions.create.side_effect = RuntimeError(
        "invalid request"
    )

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    with pytest.raises(RuntimeError, match="invalid request"):
        service.invoke_structured(
            messages,
            ExampleStructuredOutput,
        )

    groq_client.chat.completions.create.assert_called_once()


def test_structured_invalid_json_raises_validation_error():
    groq_client = Mock()

    response = Mock()
    response.choices = [
        Mock(
            message=Mock(
                content="not valid json"
            )
        )
    ]

    groq_client.chat.completions.create.return_value = response

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    with pytest.raises(
        ValueError,
        match="could not be validated",
    ):
        service.invoke_structured(
            messages,
            ExampleStructuredOutput,
        )

@pytest.mark.parametrize(
    "error_code,failed_generation",
    [
        (
            "output_parse_failed",
            '{"answer": "recovered response"}',
        ),
        (
            "json_validate_failed",
            '{"answer": "recovered response"}',
        ),
        (
            "tool_use_failed",
            (
                '{"name": "examplestructuredoutput", '
                '"arguments": {"answer": "recovered response"}}'
            ),
        ),
    ],
)
def test_structured_protocol_errors_recover_valid_json(
    error_code: str,
    failed_generation: str,
):
    groq_client = Mock()

    groq_client.chat.completions.create.side_effect = [
        FakeStructuredOutputError(
            error_code,
            failed_generation,
        )
    ]

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    result = service.invoke_structured(
        messages,
        ExampleStructuredOutput,
    )

    assert result == ExampleStructuredOutput(
        answer="recovered response"
    )

    groq_client.chat.completions.create.assert_called_once()

def test_structured_protocol_error_with_unusable_generation_uses_fallback():
    groq_client = Mock()

    primary_error = FakeStructuredOutputError(
        "tool_use_failed",
        '{"name": "examplestructuredoutput", "arguments": {"wrong": 123}}',
    )

    fallback_response = Mock()
    fallback_response.choices = [
        Mock(
            message=Mock(
                content='{"answer": "fallback response"}'
            )
        )
    ]

    groq_client.chat.completions.create.side_effect = [
        primary_error,
        fallback_response,
    ]

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    result = service.invoke_structured(
        messages,
        ExampleStructuredOutput,
    )

    assert result == ExampleStructuredOutput(
        answer="fallback response"
    )

    assert groq_client.chat.completions.create.call_count == 2

def test_structured_tool_use_failure_recovers_critique_style_arguments():
    groq_client = Mock()

    primary_error = FakeStructuredOutputError(
        "tool_use_failed",
        (
            '{"name": "critiqueassessment", "arguments": '
            '{"answer": "recovered response"}}'
        ),
    )

    groq_client.chat.completions.create.side_effect = [
        primary_error,
    ]

    service = LLMService(
        primary_model=Mock(),
        fallback_model=Mock(),
        groq_client=groq_client,
        primary_model_name="primary-model",
        fallback_model_name="fallback-model",
    )

    messages = [HumanMessage(content="test")]

    result = service.invoke_structured(
        messages,
        ExampleStructuredOutput,
    )

    assert result == ExampleStructuredOutput(
        answer="recovered response"
    )

    groq_client.chat.completions.create.assert_called_once()