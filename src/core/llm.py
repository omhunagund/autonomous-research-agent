import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_groq import ChatGroq


load_dotenv()


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing or invalid."""


class LLMInvocationError(RuntimeError):
    """Raised when both primary and fallback models fail."""


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise LLMConfigurationError(
            f"Required environment variable '{name}' is missing or empty."
        )

    return value


def _is_retryable_error(exc: Exception) -> bool:
    """
    Return True when a failure should trigger the fallback model.

    Retryable failures:
    - unavailable / unsupported model
    - rate limits
    - request timeouts
    - network/connection failures
    - transient server/API failures

    Non-retryable failures:
    - authentication/configuration errors
    - malformed requests
    - application-side programming errors
    """

    status_code = getattr(exc, "status_code", None)

    if status_code in {408, 409, 429}:
        return True

    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True

    exception_name = type(exc).__name__.lower()
    exception_message = str(exc).lower()

    # Groq/OpenAI-style structured error codes.
    error_code = getattr(exc, "code", None)
    if isinstance(error_code, str):
        error_code = error_code.lower()

    retryable_error_codes = {
        "model_not_found",
        "service_unavailable",
        "rate_limit_exceeded",
        "internal_server_error",
    }

    if error_code in retryable_error_codes:
        return True

    retryable_exception_names = {
        "timeout",
        "timeouterror",
        "connecterror",
        "connectionerror",
        "ratelimiterror",
        "serviceunavailableerror",
        "internalservererror",
        "notfounderror",
    }

    if exception_name in retryable_exception_names:
        return True

    retryable_keywords = (
        "model_not_found",
        "model does not exist",
        "you do not have access to it",
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "server error",
    )

    return any(keyword in exception_message for keyword in retryable_keywords)


def _create_chat_model(model_name: str) -> ChatGroq:
    api_key = _get_required_env("GROQ_API_KEY")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
    )


class LLMService:
    """
    Centralized Groq LLM service with primary → fallback behavior.

    All application agents should use this service instead of creating
    ChatGroq clients directly.
    """

    def __init__(
        self,
        primary_model: ChatGroq,
        fallback_model: ChatGroq,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        Invoke the primary model.

        If the primary model encounters a retryable failure, invoke
        the fallback model once.

        Non-retryable primary failures are raised immediately.
        If the fallback also fails, raise LLMInvocationError.
        """

        try:
            return self.primary_model.invoke(messages, **kwargs)
        except Exception as primary_error:
            if not _is_retryable_error(primary_error):
                raise

            try:
                return self.fallback_model.invoke(messages, **kwargs)
            except Exception as fallback_error:
                raise LLMInvocationError(
                    "Both primary and fallback Groq models failed. "
                    f"Primary error: {primary_error}. "
                    f"Fallback error: {fallback_error}."
                ) from fallback_error


def get_llm_service() -> LLMService:
    """
    Create the application's shared LLM service from environment config.
    """

    primary_model_name = _get_required_env("PRIMARY_MODEL")
    fallback_model_name = _get_required_env("FALLBACK_MODEL")

    primary_model = _create_chat_model(primary_model_name)
    fallback_model = _create_chat_model(fallback_model_name)

    return LLMService(
        primary_model=primary_model,
        fallback_model=fallback_model,
    )