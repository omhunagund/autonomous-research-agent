import os
from typing import Any, TypeVar

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import AIMessage, BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel


load_dotenv()


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


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


def _message_to_groq_dict(message: BaseMessage) -> dict[str, Any]:
    """
    Convert a LangChain message into the role/content structure expected
    by Groq's Chat Completions API.
    """

    role_mapping = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }

    role = role_mapping.get(message.type)

    if role is None:
        raise ValueError(
            f"Unsupported LangChain message type: {message.type!r}"
        )

    return {
        "role": role,
        "content": message.content,
    }


def _messages_to_groq(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [_message_to_groq_dict(message) for message in messages]


def _build_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """
    Build a Groq-compatible JSON Schema from a Pydantic model.

    Groq strict structured outputs require object schemas to disallow
    additional properties and require every field.
    """

    json_schema = schema.model_json_schema()

    def normalize_object_schema(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False

            for value in node.values():
                normalize_object_schema(value)

        elif isinstance(node, list):
            for item in node:
                normalize_object_schema(item)

    normalize_object_schema(json_schema)

    return json_schema


def _invoke_groq_structured(
    client: Groq,
    model_name: str,
    messages: list[BaseMessage],
    schema: type[StructuredModel],
    **kwargs: Any,
) -> StructuredModel:
    """
    Invoke Groq using native JSON Schema structured outputs and validate
    the response through the supplied Pydantic model.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=_messages_to_groq(messages),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__.lower(),
                "strict": True,
                "schema": _build_json_schema(schema),
            },
        },
        **kwargs,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            f"Structured response from model '{model_name}' was empty."
        )

    try:
        return schema.model_validate_json(content)
    except Exception as exc:
        raise ValueError(
            f"Structured response from model '{model_name}' "
            "could not be validated against the requested schema."
        ) from exc


class LLMService:
    """
    Centralized Groq LLM service.

    Normal invocations use LangChain's ChatGroq wrapper.

    Structured invocations use Groq's native JSON Schema API so the
    application does not depend on LangChain's tool-calling structured
    output implementation.
    """

    def __init__(
        self,
        primary_model: ChatGroq,
        fallback_model: ChatGroq,
        groq_client: Groq,
        primary_model_name: str,
        fallback_model_name: str,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        self.groq_client = groq_client

        self.primary_model_name = primary_model_name
        self.fallback_model_name = fallback_model_name

    def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        Invoke the primary model.

        Retryable provider failures trigger one fallback attempt.
        Non-retryable failures are raised immediately.
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

    def invoke_structured(
        self,
        messages: list[BaseMessage],
        schema: type[StructuredModel],
        **kwargs: Any,
    ) -> StructuredModel:
        """
        Invoke a Pydantic structured-output request through Groq's native
        JSON Schema API.

        Provider-level retryable failures trigger the fallback model.

        Structured-output validation errors are propagated to the caller
        because they belong to the caller's validation/retry policy, not
        the provider fallback policy.
        """

        try:
            return _invoke_groq_structured(
                client=self.groq_client,
                model_name=self.primary_model_name,
                messages=messages,
                schema=schema,
                **kwargs,
            )

        except Exception as primary_error:
            if not _is_retryable_error(primary_error):
                raise

            try:
                return _invoke_groq_structured(
                    client=self.groq_client,
                    model_name=self.fallback_model_name,
                    messages=messages,
                    schema=schema,
                    **kwargs,
                )

            except Exception as fallback_error:
                if not _is_retryable_error(fallback_error):
                    raise

                raise LLMInvocationError(
                    "Both primary and fallback Groq models failed during "
                    "structured invocation. "
                    f"Primary error: {primary_error}. "
                    f"Fallback error: {fallback_error}."
                ) from fallback_error


def get_llm_service() -> LLMService:
    """
    Create the application's shared LLM service from environment config.
    """

    api_key = _get_required_env("GROQ_API_KEY")
    primary_model_name = _get_required_env("PRIMARY_MODEL")
    fallback_model_name = _get_required_env("FALLBACK_MODEL")

    primary_model = _create_chat_model(primary_model_name)
    fallback_model = _create_chat_model(fallback_model_name)

    groq_client = Groq(api_key=api_key)

    return LLMService(
        primary_model=primary_model,
        fallback_model=fallback_model,
        groq_client=groq_client,
        primary_model_name=primary_model_name,
        fallback_model_name=fallback_model_name,
    )