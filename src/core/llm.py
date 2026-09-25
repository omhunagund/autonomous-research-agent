import json
import os
from typing import Any, TypeVar

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import AIMessage, BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from collections.abc import Callable


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

    if status_code in {408, 429}:
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
                properties = node.get("properties")
                if isinstance(properties, dict):
                    # Groq strict JSON Schema requires every object property
                    # to be listed in `required`, including nullable fields.
                    # Pydantic omits fields with defaults from `required`, so
                    # normalize the provider-facing schema without changing
                    # the Pydantic model's Python-side optional semantics.
                    node["required"] = list(properties)

            for value in node.values():
                normalize_object_schema(value)

        elif isinstance(node, list):
            for item in node:
                normalize_object_schema(item)

    normalize_object_schema(json_schema)

    return json_schema


_STRUCTURED_OUTPUT_ERROR_CODES = {
    "output_parse_failed",
    "json_validate_failed",
    "tool_use_failed",
}


def _get_structured_error_code(exc: Exception) -> str | None:
    code = getattr(exc, "code", None)

    if isinstance(code, str):
        return code.lower()

    body = getattr(exc, "body", None)

    if isinstance(body, dict):
        error = body.get("error")

        if isinstance(error, dict):
            body_code = error.get("code")

            if isinstance(body_code, str):
                return body_code.lower()

    return None


def _is_structured_output_error(exc: Exception) -> bool:
    return _get_structured_error_code(exc) in _STRUCTURED_OUTPUT_ERROR_CODES


def _get_failed_generation(exc: Exception) -> str | None:
    """
    Extract Groq's failed_generation payload when available.
    """

    body = getattr(exc, "body", None)

    if isinstance(body, dict):
        error = body.get("error")

        if isinstance(error, dict):
            failed_generation = error.get("failed_generation")
            if isinstance(failed_generation, str):
                return failed_generation

        failed_generation = body.get("failed_generation")
        if isinstance(failed_generation, str):
            return failed_generation

    return None


def _extract_json_candidates(text: str) -> list[Any]:
    """
    Extract complete JSON values embedded in a failed Groq generation.

    This is deliberately conservative:
    - complete JSON objects/arrays are considered
    - no guessing or repair of malformed JSON is performed
    """

    candidates: list[Any] = []

    stripped = text.strip()

    if not stripped:
        return candidates

    try:
        candidates.append(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(stripped):
        if character not in "{[":
            continue

        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue

        if value not in candidates:
            candidates.append(value)

    return candidates


def _recover_structured_output(
    exc: Exception,
    schema: type[StructuredModel],
) -> StructuredModel | None:
    """
    Recover a schema-valid structured result from Groq's failed_generation.

    This recovery is intentionally generic and provider-specific. It only
    accepts complete JSON that independently validates against the requested
    Pydantic schema.

    Tool-call-shaped generations such as:

        {"name": "critiqueassessment", "arguments": {...}}

    are unwrapped through their `arguments` object.

    Malformed or incomplete generations are never repaired or guessed.
    """

    if not _is_structured_output_error(exc):
        return None

    failed_generation = _get_failed_generation(exc)

    if not failed_generation:
        return None

    for candidate in _extract_json_candidates(failed_generation):
        if not isinstance(candidate, dict):
            continue

        nested_arguments = candidate.get("arguments")

        if isinstance(nested_arguments, dict):
            candidate = nested_arguments

        try:
            return schema.model_validate(candidate)
        except Exception:
            continue

    return None


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

    Structured calls use low reasoning effort to favor direct schema-conforming
    output while leaving normal LLM invocations unchanged.
    """
    request_kwargs = dict(kwargs)
    request_kwargs.setdefault("reasoning_effort", "low")

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
        **request_kwargs,
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
        recovery_handler: Callable[[Exception], StructuredModel | None] | None = None,
        **kwargs: Any,
    ) -> StructuredModel:
        """
        Invoke a Pydantic structured-output request through Groq's native
        JSON Schema API.

        Provider-level retryable failures trigger the fallback model.

        Structured-output protocol failures are handled centrally:
        1. Attempt safe recovery from Groq's failed_generation.
        2. If recovery is impossible and the failure is a recognized
           structured-output protocol error, use the fallback model.
        3. Validate every recovered/fallback result through the requested
           Pydantic schema.
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
            recovered = _recover_structured_output(
                primary_error,
                schema,
            )

            if recovered is not None:
                return recovered

            if recovery_handler is not None and _is_structured_output_error(
                primary_error
            ):
                recovered = recovery_handler(primary_error)

                if recovered is not None:
                    return recovered

            if not (
                _is_retryable_error(primary_error)
                or _is_structured_output_error(primary_error)
            ):
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
                recovered = _recover_structured_output(
                    fallback_error,
                    schema,
                )

                if recovered is not None:
                    return recovered

                if recovery_handler is not None and _is_structured_output_error(
                    fallback_error
                ):
                    recovered = recovery_handler(fallback_error)

                    if recovered is not None:
                        return recovered

                if not _is_retryable_error(fallback_error):
                    raise LLMInvocationError(
                        "Both primary and fallback Groq models failed during "
                        "structured invocation. "
                        f"Primary error: {primary_error}. "
                        f"Fallback error: {fallback_error}."
                    ) from fallback_error

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