from __future__ import annotations

from src.core.llm import _build_json_schema
from src.models.schemas import CritiqueAssessment, InternalAnalysis


def _object_schemas(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _object_schemas(value)
    elif isinstance(node, list):
        for value in node:
            yield from _object_schemas(value)


def test_groq_schema_requires_every_object_property() -> None:
    schema = _build_json_schema(InternalAnalysis)

    for object_schema in _object_schemas(schema):
        properties = object_schema.get("properties", {})
        assert object_schema.get("additionalProperties") is False
        assert set(object_schema.get("required", [])) == set(properties)


def test_gap_nullable_fields_are_required_for_groq() -> None:
    schema = _build_json_schema(InternalAnalysis)
    gap_schema = schema["$defs"]["Gap"]

    assert "related_sub_question" in gap_schema["required"]
    assert "related_claim" in gap_schema["required"]
    assert gap_schema["properties"]["related_sub_question"]["anyOf"][-1] == {"type": "null"}
    assert gap_schema["properties"]["related_claim"]["anyOf"][-1] == {"type": "null"}

def test_critique_assessment_schema_does_not_require_verdict() -> None:
    schema = _build_json_schema(CritiqueAssessment)

    properties = schema["properties"]

    assert "verdict" not in properties
    assert set(schema["required"]) == set(properties)
