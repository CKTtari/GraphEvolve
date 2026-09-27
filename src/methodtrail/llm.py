"""OpenAI-compatible structured-output client, including DashScope compatibility."""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def complete(self, system: str, user: str, response_model: type[T]) -> T: ...


class OpenAICompatibleLLM:
    """Uses an API key from the environment; never stores it in experiment artifacts."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in environment variable {api_key_env}")
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def complete(self, system: str, user: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        # Some OpenAI-compatible gateways accept `json_object` but do not enforce a
        # JSON Schema. Put the exact contract in the prompt as well, so the model
        # sees the required field names instead of inferring a free-form format.
        schema_prompt = (
            "\n\nReturn ONLY one valid JSON object. Use the exact field names and "
            "value types in this schema; do not rename, omit, or add top-level "
            "fields:\n"
            + json.dumps(schema, ensure_ascii=False, indent=2)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + schema_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
        except (
            Exception
        ) as exc:  # API exceptions need to become useful recovery evidence.
            raise RuntimeError(
                f"LLM request failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return response_model.model_validate_json(_json_object(content))
        except Exception as exc:
            raise RuntimeError(
                f"LLM response did not match {response_model.__name__}; expected schema keys: {list(schema.get('properties', {}))}"
            ) from exc


def _json_object(content: str) -> str:
    """Extract one JSON object from occasional fenced model output."""

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else content.strip()
    json.loads(candidate)
    return candidate
