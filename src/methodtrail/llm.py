"""OpenAI-compatible structured-output client, including DashScope compatibility."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
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
        log_path: str | Path | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in environment variable {api_key_env}")
        self.model = model
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def complete(self, system: str, user: str, response_model: type[T]) -> T:
        """Request one typed artifact, retrying recoverable schema mistakes.

        OpenAI-compatible gateways sometimes return JSON that is syntactically
        valid but violates a nested validator (for example, a ``replace`` edit
        with an empty ``old_text``).  That is a model-output error, not a task
        failure, so give the model a bounded correction attempt before letting
        the research loop classify a genuine API or execution failure.
        """
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
        correction = ""
        for attempt in range(3):
            call_id = uuid.uuid4().hex
            started = time.perf_counter()
            request_user = user + schema_prompt + correction
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": request_user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2 if attempt == 0 else 0.0,
                )
                content = response.choices[0].message.content or "{}"
            except Exception as exc:  # API errors remain visible to recovery.
                self._log_call(
                    call_id,
                    response_model,
                    system,
                    request_user,
                    None,
                    started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"LLM request failed: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                parsed = response_model.model_validate_json(_json_object(content))
            except Exception as exc:
                self._log_call(
                    call_id,
                    response_model,
                    system,
                    request_user,
                    content,
                    started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt < 2:
                    correction = (
                        "\n\nYour previous JSON failed validation with this error:\n"
                        f"{exc}\nReturn a corrected JSON object only. Preserve the research question. "
                        "For FileEdit, use operation=create only for an absent file; "
                        "for operation=replace provide a non-empty exact old_text snippet."
                    )
                    continue
                raise RuntimeError(
                    f"LLM response did not match {response_model.__name__}; expected schema keys: {list(schema.get('properties', {}))}"
                ) from exc
            self._log_call(
                call_id,
                response_model,
                system,
                request_user,
                content,
                started,
            )
            return parsed
        raise RuntimeError(f"LLM response did not match {response_model.__name__}")

    def _log_call(
        self,
        call_id: str,
        response_model: type[T],
        system: str,
        user: str,
        response: str | None,
        started: float,
        error: str | None = None,
    ) -> None:
        if self.log_path is None:
            return
        record = {
            "call_id": call_id,
            "at": datetime.now(UTC).isoformat(),
            "model": self.model,
            "response_model": response_model.__name__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "system": system,
            "user": user,
            "response": response,
            "error": error,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_object(content: str) -> str:
    """Extract one JSON object from occasional fenced model output."""

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else content.strip()
    json.loads(candidate)
    return candidate
