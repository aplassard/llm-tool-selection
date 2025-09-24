"""Helper client wrapping the OpenAI SDK for OpenRouter usage."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import APIStatusError, OpenAI

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Thin wrapper that configures the OpenAI SDK for OpenRouter endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        app_title: str = "tool-calling-evaluation",
        app_referer: str = "https://tool-calling-evaluation",
    ) -> None:
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set in the environment")

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._extra_headers = {
            "HTTP-Referer": app_referer,
            "X-Title": app_title,
        }

    def chat_completion(
        self,
        model: str,
        messages: Any,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools is not None:
            request_payload["tools"] = tools
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            request_payload["parallel_tool_calls"] = parallel_tool_calls

        try:
            completion = self._client.chat.completions.create(
                **request_payload,
                extra_headers=self._extra_headers,
            )
            data = completion.model_dump()
            meta = data.setdefault("_openrouter", {})
            meta.setdefault("requested_model", model)
            meta.setdefault("resolved_model", data.get("model", model))
            return data
        except APIStatusError as exc:
            if ":free" in model and exc.status_code in {400, 404, 422}:
                fallback_model = model.split(":", 1)[0]
                fallback_payload = dict(request_payload)
                fallback_payload["model"] = fallback_model
                completion = self._client.chat.completions.create(
                    **fallback_payload,
                    extra_headers=self._extra_headers,
                )
                data = completion.model_dump()
                meta = data.setdefault("_openrouter", {})
                meta.update(
                    {
                        "requested_model": model,
                        "resolved_model": data.get("model", fallback_model),
                    }
                )
                return data
            raise

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
