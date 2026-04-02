from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .utils import read_env_required

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 4096


def load_llm_config_from_env(
    *, temperature: float = 0.0, max_output_tokens: int = 4096
) -> LLMConfig:
    return LLMConfig(
        base_url=read_env_required("OPENAI_BASE_URL"),
        api_key=read_env_required("OPENAI_API_KEY"),
        model=read_env_required("OPENAI_MODEL"),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from text that may contain markdown fences or prose.

    Handles cases where the model wraps JSON in ```json ... ``` blocks
    or includes explanatory text around the JSON.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from fenced code block
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the outermost { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: return empty
    logger.warning("Could not parse JSON from LLM response, returning empty dict")
    return {}


@dataclass
class TokenUsage:
    """Accumulates token usage across all API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    total_latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, prompt: int, completion: int, latency: float) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_calls += 1
        self.total_latency_s += latency

    def summary(self) -> str:
        avg_latency = (self.total_latency_s / self.total_calls) if self.total_calls else 0
        return (
            f"LLM Usage: {self.total_calls} calls | "
            f"{self.prompt_tokens:,} prompt + {self.completion_tokens:,} completion = "
            f"{self.total_tokens:,} total tokens | "
            f"{self.total_latency_s:.1f}s total ({avg_latency:.1f}s avg)"
        )


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.usage = TokenUsage()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=30))
    def json_chat(self, *, system: str, user: str) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            # Some providers don't support response_format; fall back
            if "response_format" in str(e).lower() or "json_object" in str(e).lower():
                logger.info("response_format not supported, falling back to plain chat")
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    messages=[
                        {"role": "system", "content": system + "\n\nYou MUST respond with valid JSON only, no other text."},
                        {"role": "user", "content": user},
                    ],
                )
            else:
                raise

        latency = time.monotonic() - t0

        # Track token usage
        if resp.usage:
            self.usage.record(
                prompt=resp.usage.prompt_tokens or 0,
                completion=resp.usage.completion_tokens or 0,
                latency=latency,
            )

        content = resp.choices[0].message.content or "{}"
        result = _extract_json_from_text(content)

        logger.debug(
            "LLM call: model=%s tokens=%d+%d latency=%.1fs",
            self.config.model,
            resp.usage.prompt_tokens if resp.usage else 0,
            resp.usage.completion_tokens if resp.usage else 0,
            latency,
        )

        return result
