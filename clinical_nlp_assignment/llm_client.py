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


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Try to parse text as JSON.  If it fails, look for a fenced code block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from fenced code block
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Last resort: find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def token_summary(self) -> str:
        return (
            f"LLM Usage: {self._total_calls} calls | "
            f"prompt={self._total_prompt_tokens:,} + completion={self._total_completion_tokens:,} "
            f"= {self.total_tokens:,} total tokens"
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20))
    def json_chat(self, *, system: str, user: str) -> dict[str, Any]:
        t0 = time.perf_counter()
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
        except Exception:
            # Some providers don't support response_format; fall back
            logger.warning("response_format=json_object not supported; retrying without it.")
            resp = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                messages=[
                    {"role": "system", "content": system + "\n\nYou MUST respond with valid JSON only. No other text."},
                    {"role": "user", "content": user},
                ],
            )

        elapsed = time.perf_counter() - t0

        # Track tokens
        usage = resp.usage
        prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
        completion_tok = getattr(usage, "completion_tokens", 0) or 0
        self._total_prompt_tokens += prompt_tok
        self._total_completion_tokens += completion_tok
        self._total_calls += 1

        logger.info(
            "LLM call #%d: model=%s prompt_tok=%d compl_tok=%d latency=%.1fs",
            self._total_calls,
            self.config.model,
            prompt_tok,
            completion_tok,
            elapsed,
        )

        content = resp.choices[0].message.content or "{}"
        return _extract_json(content)
