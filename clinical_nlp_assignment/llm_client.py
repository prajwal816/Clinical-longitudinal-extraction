from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .utils import read_env_required


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 2048


def load_llm_config_from_env(
    *, temperature: float = 0.0, max_output_tokens: int = 2048
) -> LLMConfig:
    return LLMConfig(
        base_url=read_env_required("OPENAI_BASE_URL"),
        api_key=read_env_required("OPENAI_API_KEY"),
        model=read_env_required("OPENAI_MODEL"),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20))
    def json_chat(self, *, system: str, user: str) -> dict[str, Any]:
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
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

