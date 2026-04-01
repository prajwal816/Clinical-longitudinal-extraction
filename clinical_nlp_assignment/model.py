from __future__ import annotations

from .llm_client import LLMConfig, OpenAICompatibleClient, load_llm_config_from_env


def build_llm_client(
    *, temperature: float = 0.0, max_output_tokens: int = 2048
) -> OpenAICompatibleClient:
    """
    Build an OpenAI-compatible client using required env vars:
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
    """
    cfg: LLMConfig = load_llm_config_from_env(
        temperature=temperature, max_output_tokens=max_output_tokens
    )
    return OpenAICompatibleClient(cfg)

