import os
import sys
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import LLMTier

ENV_PATH = PROJECT_ROOT / ".env"

_TIER_CONFIG = {
    LLMTier.LOW: {
        "model": "gpt-5.6-luna",
        "temperature": 0.0,
        "reasoning_effort": "none",
    },
    LLMTier.MEDIUM: {
        "model": "gpt-5.6-terra",
        "temperature": 0.0,
        "reasoning_effort": "none",
    },
    LLMTier.HIGH: {
        "model": "gpt-5.6-sol",
        "temperature": 0.0,
        "reasoning_effort": "none",
    },
}


def reload_environment() -> None:
    if not ENV_PATH.is_file():
        raise FileNotFoundError(f".env file not found: {ENV_PATH}")

    # Replace an old key already present in the process environment.
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError(f"OPENAI_API_KEY is missing or empty in {ENV_PATH}")


@lru_cache(maxsize=None)
def _switch_llm_cached(tier: LLMTier) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        **_TIER_CONFIG[tier],
    )


def switch_llm(
    tier: LLMTier | str,
    force_refresh: bool = False,
) -> ChatOpenAI:
    if isinstance(tier, str):
        tier = LLMTier(tier)

    if force_refresh:
        reload_environment()
        _switch_llm_cached.cache_clear()

    return _switch_llm_cached(tier)


reload_environment()


