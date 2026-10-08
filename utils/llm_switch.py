from functools import lru_cache

from langchain_openai import ChatOpenAI  # or your provider

from config.settings import LLMTier


_TIER_CONFIG = {
    LLMTier.LOW:    {"model": "gpt-5.6-luna", "temperature": 0.0},
    LLMTier.MEDIUM: {"model": "gpt-5.6-terra",      "temperature": 0.0},
    LLMTier.HIGH:   {"model": "gpt-5.6-sol", "temperature": 0.0},
}


@lru_cache(maxsize=None)
def switch_llm(tier: LLMTier) -> ChatOpenAI:
    """Return a cached, configured ChatModel for the given tier."""
    if isinstance(tier, str):
        tier = LLMTier(tier)
    cfg = _TIER_CONFIG[tier]
    return ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])



if __name__=="__main__":

    llm_obj=switch_llm("low")
    print(llm_obj.invoke("What is the capital of Italy?"))