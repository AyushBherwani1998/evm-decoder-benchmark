from typing import Optional

from langfuse import Langfuse
import openai as openai_mod

from .config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_BASE_URL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
)

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL,
)


def _make_openai_compat(api_key: Optional[str], base_url: Optional[str] = None):
    if not api_key:
        return None
    try:
        return openai_mod.OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


litellm_client = _make_openai_compat(LITELLM_API_KEY, LITELLM_BASE_URL)

OPENAI_COMPAT_CLIENTS: dict = {
    "litellm": litellm_client,
}
