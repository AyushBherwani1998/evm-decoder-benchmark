from .clients import OPENAI_COMPAT_CLIENTS

MODELS: list[dict] = [
    {"id": "claude-sonnet-4.6", "provider": "litellm", "label": "Claude Sonnet 4.6"},
    {"id": "claude-opus-4.6",  "provider": "litellm", "label": "Claude Opus 4.6"},
    {"id": "gpt-5.4", "provider": "litellm", "label": "GPT-5.4"},
    {"id": "qwen3-235b-a22b-2507", "provider": "litellm", "label": "Qwen3 235B"},
    {"id": "gemini-3-flash-preview", "provider": "litellm", "label": "Gemini 3 Flash Preview"},
]

MODELS = [m for m in MODELS if OPENAI_COMPAT_CLIENTS.get(m["provider"]) is not None]
