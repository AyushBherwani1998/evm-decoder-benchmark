import json
from typing import Any, Optional

from .clients import OPENAI_COMPAT_CLIENTS
from .decoder import lookup_token_rpc

# ── Tool definitions ─────────────────────────────────────────────────────────

_TOOL_LOOKUP_TOKEN = "lookup_token"

_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": _TOOL_LOOKUP_TOKEN,
        "description": (
            "Fetch ERC-20 token metadata (name, symbol, decimals) for a contract address "
            "on Base chain via RPC eth_call. Call this for every address in the decoded "
            "parameters before generating the intent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "EVM contract address (0x...)"},
            },
            "required": ["address"],
        },
    },
}


_SCORE_INT = {"type": "integer", "minimum": 1, "maximum": 10}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_compliance": _SCORE_INT,
        "summary_quality":   _SCORE_INT,
        "details_grounding": _SCORE_INT,
        "token_accuracy":    _SCORE_INT,
        "action_semantics":  _SCORE_INT,
        "numeric_accuracy":  _SCORE_INT,
        "clarity":           _SCORE_INT,
        "no_hallucinations": _SCORE_INT,
        "overall":           _SCORE_INT,
        "pass":              {"type": "boolean"},
        "errors":            {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_compliance", "summary_quality", "details_grounding",
        "token_accuracy", "action_semantics", "numeric_accuracy",
        "clarity", "no_hallucinations", "overall", "pass", "errors",
    ],
}

_TOOL_JUDGE_SCORE_OPENAI = {
    "type": "function",
    "function": {
        "name": "score_intent",
        "description": "Record evaluation scores for the candidate intent JSON against the ground truth.",
        "parameters": _JUDGE_SCHEMA,
    },
}


def _run_tool(name: str, inputs: dict) -> str:
    if name == _TOOL_LOOKUP_TOKEN:
        return json.dumps(lookup_token_rpc(inputs["address"]))
    raise ValueError(f"Unknown tool: {name}")


# ── LLM callers ──────────────────────────────────────────────────────────────

def call_llm(
    provider: str,
    model_id: str,
    system: str,
    user_msg: str,
    use_tools: bool = False,
) -> str:
    client = OPENAI_COMPAT_CLIENTS.get(provider)
    if client is None:
        raise RuntimeError(f"Provider '{provider}' is not configured (check LITELLM_API_KEY / LITELLM_BASE_URL)")
    return _call_openai_compat(client, model_id, system, user_msg, use_tools)


def _call_openai_compat(client: Any, model_id: str, system: str, user_msg: str, use_tools: bool) -> str:
    messages: list = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    kwargs: dict = {"model": model_id, "max_tokens": 2048}
    if use_tools:
        kwargs["tools"] = [_TOOL_OPENAI]

    while True:
        resp = client.chat.completions.create(**kwargs, messages=messages)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg)
        for tc in msg.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _run_tool(tc.function.name, json.loads(tc.function.arguments)),
            })


def call_judge(provider: str, model_id: str, system: str, user_msg: str) -> dict:
    """Force structured score output via tool calling. Returns the parsed score dict."""
    client = OPENAI_COMPAT_CLIENTS.get(provider)
    if client is None:
        raise RuntimeError(f"Judge provider '{provider}' is not configured (check LITELLM_API_KEY / LITELLM_BASE_URL)")
    return _call_judge_openai_compat(client, model_id, system, user_msg)


def _call_judge_openai_compat(client: Any, model_id: str, system: str, user_msg: str) -> dict:
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        tools=[_TOOL_JUDGE_SCORE_OPENAI],
        tool_choice={"type": "function", "function": {"name": "score_intent"}},
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise RuntimeError(
            f"Judge did not produce a score_intent tool_call "
            f"(finish_reason={resp.choices[0].finish_reason}, content={(msg.content or '')[:200]})"
        )
    for tc in msg.tool_calls:
        if tc.function.name == "score_intent":
            return json.loads(tc.function.arguments)
    raise RuntimeError(f"Judge tool_calls did not include score_intent: {[tc.function.name for tc in msg.tool_calls]}")


# ── JSON helpers ─────────────────────────────────────────────────────────────

def validate_intent_payload(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root must be a JSON object"
    if "intent" not in obj:
        return False, 'missing top-level "intent" key'
    intent = obj["intent"]
    if not isinstance(intent, dict):
        return False, '"intent" must be an object'
    summary = intent.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False, "intent.summary must be a non-empty string"
    details = intent.get("details")
    if not isinstance(details, list):
        return False, "intent.details must be an array of strings"
    for i, line in enumerate(details):
        if not isinstance(line, str):
            return False, f"intent.details[{i}] must be a string"
    return True, ""


def safe_parse_json(text: str) -> Optional[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.split("\n") if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def parse_and_normalize_intent_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    parsed = safe_parse_json(text)
    if parsed is None:
        return None, "invalid JSON"
    ok, err = validate_intent_payload(parsed)
    if not ok:
        return None, err
    return {"intent": parsed["intent"]}, None
