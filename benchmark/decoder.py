import requests
from datetime import datetime, timezone
from typing import Any, Optional

from .config import BENCHMARK_MSG_VALUE_WEI, RPC_URL, TO_ADDRESS

# Heuristic range for unix timestamps (2001–2286)
_TS_MIN = 1_000_000_000
_TS_MAX = 10_000_000_000


def lookup_selector(selector: str) -> Optional[dict]:
    url = f"https://api.openchain.xyz/signature-database/v1/lookup?function=0x{selector}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", {}).get("function", {}).get(f"0x{selector}", [])
        valid = [r for r in results if "_tg_inv" not in r["name"] and "func_" not in r["name"]]
        if valid:
            sig = valid[0]["name"]
            name = sig.split("(")[0]
            params_str = sig[sig.index("(") + 1 : -1]
            param_types = [p.strip() for p in params_str.split(",")] if params_str else []
            return {"signature": sig, "name": name, "param_types": param_types}
    except Exception:
        pass
    return None


_ERC20_NAME     = "0x06fdde03"
_ERC20_SYMBOL   = "0x95d89b41"
_ERC20_DECIMALS = "0x313ce567"


def _eth_call(to: str, data: str) -> Optional[str]:
    payload = {"jsonrpc": "2.0", "method": "eth_call", "params": [{"to": to, "data": data}, "latest"], "id": 1}
    resp = requests.post(RPC_URL, json=payload, timeout=10)
    resp.raise_for_status()
    result = resp.json().get("result")
    return result if result and result not in ("0x", "0x0") else None


def _decode_string(hex_result: str) -> str:
    data = hex_result[2:] if hex_result.startswith("0x") else hex_result
    if len(data) < 64:
        return ""
    try:
        # Standard ABI string: word0=offset(32), word1=length, word2+=data
        if int(data[:64], 16) == 32 and len(data) >= 128:
            length = int(data[64:128], 16)
            return bytes.fromhex(data[128:128 + length * 2]).decode("utf-8", errors="replace")
    except Exception:
        pass
    # Fallback: bytes32 right-padded (old tokens)
    try:
        return bytes.fromhex(data[:64]).rstrip(b"\x00").decode("utf-8", errors="replace")
    except Exception:
        return ""


def lookup_token_rpc(address: str) -> dict:
    """Called as an LLM tool — fetches ERC-20 name, symbol, decimals via RPC eth_call."""
    name, symbol, decimals = "Unknown", "???", None
    try:
        if r := _eth_call(address, _ERC20_NAME):
            name = _decode_string(r) or name
    except Exception:
        pass
    try:
        if r := _eth_call(address, _ERC20_SYMBOL):
            symbol = _decode_string(r) or symbol
    except Exception:
        pass
    try:
        if r := _eth_call(address, _ERC20_DECIMALS):
            decimals = int(r, 16)
    except Exception:
        pass
    return {"address": address, "name": name, "symbol": symbol, "decimals": decimals, "source": "rpc"}


def _is_dynamic(ptype: str) -> bool:
    return ptype in ("bytes", "string") or "[]" in ptype


def _decode_static(word: str, ptype: str) -> Any:
    if ptype == "address":
        return "0x" + word[-40:]
    if ptype == "bool":
        return bool(int(word, 16))
    if ptype.startswith("uint"):
        return int(word, 16)
    if ptype.startswith("int"):
        bits = int(ptype[3:]) if ptype[3:] else 256
        val = int(word, 16)
        return val - (2 ** bits) if val >= 2 ** (bits - 1) else val
    if ptype.startswith("bytes") and ptype != "bytes":
        n = int(ptype[5:])
        return "0x" + word[: n * 2]
    return word


def _decode_dynamic(words: list[str], data_idx: int, ptype: str) -> Any:
    if data_idx >= len(words):
        return None
    if ptype.endswith("[]"):
        elem_type = ptype[:-2]
        length = int(words[data_idx], 16)
        return [
            _decode_static(words[data_idx + 1 + i], elem_type)
            for i in range(length)
            if data_idx + 1 + i < len(words)
        ]
    if ptype == "string":
        length = int(words[data_idx], 16)
        chunks = words[data_idx + 1 : data_idx + 1 + (length + 31) // 32]
        return bytes.fromhex("".join(chunks)[: length * 2]).decode("utf-8", errors="replace")
    if ptype == "bytes":
        length = int(words[data_idx], 16)
        chunks = words[data_idx + 1 : data_idx + 1 + (length + 31) // 32]
        return "0x" + "".join(chunks)[: length * 2]
    return words[data_idx]


def deterministic_decode(raw_calldata: str) -> dict:
    hex_str = raw_calldata.strip().replace(" ", "").replace("\n", "")
    if hex_str.startswith(("0x", "0X")):
        hex_str = hex_str[2:]
    hex_str = hex_str.lower()

    bad_chars = [(i, c) for i, c in enumerate(hex_str) if c not in "0123456789abcdef"]
    if bad_chars:
        raise ValueError(f"Non-hex characters at positions: {bad_chars}")
    if len(hex_str) % 2 != 0:
        raise ValueError(f"Odd-length hex string ({len(hex_str)} chars)")

    selector = hex_str[:8]
    params_hex = hex_str[8:]

    fn_info = lookup_selector(selector)
    if not fn_info:
        raise ValueError(f"Unknown selector: 0x{selector}")

    words = [params_hex[i : i + 64] for i in range(0, len(params_hex), 64)]
    params = []
    for i, ptype in enumerate(fn_info["param_types"]):
        if i >= len(words):
            break
        value = _decode_dynamic(words, int(words[i], 16) // 32, ptype) if _is_dynamic(ptype) else _decode_static(words[i], ptype)
        param: dict = {"index": i, "type": ptype, "value": value}
        if ptype.startswith("uint") and isinstance(value, int) and _TS_MIN < value < _TS_MAX:
            param["utc"] = datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
        params.append(param)

    msg_wei = BENCHMARK_MSG_VALUE_WEI

    return {
        "selector": f"0x{selector}",
        "function_signature": fn_info["signature"],
        "function_name": fn_info["name"],
        "to": TO_ADDRESS,
        "msg_value_wei": str(msg_wei),
        "msg_value_eth": msg_wei / (10 ** 18) if msg_wei else None,
        "parameters": params,
    }
