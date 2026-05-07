import json
import time
from dataclasses import dataclass
from typing import Optional

from .clients import langfuse
from .config import BENCHMARK_GROUND_TRUTH, JUDGE_MODEL, JUDGE_PROVIDER
from .llm import call_judge, call_llm, parse_and_normalize_intent_json
from .prompts import HUMAN_READABLE_SYSTEM, JUDGE_SYSTEM, SCORE_FIELDS


@dataclass
class IterationResult:
    iteration: int
    model_id: str
    model_label: str
    human_readable: Optional[str] = None
    parsed_intent: Optional[dict] = None
    intent_parse_error: Optional[str] = None
    generation_error: Optional[str] = None
    generation_time_ms: float = 0.0
    judge_scores: Optional[dict] = None
    judge_error: Optional[str] = None
    judge_time_ms: float = 0.0
    total_time_ms: float = 0.0
    passed: bool = False
    overall_score: float = 0.0
    trace_id: Optional[str] = None


def run_iteration(model_cfg: dict, iteration: int, decoded_data: dict) -> IterationResult:
    result = IterationResult(
        iteration=iteration,
        model_id=model_cfg["id"],
        model_label=model_cfg["label"],
    )

    trace = langfuse.start_observation(
        name="evm-decoder-hr-benchmark",
        as_type="span",
        input=decoded_data,
        metadata={
            "session_id": f"benchmark-hr-{model_cfg['id']}",
            "user_id": "benchmark-runner",
            "model_id": model_cfg["id"],
            "model_label": model_cfg["label"],
            "provider": model_cfg["provider"],
            "iteration": iteration,
            "tags": ["benchmark", "evm-decoder", "human-readable", model_cfg["id"]],
        },
    )
    # trace.trace_id is the trace id used for scores & URLs; trace.id is the root observation id
    result.trace_id = trace.trace_id

    total_start = time.perf_counter()
    decoded_data_json = json.dumps(decoded_data, indent=2)

    # ── Step 1: generation ──────────────────────────────────────────────
    gen_span = trace.start_observation(
        name="human-readable-generation",
        as_type="generation",
        model=model_cfg["id"],
        input=decoded_data_json,
        metadata={"step": "generate", "iteration": iteration},
    )

    user_prompt = (
        "Here is the structured decode of an EVM transaction "
        "(token data enriched via DexScreener API). "
        "Output ONLY the JSON object matching intent_output_spec / INTENT schema "
        f"(see system message).\n\n{decoded_data_json}"
    )

    t0 = time.perf_counter()
    try:
        result.human_readable = call_llm(
            provider=model_cfg["provider"],
            model_id=model_cfg["id"],
            system=HUMAN_READABLE_SYSTEM,
            user_msg=user_prompt,
            use_tools=True,
        )
    except Exception as e:
        result.generation_error = f"{type(e).__name__}: {e}"

    if result.human_readable and not result.generation_error:
        normalized, perr = parse_and_normalize_intent_json(result.human_readable)
        if normalized is not None:
            result.parsed_intent = normalized
            result.human_readable = json.dumps(normalized, ensure_ascii=False)
        else:
            result.intent_parse_error = perr

    result.generation_time_ms = (time.perf_counter() - t0) * 1000

    gen_span.update(
        output={
            "raw_or_canonical_json": result.human_readable,
            "parsed_ok": result.parsed_intent is not None,
            "intent_parse_error": result.intent_parse_error,
            "generation_error": result.generation_error,
        },
        metadata={
            "time_ms": round(result.generation_time_ms, 2),
            "error": result.generation_error or result.intent_parse_error,
        },
        level="ERROR" if (result.generation_error or result.intent_parse_error) else "DEFAULT",
    )
    gen_span.end()

    # ── Step 2: judge ───────────────────────────────────────────────────
    judge_span = trace.start_observation(
        name="llm-judge",
        as_type="generation",
        model=JUDGE_MODEL,
        input={
            "decoded_data": decoded_data,
            "candidate": result.human_readable or result.generation_error,
        },
        metadata={"step": "judge", "iteration": iteration},
    )

    ground_truth_json = json.dumps(BENCHMARK_GROUND_TRUTH, indent=2)
    judge_prompt = (
        f"GROUND_TRUTH:\n{ground_truth_json}\n\n"
        f"DECODED_DATA:\n{decoded_data_json}\n\n"
        f"CANDIDATE:\n{result.human_readable or result.generation_error or 'ERROR: No output'}"
    )

    t1 = time.perf_counter()
    try:
        result.judge_scores = call_judge(
            provider=JUDGE_PROVIDER,
            model_id=JUDGE_MODEL,
            system=JUDGE_SYSTEM,
            user_msg=judge_prompt,
        )
        result.passed = result.judge_scores.get("pass", False)
        result.overall_score = result.judge_scores.get("overall", 0)
    except Exception as e:
        result.judge_error = f"{type(e).__name__}: {e}"

    result.judge_time_ms = (time.perf_counter() - t1) * 1000
    result.total_time_ms = (time.perf_counter() - total_start) * 1000

    judge_span.update(
        output=result.judge_scores or result.judge_error,
        metadata={
            "time_ms": round(result.judge_time_ms, 2),
            "error": result.judge_error,
        },
        level="ERROR" if result.judge_error else "DEFAULT",
    )
    judge_span.end()

    # ── Scores at trace level ───────────────────────────────────────────
    # score_trace() surfaces in Langfuse Scores / trace overview;
    # observation-level scores often do not.
    if result.judge_scores:
        for fname in SCORE_FIELDS:
            val = result.judge_scores.get(fname)
            if val is not None:
                trace.score_trace(
                    name=fname,
                    value=float(val),
                    data_type="NUMERIC",
                    comment=f"iter {iteration}",
                )
        trace.score_trace(
            name="pass",
            value=1.0 if result.passed else 0.0,
            data_type="BOOLEAN",
            comment=f"iter {iteration} — {'PASS' if result.passed else 'FAIL'}",
        )

    err_list = []
    if result.generation_error:
        err_list.append(result.generation_error)
    if result.intent_parse_error:
        err_list.append(f"intent JSON: {result.intent_parse_error}")
    if result.judge_error:
        err_list.append(result.judge_error)
    if result.judge_scores and result.judge_scores.get("errors"):
        err_list.extend(result.judge_scores["errors"])

    trace.update(
        output={
            "passed": result.passed,
            "overall_score": result.overall_score,
            "generation_time_ms": round(result.generation_time_ms, 2),
            "total_time_ms": round(result.total_time_ms, 2),
            "parsed_intent": result.parsed_intent,
            "errors": err_list,
        },
    )
    trace.end()

    # Flush now so scores appear in Langfuse without waiting for shutdown
    langfuse.flush()

    return result
