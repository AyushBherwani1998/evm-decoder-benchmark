"""
EVM Decoder Benchmark — Human-Readable Interpretation Quality
=============================================================

Architecture:
  1. DETERMINISTIC DECODE (EVM decoder skill pipeline)
     → Runs once. Extracts selector, params, addresses, timestamps.
     → This is the fixed input for all LLM calls.

  2. LLM HUMAN-READABLE GENERATION (the system under test)
     → Each model gets the same structured decode and must produce
       JSON: { "intent": { "summary": "...", "details": ["..."] } }.
     → Runs N times per model to measure consistency & error rate.

  3. LLM-AS-JUDGE (Claude Sonnet via Langfuse)
     → Scores each human-readable output against the structured decode
       and ground truth for accuracy, clarity, and completeness.
     → All scores traced to Langfuse.

"""

import csv
import json
import os
import statistics
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone

from .clients import OPENAI_COMPAT_CLIENTS, langfuse
from .config import (
    CALLDATA,
    ITERATIONS,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    LANGFUSE_BASE_URL,
    LANGFUSE_PROJECT_ID,
)
from .decoder import deterministic_decode
from .models import MODELS
from .prompts import SCORE_FIELDS
from .reporting import print_iteration, print_model_summary
from .runner import IterationResult, run_iteration


def main():
    if OPENAI_COMPAT_CLIENTS.get(JUDGE_PROVIDER) is None:
        raise RuntimeError(
            f"Judge provider '{JUDGE_PROVIDER}' is not configured. "
            f"Set LITELLM_API_KEY / LITELLM_BASE_URL or override JUDGE_PROVIDER / JUDGE_MODEL."
        )

    if not MODELS:
        raise RuntimeError(
            "No models to benchmark. Set LITELLM_API_KEY / LITELLM_BASE_URL to enable "
            "closed models (Claude Sonnet 4.6, Claude Opus 4.6, GPT-5.5) and OSS models "
            "(Qwen, Mistral, DeepSeek) via LiteLLM."
        )

    enabled_providers = sorted({m["provider"] for m in MODELS})

    print("=" * 76)
    print("  EVM DECODER BENCHMARK — Human-Readable Interpretation Quality")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Iterations per model: {ITERATIONS}")
    print(f"  Providers: {', '.join(enabled_providers)}")
    print(f"  Models ({len(MODELS)}):")
    for m in MODELS:
        print(f"    - {m['label']:<35s} [{m['provider']}: {m['id']}]")
    print(f"  Judge model: {JUDGE_MODEL} ({JUDGE_PROVIDER})")
    print("  Token data source:   DexScreener API (no RPC needed)")
    print(f"  Langfuse: {LANGFUSE_BASE_URL}")
    print("=" * 76)

    print("\n  Step 1: Deterministic decode (EVM decoder skill + DexScreener)...")
    t_decode = time.perf_counter()
    decoded_data = deterministic_decode(CALLDATA)
    decode_ms = (time.perf_counter() - t_decode) * 1000
    print(f"\n  Decoded in {decode_ms:.0f}ms")
    print(f"  Function: {decoded_data['function_signature']}")
    print(f"  Selector: {decoded_data['selector']}")
    print(f"  Params:   {len(decoded_data['parameters'])} (addresses resolved by LLM via lookup_token tool)")
    print(f"\n  This structured decode is the FIXED INPUT for all LLM calls.\n")

    ref_trace = langfuse.start_observation(
        name="evm-decoder-reference-decode",
        as_type="span",
        input=CALLDATA,
        output=decoded_data,
        metadata={
            "user_id": "benchmark-runner",
            "tags": ["benchmark", "reference", "deterministic-decode"],
            "decode_time_ms": round(decode_ms, 2),
        },
    )
    print(f"  Reference trace: {ref_trace.id}")
    ref_trace.end()

    all_results: dict[str, list[IterationResult]] = {}

    for model_cfg in MODELS:
        label = model_cfg["label"]
        print(f"\n{'━' * 76}")
        print(f"  Benchmarking: {label} ({model_cfg['id']})")
        print(f"  Task: Generate intent JSON (summary + details) from structured decode")
        print(f"{'━' * 76}")

        model_results: list[IterationResult] = []
        for i in range(1, ITERATIONS + 1):
            try:
                r = run_iteration(model_cfg, i, decoded_data)
            except Exception as e:
                r = IterationResult(
                    iteration=i,
                    model_id=model_cfg["id"],
                    model_label=label,
                    generation_error=f"Unhandled: {type(e).__name__}: {e}",
                )
                traceback.print_exc()

            model_results.append(r)
            print_iteration(r)

        all_results[label] = model_results
        print_model_summary(label, model_results)

    if len(all_results) > 1:
        print("\n" + "=" * 76)
        print("  CROSS-MODEL COMPARISON")
        print("=" * 76)
        print(f"  {'Model':<25s} {'Pass%':>7s} {'ErrRate':>8s} {'AvgScore':>9s} {'AvgGen':>9s}")
        print(f"  {'─'*25} {'─'*7} {'─'*8} {'─'*9} {'─'*9}")
        for label, results in all_results.items():
            n = len(results)
            passed = sum(1 for r in results if r.passed)
            scores = [r.overall_score for r in results if r.overall_score > 0]
            times = [r.generation_time_ms for r in results if r.generation_time_ms > 0]
            avg_score = f"{statistics.mean(scores):.2f}" if scores else "N/A"
            avg_time = f"{statistics.mean(times):.0f}ms" if times else "N/A"
            print(
                f"  {label:<25s} "
                f"{passed/n*100:>6.1f}% "
                f"{(n-passed)/n*100:>7.1f}% "
                f"{avg_score:>9s} "
                f"{avg_time:>9s}"
            )
        print("=" * 76)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    score_cols = [f for f in SCORE_FIELDS if f != "overall"]

    output_path = os.getenv(
        "BENCHMARK_OUTPUT_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results.csv"),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp", "model_label", "model_id", "iteration",
        "passed", "overall_score",
        *score_cols,
        "generation_time_ms", "judge_time_ms", "total_time_ms",
        "generation_error", "intent_parse_error", "judge_error", "judge_score_errors",
        "trace_id", "parsed_intent",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label, results in all_results.items():
            for r in results:
                row = asdict(r)
                row["timestamp"] = run_timestamp
                for col in score_cols:
                    row[col] = (r.judge_scores or {}).get(col, "")
                score_errors = (r.judge_scores or {}).get("errors") or []
                row["judge_score_errors"] = " | ".join(score_errors)
                row["parsed_intent"] = json.dumps(r.parsed_intent, ensure_ascii=False) if r.parsed_intent else ""
                writer.writerow(row)

    print(f"\n  Results saved → {output_path}")
    print("  Flushing traces to Langfuse...")
    langfuse.flush()
    print("  Done! View traces at:")
    print(f"  → {LANGFUSE_BASE_URL}/project/{LANGFUSE_PROJECT_ID}/traces")
    print(f"  Filter by tag: benchmark, human-readable, or model ID")


if __name__ == "__main__":
    main()
