import statistics
from collections import Counter

from .config import ITERATIONS
from .prompts import SCORE_FIELDS
from .runner import IterationResult


def print_iteration(r: IterationResult):
    status = "\u2705 PASS" if r.passed else "\u274c FAIL"
    score = f"{r.overall_score}/10" if r.overall_score else "N/A"
    print(
        f"  [{r.iteration:>3d}/{ITERATIONS}] {status}  score={score:<6s} "
        f"gen={r.generation_time_ms:>7.0f}ms  judge={r.judge_time_ms:>7.0f}ms  "
        f"total={r.total_time_ms:>7.0f}ms"
    )
    if r.generation_error:
        print(f"           gen_error: {r.generation_error[:120]}")
    if r.intent_parse_error:
        print(f"           intent_parse: {r.intent_parse_error[:120]}")
    if r.judge_scores and r.judge_scores.get("errors"):
        for err in r.judge_scores["errors"][:2]:
            print(f"           judge: {err[:120]}")


def print_model_summary(model_label: str, results: list[IterationResult]):
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = n - passed
    error_rate = (failed / n) * 100 if n else 0
    gen_errors = sum(1 for r in results if r.generation_error)
    judge_errors = sum(1 for r in results if r.judge_error)

    scores = [r.overall_score for r in results if r.overall_score > 0]
    gen_times = [r.generation_time_ms for r in results if r.generation_time_ms > 0]
    total_times = [r.total_time_ms for r in results if r.total_time_ms > 0]

    print(f"\n{'=' * 76}")
    print(f"  MODEL: {model_label}")
    print(f"{'=' * 76}")
    print(f"  Iterations:          {n}")
    print(f"  Passed:              {passed}  ({passed/n*100:.1f}%)")
    print(f"  Failed:              {failed}  ({error_rate:.1f}%)")
    print(f"  Generation errors:   {gen_errors}")
    print(f"  Judge parse errors:  {judge_errors}")
    print(f"  ERROR RATE:          {error_rate:.1f}%")
    print(f"{'─' * 76}")

    if scores:
        print(f"  Overall Score (/10):")
        print(f"    Mean:    {statistics.mean(scores):.2f}")
        print(f"    Median:  {statistics.median(scores):.1f}")
        if len(scores) > 1:
            print(f"    StdDev:  {statistics.stdev(scores):.2f}")
        print(f"    Min:     {min(scores):.1f}")
        print(f"    Max:     {max(scores):.1f}")

    scored_results = [r for r in results if r.judge_scores]
    if scored_results:
        print(f"{'─' * 76}")
        print(f"  Per-Criterion Averages:")
        for c in SCORE_FIELDS[:-1]:  # exclude "overall" — shown above in aggregate
            vals = [r.judge_scores.get(c, 0) for r in scored_results]
            avg = statistics.mean(vals)
            mn = min(vals)
            bar = "\u2588" * int(avg) + "\u2591" * (10 - int(avg))
            print(f"    {c:<26s} {avg:>5.2f}  min={mn:<4.1f}  {bar}")

    print(f"{'─' * 76}")
    if gen_times:
        sorted_gt = sorted(gen_times)
        m = len(sorted_gt)
        print(f"  Generation Latency:")
        print(f"    Mean:    {statistics.mean(gen_times):>8.0f} ms")
        print(f"    Median:  {statistics.median(gen_times):>8.0f} ms")
        print(f"    P95:     {sorted_gt[min(int(m*0.95), m-1)]:>8.0f} ms")
        print(f"    Min:     {min(gen_times):>8.0f} ms")
        print(f"    Max:     {max(gen_times):>8.0f} ms")
    if total_times:
        print(f"  Total wall time:     {sum(total_times)/1000:>8.1f} s")
    print(f"{'=' * 76}")

    all_errors = [
        err
        for r in results
        if r.judge_scores and r.judge_scores.get("errors")
        for err in r.judge_scores["errors"]
    ]
    if all_errors:
        print(f"\n  Top Errors (across {len(all_errors)} total):")
        for err, count in Counter(all_errors).most_common(10):
            print(f"    [{count:>3d}x] {err[:100]}")
    print()
