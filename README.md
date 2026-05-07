# EVM Decoder Benchmark

Measures how accurately LLMs convert raw EVM calldata into human-readable intent JSON.

## How it works

1. **Deterministic decode** — parses calldata ABI, resolves token metadata via RPC
2. **LLM generation** — each model produces `{ "intent": { "summary", "details" } }` from the structured decode
3. **LLM-as-judge** — Claude scores each output for accuracy, clarity, and grounding; results traced to Langfuse

## Setup

```bash
pip install langfuse openai requests python-dotenv
cp .env.example .env  # fill in keys
python -m benchmark
```

## Configuration

Edit `benchmark/config.py` to change `CALLDATA`, `BENCHMARK_GROUND_TRUTH`, `ITERATIONS`, or the model list.
