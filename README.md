# Parallel Agent Execution Framework

An async Python framework implementing parallel LLM-agent execution patterns — built to make code generation faster and cheaper by running independent agent work concurrently and routing each role to the right-sized model.

## Patterns

1. **Fan-out / fan-in** — decompose a task into independent subtasks, run them concurrently with `asyncio.gather`, then aggregate. Wall-clock time ≈ slowest branch, not the sum.
   - **Parallel planners (1b)** — `fan_out_gather(task, planner_k=K)` runs K planners concurrently (varied temperature seeds so they diverge), then a judge picks the best decomposition before fan-out. Planner is the most expensive role, so this costs K planner calls + 1 judge call; use it when decomposition quality has high variance, keep K=1 otherwise.
2. **Best-of-N** — N independent attempts at the same task (varied temperature seeds), scored by a verifier; the best wins.
3. **Tree expansion with pruning** — beam search over reasoning paths: expand every frontier node concurrently, score all candidates concurrently, keep top-k.

## Heterogeneous model routing

Cost vs. smarts per role, configured via environment variables:

- `QWEN_WORKER` (default `qwen2.5-coder:7b`) — cheap and fast, used for high-fan-out worker branches
- `QWEN_PLANNER` (default `qwen2.5-coder:32b`) — smarter, used for planner / aggregator / verifier / judge roles where call counts are low

## Usage

Runs with zero setup — `llm_call` falls back to an offline stub when the `openai` package or a model server is unavailable:

```bash
python parallel_decompose.py
```

Point it at a real model server (Ollama, vLLM, OpenRouter — anything OpenAI-compatible):

```bash
export LLM_BASE_URL="http://localhost:11434/v1"   # Ollama default
export QWEN_WORKER="qwen2.5-coder:7b"
export QWEN_PLANNER="qwen2.5-coder:32b"
export LLM_MAX_CONCURRENCY="8"                     # semaphore-bounded concurrency
python parallel_decompose.py
```

## Design notes

- Concurrency is `asyncio`-based (no threads, no processes) — it comes from awaiting LLM API I/O.
- Each branch gets its own isolated context; branches never share mutable conversation history.
- Verifier/judge quality caps best-of-N — hence the bigger model on judge roles.
- Temperature is varied per attempt so parallel branches actually diverge.

## Status

Prototype / work in progress. The framework runs end-to-end in stub mode; real-model integration is via environment configuration. No production deployment or benchmark numbers yet.
