"""
Parallel task decomposition — the core logic behind "parallel thinking".

DeepThink / Grok-4-Heavy style parallelism is NOT chain-of-thought.
It's: decompose -> run subtasks CONCURRENTLY -> aggregate.

This file is runnable with no API keys: the "llm_call" is a stub.
Replace it with a real async LLM client to use it for real work.
Set LLM_BASE_URL / QWEN_WORKER / QWEN_PLANNER env vars to point at a
real OpenAI-compatible server (Ollama, vLLM, OpenRouter...).

Three parallel patterns are shown:
  1. fan_out_gather  — independent subtasks, run concurrently, then merge
  1b. decompose_parallel — K planners propose decompositions concurrently,
                           a judge picks the best plan before fan-out
  2. best_of_n       — same task, N independent attempts, verifier picks best
  3. tree_expand     — branch at each step, prune losers, continue winners
"""
import asyncio
import os
import random
from dataclasses import dataclass, field
from typing import List, Callable, Awaitable


# ---------------------------------------------------------------------------
# Heterogeneous Qwen models.
#
# Parallelism is NOT a model feature -- any Qwen model works in this
# framework. What matters is cost vs. smarts per role:
#   - MODEL_WORKER  (7B):  cheap + fast, survives N-way fan-out (12 calls)
#   - MODEL_PLANNER (32B): smart, used where calls are few (plan/judge)
#
# Names are Ollama defaults; override with env, e.g.
#   QWEN_WORKER=qwen2.5-coder:7b  QWEN_PLANNER=qwen2.5-coder:32b
# For OpenRouter etc: LLM_BASE_URL=https://openrouter.ai/api/v1
#   QWEN_WORKER=qwen/qwen-2.5-coder-7b-instruct
# ---------------------------------------------------------------------------
MODEL_WORKER = os.environ.get("QWEN_WORKER", "qwen2.5-coder:7b")
MODEL_PLANNER = os.environ.get("QWEN_PLANNER", "qwen2.5-coder:32b")

try:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("LLM_API_KEY", "ollama"),
    )
except ImportError:  # no `openai` package -> fall back to stub below
    _client = None

_sem = asyncio.Semaphore(int(os.environ.get("LLM_MAX_CONCURRENCY", "8")))


# ---------------------------------------------------------------------------
# Usage tracking: measure token consumption per call, per role, per pattern.
#
# In real mode the counts come from resp.usage. In stub mode they are
# estimated with a len//4 heuristic so the measurement pipeline works
# end-to-end before a real model is wired up. Call reset_usage() before a
# pattern run and print_usage_summary() after to get per-pattern totals --
# this is how the token-consumption hypotheses get tested:
#   H1 fan-out  ~ linear in #subtasks
#   H2 best-of-N ~ linear in N, quality gains diminish
#   H3 tree     ~ multiplicative in depth x branch x beam
#   H4 heterogeneous: 32B calls are few (<10%) but cap quality
# ---------------------------------------------------------------------------
USAGE_LOG: List[dict] = []


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def reset_usage() -> None:
    USAGE_LOG.clear()


def usage_summary() -> dict:
    by_tag: dict = {}
    for e in USAGE_LOG:
        d = by_tag.setdefault(e["tag"], {"calls": 0, "prompt": 0, "completion": 0})
        d["calls"] += 1
        d["prompt"] += e["prompt_tokens"]
        d["completion"] += e["completion_tokens"]
    return {
        "calls": len(USAGE_LOG),
        "prompt_tokens": sum(e["prompt_tokens"] for e in USAGE_LOG),
        "completion_tokens": sum(e["completion_tokens"] for e in USAGE_LOG),
        "by_tag": by_tag,
    }


def print_usage_summary() -> None:
    s = usage_summary()
    print(f"--- usage: {s['calls']} calls, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens ---")
    for tag, d in sorted(s["by_tag"].items()):
        print(f"  {tag:12s} x{d['calls']:2d}  in={d['prompt']:5d}  out={d['completion']:5d}")


async def llm_call(prompt: str, tag: str = "",
                   model: str = MODEL_WORKER,
                   max_tokens: int = 512,
                   temperature: float = 0.7) -> str:
    """One LLM call. Real client when `openai` pkg + server exist, else stub."""
    if _client is None:
        await asyncio.sleep(random.uniform(0.3, 0.8))  # network latency
        out = f"[{tag}:{model}] result for: {prompt[:40]}..."
        USAGE_LOG.append({"tag": tag or "untagged", "model": model,
                          "prompt_tokens": _estimate_tokens(prompt),
                          "completion_tokens": _estimate_tokens(out),
                          "estimated": True})
        return out
    async with _sem:  # bound concurrency: parallel, not unbounded
        resp = await _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,      # per-call token budget (dev-set ceiling)
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        u = resp.usage  # real token counts when the server reports them
        USAGE_LOG.append({"tag": tag or "untagged", "model": model,
                          "prompt_tokens": u.prompt_tokens if u else _estimate_tokens(prompt),
                          "completion_tokens": u.completion_tokens if u else _estimate_tokens(content),
                          "estimated": u is None})
        return content


# ---------------------------------------------------------------------------
# Pattern 1: FAN-OUT / FAN-IN (this is "parallel decomposition")
#
#   task ──decompose──> [sub1, sub2, sub3]
#                           │      │      │   <- all run at the same time
#                           v      v      v
#                        [res1, res2, res3] ──aggregate──> final answer
#
# The ONLY thing that makes it parallel is asyncio.gather.
# Total wall-clock time ≈ max(subtask times), not sum.
# ---------------------------------------------------------------------------
async def decompose(task: str, seed: int = 0, n_subtasks: int = 3) -> List[str]:
    # Planner role -> big model, low temperature, bigger budget.
    # Temperature varies per seed so parallel planners actually diverge
    # (same trick as best-of-N attempts varying temperature per seed).
    plan_text = await llm_call(f"Break into independent subtasks: {task}",
                               tag=f"planner-{seed}",
                               model=MODEL_PLANNER,
                               temperature=0.3 + 0.1 * (seed % 3),
                               max_tokens=1024)
    # Real code: parse the model's numbered list (like LangChain's PlanningOutputParser).
    # Stub: pretend it returned n_subtasks subtasks (tagged by seed so the
    # demo shows divergence). n_subtasks is stub-only; real parsing decides.
    _ = plan_text
    return [f"{task} :: subtask {i} [plan-{seed}]" for i in range(1, n_subtasks + 1)]


async def judge_pick(task: str, plans: List[List[str]]) -> List[str]:
    """Pick the best decomposition out of K planner candidates.

    Judge role -> big model: plan quality caps the whole fan-out, so the
    judge gets the smart model and a low temperature. Returns the winning
    plan; falls back to plans[0] if the judge output is unparseable.
    """
    rendered = "\n\n".join(
        f"Plan {i}:\n" + "\n".join(f"  - {s}" for s in plan)
        for i, plan in enumerate(plans)
    )
    verdict = await llm_call(
        f"Task: {task}\nCandidate decompositions:\n{rendered}\n"
        f"Which plan splits the task into the most independent and complete "
        f"subtasks? Reply with ONLY the plan number (0-{len(plans) - 1}).",
        tag="plan-judge", model=MODEL_PLANNER, temperature=0.2, max_tokens=16,
    )
    try:
        idx = int(verdict.strip().split()[0])
        if 0 <= idx < len(plans):
            return plans[idx]
    except (ValueError, IndexError):
        pass
    return plans[0]  # stub mode or unparseable judge output -> safe fallback


async def decompose_parallel(task: str, k: int = 3, n_subtasks: int = 3) -> List[str]:
    """Parallel planners (planner-level best-of-N).

    K planners propose decompositions CONCURRENTLY, then a judge picks the
    best plan before fan-out begins.

    Cost note: the planner is the most expensive role (big model), so this
    costs K planner calls + 1 judge call. Worth it only when decomposition
    quality has high variance; if the planner is already stable, keep K=1.
    """
    plans: List[List[str]] = await asyncio.gather(
        *[decompose(task, seed=i, n_subtasks=n_subtasks) for i in range(k)]
    )
    return await judge_pick(task, plans)


async def solve_subtask(subtask: str, context: str = "") -> str:
    # KEY DETAIL: each branch gets its OWN isolated context.
    # Branches must not share mutable conversation history,
    # otherwise they contaminate each other.
    # Worker role -> small model, low temperature for determinism.
    return await llm_call(f"Solve: {subtask} | shared context: {context}", tag="worker",
                          temperature=0.2)


async def aggregate(task: str, subtasks: List[str], results: List[str]) -> str:
    joined = "\n".join(f"- {s} => {r}" for s, r in zip(subtasks, results))
    # Synthesis reads everything -> big model, generous budget.
    return await llm_call(f"Original task: {task}\nSubtask results:\n{joined}\nSynthesize.",
                          tag="aggregator", model=MODEL_PLANNER,
                          temperature=0.3, max_tokens=2048)


async def fan_out_gather(task: str, planner_k: int = 1, n_subtasks: int = 3) -> str:
    # planner_k=1 -> single planner (original behavior).
    # planner_k>1 -> K planners race concurrently, judge picks the best plan.
    # n_subtasks is stub-only (real parsing decides the count).
    subtasks = await (decompose_parallel(task, k=planner_k, n_subtasks=n_subtasks)
                      if planner_k > 1
                      else decompose(task, n_subtasks=n_subtasks))
    # ---- THIS IS THE PARALLELISM ----
    # asyncio.gather schedules all coroutines concurrently on one event loop.
    # No threads, no processes: concurrency comes from awaiting I/O (the LLM API).
    results: List[str] = await asyncio.gather(
        *[solve_subtask(s) for s in subtasks]
    )
    # ---- fan-in: merge ----
    return await aggregate(task, subtasks, results)


# ---------------------------------------------------------------------------
# Pattern 2: BEST-OF-N (competitive parallelism)
#
#   task ──┬──> attempt_1 ──╮
#          ├──> attempt_2 ──╬──> verifier scores all ──> pick best
#          └──> attempt_3 ──╯
#
# Same task, N independent tries (different temperature seeds / prompts).
# Used when there is an objective way to check quality:
# unit tests for code, exact match for math, reward model otherwise.
# ---------------------------------------------------------------------------
async def attempt(task: str, seed: int) -> str:
    # Worker role; vary temperature per seed so N attempts actually diverge.
    return await llm_call(f"Solve (try #{seed}): {task}", tag=f"attempt-{seed}",
                          temperature=0.7 + 0.1 * (seed % 3))


async def verifier(task: str, candidate: str) -> float:
    # Real code: run unit tests, or ask a reward model / LLM-judge to score.
    # Judge role -> big model: vote quality caps the whole best-of-N.
    score_text = await llm_call(f"Score 0-1: does this solve '{task}'? {candidate}",
                                tag="verifier", model=MODEL_PLANNER, temperature=0.2)
    _ = score_text
    return random.random()  # stub


async def best_of_n(task: str, n: int = 4) -> str:
    candidates = await asyncio.gather(*[attempt(task, i) for i in range(n)])
    scores = await asyncio.gather(*[verifier(task, c) for c in candidates])
    best = max(zip(candidates, scores), key=lambda cs: cs[1])
    return best[0]


# ---------------------------------------------------------------------------
# Pattern 3: TREE EXPANSION with pruning (Tree-of-Thoughts style)
#
#   step0 ──┬──> a1 ──┬──> a1b1 ──╮
#           │         └──> a1b2 ──╬──> prune to top-k ──> continue...
#           └──> a2 ──...
#
# Parallelism happens at EVERY level: each node expands k children
# concurrently, a value function scores them, losers are dropped.
# This is the closest public analogue to what "DeepThink" does internally.
# ---------------------------------------------------------------------------
@dataclass
class Node:
    state: str
    score: float = 0.0
    children: List["Node"] = field(default_factory=list)


async def propose_next_states(node: Node, k: int = 3) -> List[str]:
    # One LLM call asking for k diverse continuations, or k parallel calls.
    # Worker role; high temperature for diverse branching.
    proposals = await asyncio.gather(
        *[llm_call(f"Propose continuation #{i} from: {node.state}", tag="proposer",
                   temperature=0.8)
          for i in range(k)]
    )
    return proposals


async def value_fn(state: str) -> float:
    # Estimates "how promising is this partial reasoning path".
    # Real code: a trained value head, or an LLM asked to rate the state.
    # Judge role -> big model.
    _ = await llm_call(f"Rate promise 0-1 of: {state}", tag="value",
                       model=MODEL_PLANNER, temperature=0.2)
    return random.random()  # stub


async def tree_expand(root_task: str, depth: int = 2, branch: int = 3,
                      beam: int = 2) -> Node:
    """Beam search over reasoning paths. Parallel expand + prune per level."""
    frontier: List[Node] = [Node(state=root_task)]
    for _ in range(depth):
        # Expand EVERY frontier node concurrently...
        all_proposals: List[List[str]] = await asyncio.gather(
            *[propose_next_states(node, branch) for node in frontier]
        )
        candidates: List[Node] = []
        for node, proposals in zip(frontier, all_proposals):
            for p in proposals:
                child = Node(state=f"{node.state} -> {p}")
                node.children.append(child)
                candidates.append(child)
        # ...then score all candidates concurrently and keep top-k (prune).
        scores = await asyncio.gather(*[value_fn(c.state) for c in candidates])
        for c, s in zip(candidates, scores):
            c.score = s
        candidates.sort(key=lambda c: c.score, reverse=True)
        frontier = candidates[:beam]
    return max(frontier, key=lambda n: n.score)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def main() -> None:
    task = "Design a rate limiter for 100k QPS"

    print("== Pattern 1: fan-out / fan-in ==")
    reset_usage()
    print(await fan_out_gather(task))
    print_usage_summary()
    print()

    print("== Pattern 1b: fan-out / fan-in with parallel planners (k=3) ==")
    reset_usage()
    print(await fan_out_gather(task, planner_k=3))
    print_usage_summary()
    print()

    print("== Pattern 2: best-of-N ==")
    reset_usage()
    print(await best_of_n(task, n=3))
    print_usage_summary()
    print()

    print("== Pattern 3: tree expansion ==")
    reset_usage()
    winner = await tree_expand(task, depth=2, branch=2, beam=2)
    print("best path:", winner.state, "score:", round(winner.score, 3))
    print_usage_summary()


if __name__ == "__main__":
    asyncio.run(main())
