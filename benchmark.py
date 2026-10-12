#!/usr/bin/env python3
"""Token-consumption sweep across patterns and configs.

Runs each config N times, records per-pattern token usage (via USAGE_LOG)
and wall-clock time, writes one row per run to benchmark_results.csv,
then prints mean +/- std of total tokens per config.

Stub mode by default (no API keys needed). With a real model, token counts
come from resp.usage and repeats actually vary; in stub mode the token
estimates are deterministic (fixed templates), so repeats only vary
wall-clock time.

Usage:
    python3 benchmark.py            # 3 repeats per config
    python3 benchmark.py 5          # 5 repeats per config
    LLM_BASE_URL=... python3 benchmark.py   # against a real model server
"""
import asyncio
import csv
import statistics
import sys
import time

from parallel_decompose import (
    fan_out_gather,
    best_of_n,
    tree_expand,
    reset_usage,
    usage_summary,
)

TASK = "Design a rate limiter for 100k QPS"

# (label, coroutine factory). Covers:
#   H1 fan-out  ~ linear in #subtasks
#   H2 best-of-N ~ linear in N
#   H3 tree     ~ multiplicative in depth x branch x beam
#   (1b) planner parallelism cost vs k
CONFIGS = [
    ("fanout/subtasks=2", lambda: fan_out_gather(TASK, planner_k=1, n_subtasks=2)),
    ("fanout/subtasks=4", lambda: fan_out_gather(TASK, planner_k=1, n_subtasks=4)),
    ("fanout/subtasks=8", lambda: fan_out_gather(TASK, planner_k=1, n_subtasks=8)),
    ("fanout-1b/k=2", lambda: fan_out_gather(TASK, planner_k=2)),
    ("fanout-1b/k=3", lambda: fan_out_gather(TASK, planner_k=3)),
    ("fanout-1b/k=5", lambda: fan_out_gather(TASK, planner_k=5)),
    ("bestofn/n=2", lambda: best_of_n(TASK, n=2)),
    ("bestofn/n=4", lambda: best_of_n(TASK, n=4)),
    ("bestofn/n=8", lambda: best_of_n(TASK, n=8)),
    ("tree/d2-b2-beam2", lambda: tree_expand(TASK, depth=2, branch=2, beam=2)),
    ("tree/d2-b3-beam2", lambda: tree_expand(TASK, depth=2, branch=3, beam=2)),
    ("tree/d3-b2-beam2", lambda: tree_expand(TASK, depth=3, branch=2, beam=2)),
    ("tree/d2-b2-beam3", lambda: tree_expand(TASK, depth=2, branch=2, beam=3)),
    ("tree/d3-b3-beam3", lambda: tree_expand(TASK, depth=3, branch=3, beam=3)),
]


async def run_once(factory):
    reset_usage()
    t0 = time.perf_counter()
    await factory()  # result content is stub-fake; we only measure cost
    dt = time.perf_counter() - t0
    s = usage_summary()
    return {
        "calls": s["calls"],
        "prompt_tokens": s["prompt_tokens"],
        "completion_tokens": s["completion_tokens"],
        "total_tokens": s["prompt_tokens"] + s["completion_tokens"],
        "wall_seconds": round(dt, 2),
    }


async def main(repeats: int = 3):
    rows = []
    for label, factory in CONFIGS:
        for r in range(repeats):
            m = await run_once(factory)
            rows.append({"config": label, "repeat": r, **m})
            print(f"{label:22s} rep={r} calls={m['calls']:3d} "
                  f"tok={m['total_tokens']:5d} wall={m['wall_seconds']:.1f}s",
                  flush=True)

    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean total tokens per config ===")
    by_config: dict = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row["total_tokens"])
    for label, vals in by_config.items():
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{label:22s} {mean:8.1f} +/- {sd:5.1f}")
    print("\nwrote benchmark_results.csv")


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(repeats))
