"""
benchmark.py
------------
Benchmarks three scenarios side by side:
  1. NO_ROUTER   — no fallback, pipeline dies on first 429
  2. STRATEGY_A  — router with no payload normalization (schema collapse)
  3. STRATEGY_B  — router with full adapter normalization (this system)

Latency is measured as simulated latency — the sum of seeded mock provider
delays (base_latency * jitter). This is fully deterministic: identical values
on every machine, every run, regardless of OS scheduling or CPU load.

Run: python benchmarks/benchmark.py
"""

from __future__ import annotations
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from agent.mock_provider import MockProvider, ThrottleScenario
from agent.workflow_orchestrator import WorkflowOrchestrator, PipelineResult
from router.async_router import RouterConfig


async def run_scenario(
    scenario: ThrottleScenario,
    task_id: str,
    n_runs: int = 10,
    throttle_at_step: int = 1,
    seed: int = 42,
) -> List[PipelineResult]:
    results = []
    for i in range(n_runs):
        provider = MockProvider(
            scenario=scenario,
            throttle_model_a_at_step=throttle_at_step,
            latency_ms=20.0,
            seed=seed + i,
        )
        orchestrator = WorkflowOrchestrator(
            mock_provider=provider,
            config=RouterConfig(max_retries=4, max_swaps=2),
        )
        result = await orchestrator.run(task_id=f"{task_id}_{i}")
        # Attach deterministic simulated latency — seeded, OS-independent
        result.simulated_latency_ms = provider.simulated_latency_ms
        results.append(result)
    return results


def summarise(results: List[PipelineResult]) -> Dict[str, Any]:
    n = len(results)
    completed   = sum(1 for r in results if r.completed)
    swapped     = sum(1 for r in results if r.swap_count > 0)
    state_saved = sum(1 for r in results if r.state_preserved)
    # Deterministic simulated latency — identical on every machine, every run
    avg_latency = sum(r.simulated_latency_ms for r in results) / n
    avg_steps   = sum(r.steps_completed for r in results) / n

    schema_ok = 0
    for r in results:
        all_ok = True
        for step_r in r.step_results:
            if step_r.response:
                if (step_r.response.get("confidence") is None or
                        not step_r.response.get("result")):
                    all_ok = False
                    break
        if all_ok:
            schema_ok += 1

    return {
        "n_runs": n,
        "completion_rate":          round(completed / n, 3),
        "swap_rate":                round(swapped / n, 3),
        "state_preservation_rate":  round(state_saved / n, 3),
        "schema_integrity_rate":    round(schema_ok / n, 3),
        "avg_latency_ms":           round(avg_latency, 2),
        "avg_steps_completed":      round(avg_steps, 2),
        "completed": completed,
        "failed":    n - completed,
    }


def print_benchmark(summaries: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 75)
    print("  ASYNC ROUTER BENCHMARK RESULTS (seed=42, 10 runs per scenario)")
    print("  Latency = simulated (seeded, deterministic, OS-independent)")
    print("=" * 75)

    metrics = [
        ("completion_rate",         "Completion Rate"),
        ("schema_integrity_rate",   "Schema Integrity Rate"),
        ("state_preservation_rate", "State Preserved Rate"),
        ("swap_rate",               "Provider Swap Rate"),
        ("avg_latency_ms",          "Avg Simulated Latency (ms)"),
        ("avg_steps_completed",     "Avg Steps Completed"),
    ]

    col_w = 22
    header = f"{'Metric':<34}" + "".join(f"{k:>{col_w}}" for k in summaries)
    print(header)
    print("-" * (34 + col_w * len(summaries)))

    for key, label in metrics:
        row = f"{label:<34}"
        for scenario_name, data in summaries.items():
            val = data[key]
            if isinstance(val, float) and key.endswith("_rate"):
                cell = f"{val:.1%}"
            elif isinstance(val, float):
                cell = f"{val:.2f}"
            else:
                cell = str(val)
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * 75)
    print("\nKey findings:")
    no_r    = summaries["NO_ROUTER"]
    strat_b = summaries["STRATEGY_B"]
    improvement = strat_b["completion_rate"] - no_r["completion_rate"]
    print(f"  Completion rate improvement:  "
          f"+{improvement:.1%} (NO_ROUTER → STRATEGY_B)")
    schema_diff = (strat_b["schema_integrity_rate"] -
                   summaries["STRATEGY_A"]["schema_integrity_rate"])
    print(f"  Schema integrity improvement: "
          f"+{schema_diff:.1%} (STRATEGY_A → STRATEGY_B)")
    print(f"  Strategy B swap overhead:     "
          f"50ms per failover event (swap_delay_seconds=0.05, configurable)")
    print("=" * 75 + "\n")


async def main():
    print("Running benchmark (10 runs × 3 scenarios)...")

    no_router  = await run_scenario(ThrottleScenario.NO_ROUTER,  "no_router",  n_runs=10)
    strategy_a = await run_scenario(ThrottleScenario.STRATEGY_A, "strategy_a", n_runs=10)
    strategy_b = await run_scenario(ThrottleScenario.STRATEGY_B, "strategy_b", n_runs=10)

    summaries = {
        "NO_ROUTER":  summarise(no_router),
        "STRATEGY_A": summarise(strategy_a),
        "STRATEGY_B": summarise(strategy_b),
    }

    print_benchmark(summaries)
    return summaries


if __name__ == "__main__":
    asyncio.run(main())
