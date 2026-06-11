# async-router-engine

A pure-Python async router for resilient multi-agent LLM pipelines — typed rate-limit detection, per-model payload normalization, and mid-task state preservation on provider swap.

![Python Version](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

> Read the full write-up on Towards Data Science →
> **[LLM Rate Limits Corrupt Pipelines — I Built the Recovery Layer That Fixes It](https://towardsdatascience.com/author/emmimalp-alexander/)**

## What It Does

```
Request → RateLimitDetector → adapt_payload() → Provider
                ↓ on throttle
           StatePreserver.snapshot()
                ↓
           next_available model
                ↓
           adapt_payload() for new target + build_resume_message()
                ↓
           Provider (correct payload contract, full context)
```

Four components, one `router.route()` call:

| Component | Job |
|---|---|
| `RateLimitDetector` | Classifies errors into typed `ThrottleEvent`s with per-provider backoff windows |
| `ModelRegistry` | Holds `ModelProfile`s; rebuilds payloads from scratch for each target model |
| `StatePreserver` | Snapshots full agent context before every swap; builds resume messages |
| `AsyncRouter` | Coordinates the three above inside a bounded async retry loop |

## Installation

```bash
git clone https://github.com/Emmimal/async-router-engine.git
cd async-router-engine
```

No packages to install. Pure Python standard library. Python 3.12.

## Quick Start

```python
import asyncio
from router.async_router import AsyncRouter, RouterConfig
from router.model_registry import ModelRegistry
from router.rate_limit_detector import RateLimitDetector
from router.state_preserver import StatePreserver
from agent.mock_provider import MockProvider, ThrottleScenario

async def main():
    router = AsyncRouter(
        registry=ModelRegistry(),
        detector=RateLimitDetector(),
        preserver=StatePreserver(),
        config=RouterConfig(max_retries=3, max_swaps=2),
    )
    provider = MockProvider(scenario=ThrottleScenario.STRATEGY_B, seed=42)

    result = await router.route(
        system_prompt="You are a task executor. Return JSON with 'result' and 'confidence'.",
        messages=[{"role": "user", "content": "Analyse the dataset and return findings."}],
        mock_provider=provider,
        agent_id="executor",
        task_id="run_001",
        current_step=1,
        total_steps=3,
        step_label="Execute planned steps",
        expected_schema={"type": "object", "required": ["result", "confidence"]},
    )
    print(result.outcome.value, result.model_used, result.swapped)

asyncio.run(main())
```

### Running the Three-Agent Pipeline

```python
import asyncio
from agent.mock_provider import MockProvider, ThrottleScenario
from agent.workflow_orchestrator import WorkflowOrchestrator
from router.async_router import RouterConfig

async def main():
    provider = MockProvider(scenario=ThrottleScenario.STRATEGY_B, seed=42)
    orchestrator = WorkflowOrchestrator(
        mock_provider=provider,
        config=RouterConfig(max_retries=4, max_swaps=2),
    )
    result = await orchestrator.run(task_id="pipeline_run_1")
    print(result.completed, result.steps_completed, result.models_used)

asyncio.run(main())
```

## Running the Benchmark

```bash
python benchmarks/benchmark.py
```

Results are identical on every machine. No packages required. Deterministic simulated latency — not wall-clock timing.

## Configuration Reference

```python
@dataclass
class RouterConfig:
    max_retries: int = 3              # Total attempts per route() call
    retry_delay_seconds: float = 0.1  # Pause between non-swap retries
    max_swaps: int = 2                # Maximum model swaps per call
    swap_delay_seconds: float = 0.05  # Pause before hitting a new model after swap
```

```python
RateLimitDetector(
    window_seconds=60.0,  # Rolling window for request rate tracking
    max_requests=100,     # Max requests per window per provider
)
```

## Model Profiles (Built-in)

| Model | Tier | max_tokens | RPM | System prompt | JSON mode |
|---|---|---|---|---|---|
| `model_a` | PRIMARY | 8192 | 60 | ✅ | ✅ |
| `model_b` | SECONDARY | 4096 | 100 | ✅ | ✅ |
| `model_c` | TERTIARY | 2048 | 200 | ❌ | ❌ |

Add a new model by extending `MODEL_PROFILES` in `model_registry.py`. No router logic changes.

## Connecting a Real Provider

Replace `mock_provider` with any async callable matching this signature:

```python
async def my_provider(model_id: str, payload: dict) -> tuple[dict | None, str | None]:
    # payload is already normalized for this model by adapt_payload()
    # return (response_dict, None) on success
    # return (None, error_string) on failure
    ...
```

## Project Structure

```
async-router-engine/
├── __init__.py
├── router/
│   ├── __init__.py
│   ├── async_router.py          # AsyncRouter, RouterConfig, RouteResult, RouteOutcome
│   ├── rate_limit_detector.py   # RateLimitDetector, ThrottleEvent, ThrottleReason (~160 lines)
│   ├── model_registry.py        # ModelRegistry, ModelProfile, adapt_payload()
│   └── state_preserver.py       # StatePreserver, AgentState, build_resume_message() (~140 lines)
├── agent/
│   ├── __init__.py
│   ├── mock_provider.py         # MockProvider — three deterministic throttle scenarios
│   └── workflow_orchestrator.py # WorkflowOrchestrator — Planner → Executor → Validator
└── benchmarks/
    ├── __init__.py
    └── benchmark.py             # NO_ROUTER vs STRATEGY_A vs STRATEGY_B, seed=42
```

## Known Limitations

**State is in-process only.** `StatePreserver` stores snapshots in memory. Replace the dictionary with a SQLite backend for crash recovery — the `snapshot()` / `restore()` interface stays the same.

**No adaptive model selection.** The registry picks the next model by strict priority order. Log `RouteResult` schema integrity rates per model to route adaptively — `stats()` partially supports this already.

**No real provider integration.** The benchmark uses `MockProvider` exclusively. Swapping in a real provider is a one-function change to the provider callable.

## License

MIT
