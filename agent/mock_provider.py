"""
mock_provider.py
----------------
Simulates LLM provider responses with configurable throttle scenarios.
No real API calls. Seed=42 for reproducibility.

Supports three throttle scenarios:
  - NO_ROUTER:   Primary model throttles, pipeline dies (baseline)
  - STRATEGY_A:  Router swaps but no normalization (schema collapse)
  - STRATEGY_B:  Router swaps with full adapter normalization
"""

from __future__ import annotations
import asyncio
import random
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ThrottleScenario(Enum):
    NO_ROUTER  = "no_router"
    STRATEGY_A = "strategy_a"
    STRATEGY_B = "strategy_b"


MODEL_RESPONSES: Dict[str, Dict[str, Any]] = {
    "model_a": {
        "result": "Pipeline step completed with full structured analysis.",
        "confidence": 0.94,
        "metadata": {"tokens_used": 312, "model_tier": "primary"},
    },
    "model_b": {
        "result": "Step completed. Analysis attached.",
        "confidence": 0.87,
        "metadata": {"tokens_used": 198, "model_tier": "secondary"},
    },
    "model_c": {
        "result": "Task done. See output.",
        "confidence": 0.71,
        "metadata": {"tokens_used": 89, "model_tier": "tertiary"},
    },
}

STRATEGY_A_DEGRADED: Dict[str, Any] = {
    "result": "incomplete - schema mismatch during swap",
    "confidence": None,
    "metadata": {},
}


class MockProvider:
    def __init__(
        self,
        scenario: ThrottleScenario,
        throttle_model_a_at_step: int = 1,
        throttle_model_b_also: bool = False,
        latency_ms: float = 50.0,
        seed: int = 42,
    ):
        self._scenario = scenario
        self._throttle_a_at = throttle_model_a_at_step
        self._throttle_b_also = throttle_model_b_also
        self._base_latency_ms = latency_ms          # keep in ms for tracking
        self._base_latency = latency_ms / 1000.0    # seconds for asyncio.sleep
        self._rng = random.Random(seed)
        self._call_counts: Dict[str, int] = {}
        self._simulated_latency_ms: float = 0.0     # seeded, deterministic accumulator

    @property
    def simulated_latency_ms(self) -> float:
        """Total simulated latency in ms — seeded, deterministic, OS-independent."""
        return self._simulated_latency_ms

    async def __call__(
        self,
        model_id: str,
        payload: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        self._call_counts[model_id] = self._call_counts.get(model_id, 0) + 1
        call_num = self._call_counts[model_id]

        jitter = self._rng.uniform(0.8, 1.2)
        # Accumulate the intended simulated latency before the actual sleep
        self._simulated_latency_ms += self._base_latency_ms * jitter
        await asyncio.sleep(self._base_latency * jitter)

        if model_id == "model_a" and call_num >= self._throttle_a_at:
            return None, "429 Too Many Requests: rate limit exceeded"

        if model_id == "model_b" and self._throttle_b_also and call_num >= 1:
            return None, "429 Too Many Requests: rate limit exceeded"

        if self._scenario == ThrottleScenario.NO_ROUTER:
            if model_id != "model_a":
                return None, "503 Service Unavailable: no router configured"

        if self._scenario == ThrottleScenario.STRATEGY_A:
            if model_id in ("model_b", "model_c"):
                return STRATEGY_A_DEGRADED, None

        response = dict(MODEL_RESPONSES.get(model_id, MODEL_RESPONSES["model_c"]))
        return response, None

    def call_stats(self) -> Dict[str, int]:
        return dict(self._call_counts)
