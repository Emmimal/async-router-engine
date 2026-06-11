"""
async_router.py
---------------
Routes LLM requests to the best available provider in real time.
On throttle detection, snapshots state and hot-swaps to a fallback
model with full Strategy B payload normalization.

Pure Python asyncio. Zero external dependencies.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from router.rate_limit_detector import RateLimitDetector
from router.model_registry import ModelRegistry
from router.state_preserver import StatePreserver


class RouteOutcome(Enum):
    SUCCESS          = "success"
    THROTTLED_ROUTED = "throttled_routed"
    ALL_THROTTLED    = "all_throttled"
    MAX_RETRIES      = "max_retries"
    ERROR            = "error"


@dataclass
class RouteResult:
    outcome: RouteOutcome
    model_used: str
    attempts: int
    total_latency_ms: float
    response: Optional[Dict[str, Any]]
    swapped: bool
    swap_count: int
    state_preserved: bool
    error_message: Optional[str] = None
    snapshot_id: Optional[str] = None


@dataclass
class RouterConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 0.1
    max_swaps: int = 2
    swap_delay_seconds: float = 0.05


class AsyncRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        detector: RateLimitDetector,
        preserver: StatePreserver,
        config: Optional[RouterConfig] = None,
    ):
        self._registry = registry
        self._detector = detector
        self._preserver = preserver
        self._config = config or RouterConfig()
        self._route_count = 0
        self._swap_count = 0
        self._success_count = 0

    async def route(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        mock_provider: Callable,
        agent_id: str = "agent",
        task_id: str = "task",
        current_step: int = 0,
        total_steps: int = 1,
        step_label: str = "step",
        partial_output: Optional[Dict[str, Any]] = None,
        task_context: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        preferred_model: Optional[str] = None,
    ) -> RouteResult:
        self._route_count += 1
        t_start = time.monotonic()
        partial_output = partial_output or {}
        task_context = task_context or {}

        tried_models: List[str] = []
        swap_count = 0
        snapshot_id = None
        current_messages = list(messages)

        if preferred_model and not self._detector.is_throttled(preferred_model):
            model = self._registry.get(preferred_model)
        else:
            model = self._registry.next_available(exclude=tried_models)

        if model is None:
            return RouteResult(
                outcome=RouteOutcome.ALL_THROTTLED,
                model_used="none",
                attempts=0,
                total_latency_ms=0.0,
                response=None,
                swapped=False,
                swap_count=0,
                state_preserved=False,
                error_message="No models available at startup",
            )

        attempts = 0

        while attempts < self._config.max_retries:
            attempts += 1

            if self._detector.is_throttled(model.model_id):
                tried_models.append(model.model_id)
                next_model = self._registry.next_available(exclude=tried_models)
                if next_model is None:
                    break
                model = next_model
                continue

            payload = self._registry.adapt_payload(
                target_model_id=model.model_id,
                system_prompt=system_prompt,
                messages=current_messages,
            )

            self._detector.record_request(model.model_id)

            try:
                response, error_msg = await mock_provider(
                    model_id=model.model_id,
                    payload=payload,
                )
            except Exception as e:
                error_msg = str(e)
                response = None

            if error_msg:
                throttle_event = self._detector.inspect(
                    provider_id=model.model_id,
                    error_message=error_msg,
                )

                if throttle_event:
                    self._detector.record(throttle_event)

                    state = self._preserver.snapshot(
                        agent_id=agent_id,
                        task_id=task_id,
                        source_model_id=model.model_id,
                        messages=current_messages,
                        system_prompt=system_prompt,
                        current_step=current_step,
                        total_steps=total_steps,
                        step_label=step_label,
                        partial_output=partial_output,
                        task_context=task_context,
                        expected_output_schema=expected_schema,
                    )
                    snapshot_id = state.snapshot_id

                    tried_models.append(model.model_id)
                    if swap_count >= self._config.max_swaps:
                        break

                    next_model = self._registry.next_available(exclude=tried_models)
                    if next_model is None:
                        break

                    resume_hint = self._preserver.build_resume_message(state)
                    current_messages = list(current_messages) + [{
                        "role": "user",
                        "content": resume_hint,
                    }]

                    model = next_model
                    swap_count += 1
                    self._swap_count += 1

                    await asyncio.sleep(self._config.swap_delay_seconds)
                    continue

                else:
                    latency_ms = (time.monotonic() - t_start) * 1000
                    return RouteResult(
                        outcome=RouteOutcome.ERROR,
                        model_used=model.model_id,
                        attempts=attempts,
                        total_latency_ms=round(latency_ms, 2),
                        response=None,
                        swapped=swap_count > 0,
                        swap_count=swap_count,
                        state_preserved=snapshot_id is not None,
                        error_message=error_msg,
                        snapshot_id=snapshot_id,
                    )

            # Success
            self._success_count += 1
            latency_ms = (time.monotonic() - t_start) * 1000
            outcome = (
                RouteOutcome.THROTTLED_ROUTED if swap_count > 0
                else RouteOutcome.SUCCESS
            )
            return RouteResult(
                outcome=outcome,
                model_used=model.model_id,
                attempts=attempts,
                total_latency_ms=round(latency_ms, 2),
                response=response,
                swapped=swap_count > 0,
                swap_count=swap_count,
                state_preserved=snapshot_id is not None,
                snapshot_id=snapshot_id,
            )

        # Exhausted retries
        latency_ms = (time.monotonic() - t_start) * 1000
        all_throttled = all(
            self._detector.is_throttled(m.model_id)
            for m in self._registry.all_models()
        )
        outcome = (
            RouteOutcome.ALL_THROTTLED if all_throttled
            else RouteOutcome.MAX_RETRIES
        )
        return RouteResult(
            outcome=outcome,
            model_used=model.model_id,
            attempts=attempts,
            total_latency_ms=round(latency_ms, 2),
            response=None,
            swapped=swap_count > 0,
            swap_count=swap_count,
            state_preserved=snapshot_id is not None,
            error_message="Exhausted all retry attempts",
            snapshot_id=snapshot_id,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "total_routes": self._route_count,
            "successful_routes": self._success_count,
            "total_swaps": self._swap_count,
            "success_rate": round(
                self._success_count / max(self._route_count, 1), 4
            ),
        }
