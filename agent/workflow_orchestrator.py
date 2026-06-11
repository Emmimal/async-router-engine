"""
workflow_orchestrator.py
------------------------
Runs a 3-agent pipeline (Planner → Executor → Validator) using
the AsyncRouter for resilient execution.

Pure Python asyncio. Zero dependencies.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from router.rate_limit_detector import RateLimitDetector
from router.model_registry import ModelRegistry
from router.state_preserver import StatePreserver
from router.async_router import AsyncRouter, RouteResult, RouteOutcome, RouterConfig


@dataclass
class AgentStep:
    agent_id: str
    label: str
    system_prompt: str
    user_message: str
    expected_schema: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    task_id: str
    completed: bool
    steps_completed: int
    total_steps: int
    total_latency_ms: float
    swap_count: int
    state_preserved: bool
    models_used: List[str]
    step_results: List[RouteResult]
    failure_reason: Optional[str] = None

    def completion_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.steps_completed / self.total_steps


def default_pipeline_steps() -> List[AgentStep]:
    return [
        AgentStep(
            agent_id="planner",
            label="Plan task decomposition",
            system_prompt=(
                "You are a task planner. Decompose the user request into "
                "a structured execution plan. Return JSON with 'plan' and "
                "'confidence' fields."
            ),
            user_message=(
                "Plan the execution of: Analyse EmiTechLogic tutorial "
                "content and generate a structured learning path."
            ),
            expected_schema={"type": "object", "required": ["result", "confidence"]},
        ),
        AgentStep(
            agent_id="executor",
            label="Execute planned steps",
            system_prompt=(
                "You are a task executor. Execute the provided plan and "
                "produce a structured result. Return JSON with 'result', "
                "'confidence', and 'metadata' fields."
            ),
            user_message=(
                "Execute the plan from the previous step and produce the "
                "structured learning path output."
            ),
            expected_schema={"type": "object", "required": ["result", "confidence"]},
        ),
        AgentStep(
            agent_id="validator",
            label="Validate and finalise output",
            system_prompt=(
                "You are a quality validator. Check the executor output for "
                "completeness and correctness. Return JSON with 'result', "
                "'confidence', and 'validation_passed' fields."
            ),
            user_message=(
                "Validate the output from the executor step and confirm "
                "it meets quality requirements."
            ),
            expected_schema={"type": "object", "required": ["result", "confidence"]},
        ),
    ]


class WorkflowOrchestrator:
    def __init__(
        self,
        mock_provider: Callable,
        config: Optional[RouterConfig] = None,
    ):
        self._registry  = ModelRegistry()
        self._detector  = RateLimitDetector()
        self._preserver = StatePreserver()
        self._router = AsyncRouter(
            registry=self._registry,
            detector=self._detector,
            preserver=self._preserver,
            config=config or RouterConfig(),
        )
        self._mock_provider = mock_provider

    async def run(
        self,
        task_id: str,
        steps: Optional[List[AgentStep]] = None,
    ) -> PipelineResult:
        steps = steps or default_pipeline_steps()
        total_steps = len(steps)
        t_start = time.monotonic()

        completed_steps = 0
        swap_count = 0
        state_preserved = False
        models_used: List[str] = []
        step_results: List[RouteResult] = []
        partial_output: Dict[str, Any] = {}
        messages: List[Dict[str, str]] = []
        failure_reason: Optional[str] = None

        for i, step in enumerate(steps):
            messages.append({"role": "user", "content": step.user_message})

            result = await self._router.route(
                system_prompt=step.system_prompt,
                messages=messages,
                mock_provider=self._mock_provider,
                agent_id=step.agent_id,
                task_id=task_id,
                current_step=i,
                total_steps=total_steps,
                step_label=step.label,
                partial_output=partial_output,
                task_context={"step_index": i, "agent": step.agent_id},
                expected_schema=step.expected_schema,
            )

            step_results.append(result)
            models_used.append(result.model_used)

            if result.swapped:
                swap_count += result.swap_count
            if result.state_preserved:
                state_preserved = True

            if result.outcome in (
                RouteOutcome.ALL_THROTTLED,
                RouteOutcome.MAX_RETRIES,
                RouteOutcome.ERROR,
            ):
                failure_reason = (
                    f"Step {i+1} ({step.label}) failed: "
                    f"{result.outcome.value} — {result.error_message}"
                )
                break

            if result.response:
                partial_output[step.agent_id] = result.response
                messages.append({
                    "role": "assistant",
                    "content": str(result.response.get("result", "")),
                })

            completed_steps += 1

        total_latency_ms = (time.monotonic() - t_start) * 1000

        return PipelineResult(
            task_id=task_id,
            completed=(completed_steps == total_steps),
            steps_completed=completed_steps,
            total_steps=total_steps,
            total_latency_ms=round(total_latency_ms, 2),
            swap_count=swap_count,
            state_preserved=state_preserved,
            models_used=models_used,
            step_results=step_results,
            failure_reason=failure_reason,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "router":    self._router.stats(),
            "preserver": self._preserver.stats(),
            "detector":  self._detector.stats(),
            "registry":  self._registry.stats(),
        }
