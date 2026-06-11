"""
state_preserver.py
------------------
Snapshots complete agent state before a provider swap and restores
it after — ensuring zero context loss during mid-task hot-swaps.

Pure Python. Zero dependencies.
"""

from __future__ import annotations
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    snapshot_id: str
    agent_id: str
    task_id: str
    source_model_id: str
    timestamp: float
    messages: List[Dict[str, str]]
    system_prompt: str
    current_step: int
    total_steps: int
    step_label: str
    partial_output: Dict[str, Any]
    task_context: Dict[str, Any]
    expected_output_schema: Optional[Dict[str, Any]] = None

    def completion_ratio(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.current_step / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "source_model_id": self.source_model_id,
            "timestamp": self.timestamp,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "step_label": self.step_label,
            "partial_output": self.partial_output,
            "task_context": self.task_context,
            "expected_output_schema": self.expected_output_schema,
        }


def _make_snapshot_id(agent_id: str, task_id: str, timestamp: float) -> str:
    raw = f"{agent_id}:{task_id}:{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class StatePreserver:
    def __init__(self):
        self._store: Dict[str, AgentState] = {}
        self._snapshot_count = 0
        self._restore_count = 0

    def snapshot(
        self,
        agent_id: str,
        task_id: str,
        source_model_id: str,
        messages: List[Dict[str, str]],
        system_prompt: str,
        current_step: int,
        total_steps: int,
        step_label: str,
        partial_output: Dict[str, Any],
        task_context: Dict[str, Any],
        expected_output_schema: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        ts = time.monotonic()
        snapshot_id = _make_snapshot_id(agent_id, task_id, ts)
        state = AgentState(
            snapshot_id=snapshot_id,
            agent_id=agent_id,
            task_id=task_id,
            source_model_id=source_model_id,
            timestamp=ts,
            messages=list(messages),
            system_prompt=system_prompt,
            current_step=current_step,
            total_steps=total_steps,
            step_label=step_label,
            partial_output=dict(partial_output),
            task_context=dict(task_context),
            expected_output_schema=expected_output_schema,
        )
        self._store[snapshot_id] = state
        self._snapshot_count += 1
        return state

    def restore(self, snapshot_id: str) -> Optional[AgentState]:
        state = self._store.get(snapshot_id)
        if state:
            self._restore_count += 1
        return state

    def latest_for_task(self, task_id: str) -> Optional[AgentState]:
        candidates = [s for s in self._store.values() if s.task_id == task_id]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.timestamp)

    def build_resume_message(self, state: AgentState) -> str:
        lines = [
            f"[RESUME] Task '{state.task_id}' interrupted at step "
            f"{state.current_step}/{state.total_steps} ({state.step_label}).",
            f"Previous model: {state.source_model_id}.",
            f"Progress: {state.completion_ratio():.0%} complete.",
        ]
        if state.partial_output:
            lines.append(
                f"Partial output so far: "
                f"{json.dumps(state.partial_output, indent=2)}"
            )
        lines.append("Continue from where the previous model stopped.")
        if state.expected_output_schema:
            lines.append(
                f"Required output schema: "
                f"{json.dumps(state.expected_output_schema, indent=2)}"
            )
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_snapshots": self._snapshot_count,
            "total_restores": self._restore_count,
            "stored_snapshots": len(self._store),
        }
