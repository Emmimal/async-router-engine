"""
model_registry.py
-----------------
Tracks available LLM providers, their capabilities, rate limits,
and per-model adapter templates for Strategy B state normalization.

Pure Python. Zero dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class ModelTier(Enum):
    PRIMARY   = "primary"
    SECONDARY = "secondary"
    TERTIARY  = "tertiary"


class OutputFormat(Enum):
    STRUCTURED_JSON = "structured_json"
    SIMPLE_JSON     = "simple_json"
    TEXT            = "text"


@dataclass
class ModelProfile:
    model_id: str
    tier: ModelTier
    output_format: OutputFormat
    max_tokens: int
    requests_per_minute: int
    supports_system_prompt: bool = True
    supports_json_mode: bool = True
    system_prompt_template: str = "{system_prompt}"
    response_schema: Optional[Dict[str, Any]] = None
    priority: int = 0


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "model_a": ModelProfile(
        model_id="model_a",
        tier=ModelTier.PRIMARY,
        output_format=OutputFormat.STRUCTURED_JSON,
        max_tokens=8192,
        requests_per_minute=60,
        supports_system_prompt=True,
        supports_json_mode=True,
        system_prompt_template=(
            "SYSTEM: {system_prompt}\n"
            "Respond ONLY with valid JSON matching the required schema."
        ),
        response_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"},
                "metadata": {"type": "object"},
            },
            "required": ["result", "confidence"],
        },
        priority=1,
    ),
    "model_b": ModelProfile(
        model_id="model_b",
        tier=ModelTier.SECONDARY,
        output_format=OutputFormat.SIMPLE_JSON,
        max_tokens=4096,
        requests_per_minute=100,
        supports_system_prompt=True,
        supports_json_mode=True,
        system_prompt_template=(
            "Instructions: {system_prompt}\n"
            "Return a JSON object with 'result' and 'confidence' fields."
        ),
        response_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["result", "confidence"],
        },
        priority=2,
    ),
    "model_c": ModelProfile(
        model_id="model_c",
        tier=ModelTier.TERTIARY,
        output_format=OutputFormat.TEXT,
        max_tokens=2048,
        requests_per_minute=200,
        supports_system_prompt=False,
        supports_json_mode=False,
        system_prompt_template="{system_prompt}\nRespond in plain text.",
        response_schema=None,
        priority=3,
    ),
}


class ModelRegistry:
    def __init__(self, profiles: Optional[Dict[str, ModelProfile]] = None):
        self._profiles: Dict[str, ModelProfile] = profiles or dict(MODEL_PROFILES)
        self._unavailable: set[str] = set()

    def get(self, model_id: str) -> Optional[ModelProfile]:
        return self._profiles.get(model_id)

    def all_models(self) -> List[ModelProfile]:
        return sorted(self._profiles.values(), key=lambda m: m.priority)

    def available_models(self) -> List[ModelProfile]:
        return [m for m in self.all_models()
                if m.model_id not in self._unavailable]

    def mark_unavailable(self, model_id: str) -> None:
        self._unavailable.add(model_id)

    def mark_available(self, model_id: str) -> None:
        self._unavailable.discard(model_id)

    def next_available(self, exclude: Optional[List[str]] = None) -> Optional[ModelProfile]:
        excluded = set(exclude or [])
        for model in self.available_models():
            if model.model_id not in excluded:
                return model
        return None

    def adapt_payload(
        self,
        target_model_id: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        profile = self._profiles.get(target_model_id)
        if not profile:
            raise ValueError(f"Unknown model: {target_model_id}")

        adapted_system = profile.system_prompt_template.format(
            system_prompt=system_prompt
        )

        adapted_messages: List[Dict[str, str]] = []

        if profile.supports_system_prompt:
            adapted_messages.append({"role": "system", "content": adapted_system})
        else:
            if messages:
                first = messages[0].copy()
                first["content"] = f"{adapted_system}\n\n{first['content']}"
                adapted_messages.append(first)
                messages = messages[1:]

        adapted_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": target_model_id,
            "messages": adapted_messages,
            "max_tokens": profile.max_tokens,
        }

        if profile.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}

        if profile.response_schema:
            payload["schema"] = profile.response_schema

        return payload

    def stats(self) -> Dict[str, Any]:
        return {
            "total_models": len(self._profiles),
            "available": len(self.available_models()),
            "unavailable": list(self._unavailable),
            "priority_order": [m.model_id for m in self.available_models()],
        }
