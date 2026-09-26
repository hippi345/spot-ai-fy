"""Types for deterministic chat shortcut outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeterministicChatResult:
    reply: str
    tool_steps: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [name for name, _, _ in self.tool_steps]

    def tool_raw_results(self) -> list[str]:
        return [raw for _, _, raw in self.tool_steps]
