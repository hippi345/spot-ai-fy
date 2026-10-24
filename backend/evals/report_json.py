"""Serialize pydantic-evals EvaluationReport (dataclass) to JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_evals.reporting import EvaluationReport


def _json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "value") and hasattr(obj, "reason"):
        return {"value": obj.value, "reason": obj.reason}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def evaluation_report_to_json(report: EvaluationReport) -> str:
    payload = asdict(report)
    return json.dumps(payload, indent=2, default=_json_default)
