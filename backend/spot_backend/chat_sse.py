"""Server-Sent Events formatting for Spot-AI-fy."""

from __future__ import annotations

import json
from typing import Any


def sse_data(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
