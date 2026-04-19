"""Spot-AI-fy: optional markdown context appended to the agent system prompt."""

from __future__ import annotations

from pathlib import Path

from spot_backend.config import Settings

# Package parent = `backend/`.
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def load_optional_agent_context_markdown(settings: Settings) -> str:
    """
    If a context file exists, return text to append after the built-in system prompt.

    Search order (first hit wins):
    1. ``AGENT_CONTEXT_FILE`` from settings (.env), if set
    2. ``backend/AGENT_CONTEXT.md`` (good for repo-local playbooks; gitignore your real file if private)
    3. ``{DATA_DIR}/Spot-AI-fy-agent-context.md`` (e.g. next to tokens)
    """
    candidates: list[Path] = []
    raw = (settings.agent_context_file or "").strip()
    if raw:
        candidates.append(Path(raw))
    candidates.append(_BACKEND_DIR / "AGENT_CONTEXT.md")
    candidates.append(settings.data_dir / "Spot-AI-fy-agent-context.md")

    for path in candidates:
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return "\n\n## Extra context (markdown file)\n\n" + text
        except OSError:
            continue
    return ""
