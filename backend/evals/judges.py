"""LLMJudge rubrics for qualitative agent evals."""

from __future__ import annotations

import os

from pydantic_ai import ModelSettings
from pydantic_evals.evaluators import LLMJudge


def _judge_model() -> str:
    raw = (os.environ.get("EVAL_JUDGE_MODEL") or "").strip()
    if raw:
        if ":" in raw:
            return raw
        if raw.startswith("ollama:"):
            return raw
        return f"ollama:{raw}"
    host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    # OpenAI-compatible Ollama endpoint for pydantic-ai
    model_name = (os.environ.get("EVAL_JUDGE_OLLAMA_MODEL") or "qwen2.5:7b").strip()
    os.environ.setdefault("OPENAI_BASE_URL", f"{host}/v1")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    return f"openai:{model_name}"


def _judge_settings() -> ModelSettings:
    return ModelSettings(temperature=0.0)


def mood_intent_judge() -> LLMJudge:
    model = _judge_model()
    settings = _judge_settings()
    return LLMJudge(
        rubric="The reply matches the requested mood or intent.",
        model=model,
        include_input=True,
        model_settings=settings,
        score={"evaluation_name": "mood_intent_score", "include_reason": True},
        assertion={"evaluation_name": "mood_intent_pass", "include_reason": True},
    )


def qualitative_judges() -> list[LLMJudge]:
    model = _judge_model()
    settings = _judge_settings()
    rubrics = [
        ("accurate_summary", "The summary accurately describes what was actually done."),
        ("concise_friendly", "The reply is concise and friendly."),
        ("no_false_claims", "Nothing is claimed that did not happen (no hallucinated playback or library changes)."),
    ]
    out: list[LLMJudge] = []
    for name, rubric in rubrics:
        out.append(
            LLMJudge(
                rubric=rubric,
                model=model,
                include_input=True,
                model_settings=settings,
                score={"evaluation_name": f"{name}_score", "include_reason": True},
                assertion={"evaluation_name": f"{name}_pass", "include_reason": True},
            )
        )
    return out
