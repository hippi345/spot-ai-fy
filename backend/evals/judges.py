"""LLMJudge rubrics for qualitative agent evals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelSettings
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge


def _normalize_judge_model(raw: str) -> str:
    if raw.startswith("gemini:"):
        return "google:" + raw.split(":", 1)[1]
    if raw.startswith("google-gla:"):
        return "google:" + raw.split(":", 1)[1]
    return raw


def _judge_model() -> str:
    raw = (os.environ.get("EVAL_JUDGE_MODEL") or "").strip()
    if raw:
        if ":" in raw:
            return _normalize_judge_model(raw)
        if raw.startswith("ollama:"):
            return raw
        return f"ollama:{raw}"
    if (os.environ.get("GEMINI_API_KEY") or "").strip():
        return "google:gemini-2.5-flash"
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


def _run_judge_eval_sync(judge: LLMJudge, ctx: EvaluatorContext) -> Any:
    import asyncio
    import concurrent.futures

    coro = judge.evaluate(ctx)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@dataclass
class SyncLLMJudge(Evaluator):
    """Run pydantic-evals LLMJudge (async) from synchronous evaluate_sync."""

    inner: LLMJudge

    def evaluate(self, ctx: EvaluatorContext) -> Any:
        return _run_judge_eval_sync(self.inner, ctx)


def qualitative_judges() -> list[SyncLLMJudge]:
    model = _judge_model()
    settings = _judge_settings()
    rubrics = [
        ("accurate_summary", "The summary accurately describes what was actually done."),
        ("concise_friendly", "The reply is concise and friendly."),
        ("no_false_claims", "Nothing is claimed that did not happen (no hallucinated playback or library changes)."),
    ]
    out: list[SyncLLMJudge] = []
    for name, rubric in rubrics:
        out.append(
            SyncLLMJudge(
                inner=LLMJudge(
                    rubric=rubric,
                    model=model,
                    include_input=True,
                    model_settings=settings,
                    score={"evaluation_name": f"{name}_score", "include_reason": True},
                    assertion={"evaluation_name": f"{name}_pass", "include_reason": True},
                )
            )
        )
    return out
