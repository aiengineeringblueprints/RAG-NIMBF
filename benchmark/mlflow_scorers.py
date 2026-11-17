"""Post-run MLflow scorers judged by the critic LLM (issue #44).

Scores a finished config run's per-sample traces (issue #42, with ground
truth attached as expectations by #43) with MLflow built-in judges and
RAGAS-via-MLflow scorers, without re-running the RAG pipeline. The judge is
the configured critic LLM, reached through its OpenAI-compatible route so
local Ollama/vLLM/NIM models work.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow

from benchmark.providers import parse_model_id

if TYPE_CHECKING:
    from config import BenchmarkConfig

logger = logging.getLogger(__name__)

# Metric prefix that keeps judge scores apart from the direct RAGAS
# (``ragas_*``) and custom (``custom_*``) metrics on the same run.
METRIC_PREFIX = "mlflow_judge_"

# Ollama serves its OpenAI-compatible API under this path.
_OLLAMA_OPENAI_PATH = "/v1"
# MLflow's OpenAI provider refuses to start without a key; custom
# (local) endpoints ignore it.
_PLACEHOLDER_API_KEY = "not-needed"


@dataclass(frozen=True)
class JudgeEndpoint:
    """Where and how MLflow's judges reach the critic LLM.

    ``None`` base URL / key leave the process environment as it is: an
    ``openai:`` critic without a compat URL is the hosted OpenAI API and uses
    the user's own ``OPENAI_API_KEY``.
    """

    model_uri: str
    base_url: str | None
    api_key: str | None = field(repr=False)
    inference_params: dict[str, Any] = field(default_factory=dict)


def resolve_judge_endpoint(config: BenchmarkConfig) -> JudgeEndpoint:
    """Map the critic LLM settings onto an MLflow ``openai:/`` judge."""
    provider, model_name = parse_model_id(config.eval_critic_llm)
    url = (config.eval_critic_base_url(provider) or "").rstrip("/")
    if provider != "openai" and not url.endswith(_OLLAMA_OPENAI_PATH):
        url += _OLLAMA_OPENAI_PATH
    # Custom endpoint: always route there, with a placeholder key if keyless.
    # No URL (hosted OpenAI): keep the environment's base URL and key.
    base_url = url or None
    api_key = config.eval_critic_api_key(provider)
    if base_url and not api_key:
        api_key = _PLACEHOLDER_API_KEY

    inference_params: dict[str, Any] = {"max_tokens": config.eval_critic_max_tokens}
    if config.eval_critic_temperature is not None:
        inference_params = {"temperature": config.eval_critic_temperature, **inference_params}

    return JudgeEndpoint(
        model_uri=f"openai:/{model_name}",
        base_url=base_url,
        api_key=api_key,
        inference_params=inference_params,
    )


ScorerFactory = Callable[[JudgeEndpoint], Any]


def _builtin(class_name: str) -> ScorerFactory:
    def build(endpoint: JudgeEndpoint) -> Any:
        from mlflow.genai import scorers

        return getattr(scorers, class_name)(
            model=endpoint.model_uri, inference_params=dict(endpoint.inference_params)
        )

    return build


def _ragas(class_name: str) -> ScorerFactory:
    def build(endpoint: JudgeEndpoint) -> Any:
        from mlflow.genai.scorers import ragas

        return getattr(ragas, class_name)(model=endpoint.model_uri)

    return build


# Keys mirror config.MLFLOW_SCORER_NAMES.
SCORER_FACTORIES: dict[str, ScorerFactory] = {
    "retrieval_groundedness": _builtin("RetrievalGroundedness"),
    "retrieval_relevance": _builtin("RetrievalRelevance"),
    "retrieval_sufficiency": _builtin("RetrievalSufficiency"),
    "correctness": _builtin("Correctness"),
    "relevance_to_query": _builtin("RelevanceToQuery"),
    "ragas_faithfulness": _ragas("Faithfulness"),
    "ragas_context_precision": _ragas("ContextPrecision"),
    "ragas_context_recall": _ragas("ContextRecall"),
    "ragas_factual_correctness": _ragas("FactualCorrectness"),
}


@contextmanager
def judge_environment(endpoint: JudgeEndpoint) -> Iterator[None]:
    """Point MLflow's ``openai`` provider at the critic, then restore env.

    MLflow resolves the provider's base URL and key from the environment on
    every judge call, so the override must span construction and scoring.
    """
    overrides = {
        name: value
        for name, value in (
            ("OPENAI_API_BASE", endpoint.base_url),
            ("OPENAI_API_KEY", endpoint.api_key),
        )
        if value is not None
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_scorers(
    names: Sequence[str],
    endpoint: JudgeEndpoint,
    *,
    factories: Mapping[str, ScorerFactory] | None = None,
) -> dict[str, Any]:
    """Instantiate the selected scorers, keyed by their config name."""
    factories = SCORER_FACTORIES if factories is None else factories
    with judge_environment(endpoint):
        return {name: factories[name](endpoint) for name in names}


def score_run_traces(
    run_id: str,
    names: Sequence[str],
    endpoint: JudgeEndpoint,
    *,
    factories: Mapping[str, ScorerFactory] | None = None,
) -> dict[str, float]:
    """Judge a finished config run's traces and log the scores on that run.

    Reads the traces linked to ``run_id`` (no pipeline re-run), lets MLflow
    attach per-trace assessments, and logs one ``mlflow_judge_<name>_mean``
    metric per scorer. Failures are logged and tagged, never raised.
    """
    if not names:
        return {}
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_id=run_id, nested=nested):
        mlflow.set_tag("mlflow_scorer_model", endpoint.model_uri)
        try:
            status, metrics = _evaluate(run_id, names, endpoint, factories)
        except Exception as exc:
            mlflow.set_tag("mlflow_scorers", "failed")
            logger.warning("MLflow scorers failed for run %s (non-fatal): %s", run_id, exc)
            return {}
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tag("mlflow_scorers", status)
        return metrics


def _evaluate(
    run_id: str,
    names: Sequence[str],
    endpoint: JudgeEndpoint,
    factories: Mapping[str, ScorerFactory] | None,
) -> tuple[str, dict[str, float]]:
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    traces = mlflow.search_traces(locations=[experiment_id], run_id=run_id)
    if len(traces) == 0:
        return "skipped_no_traces", {}

    scorers = build_scorers(names, endpoint, factories=factories)
    with judge_environment(endpoint):
        result = mlflow.genai.evaluate(data=traces, scorers=list(scorers.values()))

    metrics: dict[str, float] = {}
    for name, scorer in scorers.items():
        value = result.metrics.get(f"{scorer.name}/mean")
        if value is None:
            logger.warning(
                "MLflow scorer %r (%s) produced no '%s/mean' metric; skipped",
                name, scorer.name, scorer.name,
            )
            continue
        metrics[f"{METRIC_PREFIX}{name}_mean"] = float(value)
    return ("logged" if metrics else "no_scores"), metrics


def score_config_run(run_id: str, config: BenchmarkConfig) -> dict[str, float]:
    """Run the config's selected scorers on its logged run (no-op when none).

    Never raises: the config already completed, so a judge-side problem must
    not mark it failed.
    """
    if not config.mlflow_scorers:
        return {}
    try:
        endpoint = resolve_judge_endpoint(config)
    except Exception as exc:
        logger.warning("MLflow scorers skipped for run %s (non-fatal): %s", run_id, exc)
        return {}
    return score_run_traces(run_id, config.mlflow_scorers, endpoint)
