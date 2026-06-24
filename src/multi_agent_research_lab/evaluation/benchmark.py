"""Benchmark helpers for single-agent vs multi-agent."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive lightweight quality metrics from the final state."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    estimated_cost = sum(
        float(result.metadata.get("cost_usd", 0.0) or 0.0)
        for result in state.agent_results
    )
    citation_coverage = (
        1.0
        if state.sources and state.final_answer and "Sources:" in state.final_answer
        else 0.0
    )
    quality_score = _estimate_quality_score(state=state, citation_coverage=citation_coverage)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors else 0.0,
        notes=f"iterations={state.iteration}; sources={len(state.sources)}",
    )
    return state, metrics


def _estimate_quality_score(state: ResearchState, citation_coverage: float) -> float:
    score = 0.0
    if state.research_notes:
        score += 3.0
    if state.analysis_notes:
        score += 3.0
    if state.final_answer:
        score += 2.0
    score += 2.0 * citation_coverage
    return min(10.0, score)
