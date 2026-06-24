"""Tracing hooks with optional LangSmith integration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


@lru_cache(maxsize=1)
def _get_langsmith_client() -> Any | None:
    settings = get_settings()
    if not _langsmith_enabled(settings):
        return None

    try:
        from langsmith import Client
    except ImportError:
        return None

    client_kwargs: dict[str, Any] = {"api_key": settings.langsmith_api_key}
    if settings.langsmith_endpoint:
        client_kwargs["api_url"] = settings.langsmith_endpoint
    return Client(**client_kwargs)


def _langsmith_enabled(settings: Settings | None = None) -> bool:
    resolved_settings = settings or get_settings()
    return bool(resolved_settings.langsmith_tracing and resolved_settings.langsmith_api_key)


def langsmith_enabled() -> bool:
    """Return whether LangSmith tracing is configured and enabled."""

    return _get_langsmith_client() is not None


@contextmanager
def workflow_trace(
    state: ResearchState,
    name: str,
    *,
    run_type: RunType = "chain",
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Wrap a workflow run in a root trace when LangSmith is available."""

    started = perf_counter()
    trace_info: dict[str, Any] = {
        "name": name,
        "attributes": metadata or {},
        "duration_seconds": None,
        "trace_url": None,
    }
    client = _get_langsmith_client()
    if client is None:
        try:
            yield trace_info
        finally:
            trace_info["duration_seconds"] = perf_counter() - started
        return

    try:
        from langsmith import trace
        from langsmith.run_helpers import tracing_context
    except ImportError:
        try:
            yield trace_info
        finally:
            trace_info["duration_seconds"] = perf_counter() - started
        return

    settings = get_settings()
    inputs = {"query": state.request.query, "audience": state.request.audience}
    context = tracing_context(
        enabled=True,
        client=client,
        project_name=settings.langsmith_project,
    )

    try:
        with context, trace(
            name=name,
            run_type=run_type,
            client=client,
            project_name=settings.langsmith_project,
            inputs=inputs,
            metadata=metadata or {},
        ) as root:
            yield trace_info
            root.add_outputs(
                {
                    "iteration": state.iteration,
                    "route_history": list(state.route_history),
                    "final_answer": state.final_answer,
                }
            )
            state.trace_id = str(root.id)
            state.trace_url = root.get_url()
            trace_info["trace_url"] = state.trace_url
    finally:
        trace_info["duration_seconds"] = perf_counter() - started
        client.flush()


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    run_type: RunType = "tool",
) -> Iterator[dict[str, Any]]:
    """Create a local span and optionally mirror it to LangSmith."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    client = _get_langsmith_client()
    traced_context: Any

    if client is None:
        traced_context = nullcontext()
    else:
        try:
            from langsmith import trace
        except ImportError:
            traced_context = nullcontext()
        else:
            traced_context = trace(
                name=name,
                run_type=run_type,
                client=client,
                project_name=get_settings().langsmith_project,
                inputs=attributes or {},
                metadata=attributes or {},
            )

    try:
        with traced_context:
            yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
