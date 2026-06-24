"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    state, metrics = run_benchmark("baseline", query, _run_baseline)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(metrics.model_dump_json(indent=2))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))
    if result.trace_url:
        console.print(Panel.fit(result.trace_url, title="LangSmith Trace"))


def _run_baseline(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    sources = SearchClient().search(query=query, max_results=request.max_sources)
    source_block = "\n".join(
        f"[{index}] {source.title}: {source.snippet}"
        for index, source in enumerate(sources, start=1)
    )
    response = LLMClient().complete(
        system_prompt="You are a helpful single-agent research assistant.",
        user_prompt=(
            f"Question: {query}\n"
            f"Audience: {request.audience}\n"
            f"Use these local sources when relevant:\n{source_block}\n\n"
            "Write a concise answer and end with a Sources section."
        ),
    )
    state.sources = sources
    state.final_answer = (
        response.content
        if "Sources:" in response.content
        else f"{response.content}\n\nSources:\n{source_block}"
    )
    return state


if __name__ == "__main__":
    app()
