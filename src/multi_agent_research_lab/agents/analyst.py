"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        with trace_span("analyst", {"query": state.request.query}) as span:
            sources_summary = "\n".join(
                f"- {source.title}: {source.snippet}" for source in state.sources
            )
            prompt = (
                f"Question: {state.request.query}\n"
                f"Research notes:\n{state.research_notes or ''}\n\n"
                f"Sources:\n{sources_summary}\n\n"
                "Produce:\n"
                "1. Key findings\n"
                "2. Conflicting viewpoints or uncertainty\n"
                "3. Practical implications for the requested audience"
            )
            response = self.llm_client.complete(
                system_prompt="You are an analyst agent that structures evidence into conclusions.",
                user_prompt=prompt,
            )
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"]["has_research_notes"] = bool(state.research_notes)
            state.add_trace_event("agent", {"name": self.name, "span": span})
        return state
