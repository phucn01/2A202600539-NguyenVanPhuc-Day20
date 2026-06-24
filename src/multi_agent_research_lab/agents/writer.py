"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        with trace_span("writer", {"query": state.request.query}) as span:
            citations = "\n".join(
                f"[{index}] {source.title} - {source.snippet}"
                for index, source in enumerate(state.sources, start=1)
            )
            prompt = (
                f"Audience: {state.request.audience}\n"
                f"Question: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or ''}\n\n"
                f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
                f"Available citations:\n{citations}\n\n"
                "Write a concise answer with a short synthesis paragraph and a 'Sources' section."
            )
            response = self.llm_client.complete(
                system_prompt=(
                    "You are a writer agent that answers clearly and cites the provided "
                    "sources."
                ),
                user_prompt=prompt,
            )
            sources_block = "\n".join(
                f"[{index}] {source.title}: {source.snippet}"
                for index, source in enumerate(state.sources, start=1)
            )
            if "Sources:" not in response.content:
                final_answer = f"{response.content.strip()}\n\nSources:\n{sources_block}"
            else:
                final_answer = response.content.strip()

            state.final_answer = final_answer
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=final_answer,
                    metadata={
                        "citations": len(state.sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"]["citation_count"] = len(state.sources)
            state.add_trace_event("agent", {"name": self.name, "span": span})
        return state
