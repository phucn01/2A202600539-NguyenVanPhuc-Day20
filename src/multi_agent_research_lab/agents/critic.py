"""Optional critic agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        findings: list[str] = []
        if not state.sources:
            findings.append("No sources were collected.")
        if not state.final_answer:
            findings.append("Final answer is missing.")
        elif "Sources:" not in state.final_answer:
            findings.append("Final answer has no explicit sources section.")

        content = "No major issues detected." if not findings else " ".join(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={"issue_count": len(findings)},
            )
        )
        state.add_trace_event("agent", {"name": self.name, "findings": findings})
        return state
