"""Workflow orchestration for the multi-agent lab."""

from __future__ import annotations

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import workflow_trace


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def build(self) -> dict[str, BaseAgent]:
        """Create a simple executable graph description."""

        return {
            "supervisor": self.supervisor,
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        graph = self.build()
        with workflow_trace(
            state,
            "multi_agent_workflow",
            metadata={"max_iterations": self.settings.max_iterations},
        ):
            while True:
                state = self.supervisor.run(state)
                next_route = state.route_history[-1]
                if next_route == "done":
                    break
                agent = graph[next_route]
                state = agent.run(state)

            if not state.final_answer:
                raise ValidationError("Workflow completed without a final answer.")

            state = self.critic.run(state)
        return state
