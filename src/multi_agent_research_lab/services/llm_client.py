"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import langsmith_enabled


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with an offline fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        If OpenAI is configured and installed, use it. Otherwise fall back to a deterministic
        local response so the lab remains runnable without network access.
        """

        if self.settings.openai_api_key:
            try:
                from openai import OpenAI
            except ImportError:
                pass
            else:
                client = OpenAI(api_key=self.settings.openai_api_key)
                if langsmith_enabled():
                    try:
                        from langsmith.wrappers import wrap_openai
                    except ImportError:
                        pass
                    else:
                        client = wrap_openai(client)
                try:
                    response = client.responses.create(
                        model=self.settings.openai_model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                except Exception:
                    pass
                else:
                    content = getattr(response, "output_text", "").strip()
                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "input_tokens", None) if usage else None
                    output_tokens = getattr(usage, "output_tokens", None) if usage else None
                    return LLMResponse(
                        content=content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=None,
                    )

        content = self._fallback_complete(system_prompt=system_prompt, user_prompt=user_prompt)
        input_tokens = max(1, len((system_prompt + user_prompt).split()))
        output_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round((input_tokens + output_tokens) * 0.000002, 6),
        )

    def _fallback_complete(self, system_prompt: str, user_prompt: str) -> str:
        system_hint = system_prompt.lower()
        prompt = user_prompt.strip()
        if "researcher" in system_hint:
            return "Research summary:\n" + prompt
        if "analyst" in system_hint:
            return "Analysis summary:\n" + prompt
        if "writer" in system_hint:
            return prompt
        return (
            "Offline fallback response generated without an external model.\n\n"
            f"Task: {prompt}"
        )
