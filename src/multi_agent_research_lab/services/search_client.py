"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import re
from pathlib import Path

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with a local corpus fallback."""

    def __init__(self, docs_root: Path | None = None) -> None:
        self.docs_root = docs_root or Path.cwd()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search local project documents relevant to a query."""

        keywords = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9-]+", query)
            if len(token) > 2
        }
        candidates: list[tuple[int, SourceDocument]] = []
        for path in self._iter_corpus_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            score = sum(text.lower().count(keyword) for keyword in keywords)
            if score == 0:
                continue
            snippet = self._extract_snippet(text, keywords)
            candidates.append(
                (
                    score,
                    SourceDocument(
                        title=path.name,
                        url=str(path),
                        snippet=snippet,
                        metadata={"path": str(path), "score": score},
                    ),
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            return [document for _, document in candidates[:max_results]]

        return [
            SourceDocument(
                title="Local fallback note",
                url=None,
                snippet=f"No strong local source match found for query: {query}",
                metadata={"path": "fallback", "score": 0},
            )
        ]

    def _iter_corpus_files(self) -> list[Path]:
        patterns = ("README.md", "docs/*.md", "src/**/*.py", "configs/*.yaml")
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(self.docs_root.glob(pattern))
        return [path for path in paths if path.is_file()]

    def _extract_snippet(self, text: str, keywords: set[str]) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in keywords):
                return line[:280]
        return lines[0][:280] if lines else ""
