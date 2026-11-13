"""Agentic RAG: a two-step LCEL decision gate followed by conditional retrieval.

Step 1 — an LCEL chain evaluates the request and code and returns JSON:
    ``{"should_retrieve": bool, "query": str, "reason": str}``

Step 2 — only when the decision is true is ChromaDB queried.  The retrieved
chunks, metadata, and similarity scores are recorded verbatim in the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator

from prompts import RAG_GATE_PROMPT, prompt_record
from vector_store import (
    RetrievalHit,
    VectorStoreManager,
    format_hits_for_prompt,
)


class RAGDecision(BaseModel):
    """Pydantic structured output for the retrieval gate."""

    should_retrieve: bool = Field(description="Whether the knowledge base should be queried")
    query: str = Field(default="", description="Focused semantic search query")
    reason: str = Field(description="One-sentence justification")

    @field_validator("query")
    @classmethod
    def normalise_query(cls, value: str) -> str:
        return (value or "").strip()


@dataclass(slots=True)
class RAGDecisionResult:
    should_retrieve: bool
    query: str
    reason: str
    source: str = "llm"
    error: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    retrieved_context: str = ""
    prompt_trace: dict[str, Any] = field(default_factory=dict)
    top_k: int = 4

    @property
    def retrieval_executed(self) -> bool:
        return self.should_retrieve

    def to_trace(self) -> dict[str, Any]:
        return {
            "should_retrieve": self.should_retrieve,
            "query": self.query,
            "reason": self.reason,
            "decision_source": self.source,
            "error": self.error,
            "retrieval_executed": self.should_retrieve,
            "top_k": self.top_k,
            "hits": [hit.trace_dict() for hit in self.hits],
            "retrieved_context": self.retrieved_context,
        }


class AgenticRAGGate:
    """LCEL-backed autonomous retrieval decision + conditional vector search."""

    def __init__(self, llm: Any, vector_store: VectorStoreManager, top_k: int = 4) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.parser = JsonOutputParser(pydantic_object=RAGDecision)
        # LCEL: prompt | llm | structured parser
        self.chain = RAG_GATE_PROMPT | llm | self.parser

    def decide(self, request: str, code: str, top_k: int | None = None) -> RAGDecisionResult:
        variables = {"request": request, "code": code}
        recorded_prompt = prompt_record("rag_gate", RAG_GATE_PROMPT, variables)
        error = ""
        try:
            raw = self.chain.invoke(variables)
            if isinstance(raw, RAGDecision):
                decision = raw
            elif isinstance(raw, dict):
                decision = RAGDecision.model_validate(raw)
            else:
                raise ValueError(f"Unexpected parser result: {type(raw).__name__}")
            source = "llm"
        except Exception as exc:
            decision = self._fallback_decision(request, code)
            source = "fallback"
            error = f"{type(exc).__name__}: {exc}"

        effective_top_k = max(1, min(int(top_k or self.top_k), 10))
        effective_query = decision.query.strip()
        if decision.should_retrieve and not effective_query:
            effective_query = self._fallback_query(request)
        if not decision.should_retrieve:
            effective_query = decision.query.strip()
        result = RAGDecisionResult(
            should_retrieve=decision.should_retrieve,
            query=effective_query,
            reason=decision.reason,
            source=source,
            error=error,
            prompt_trace=recorded_prompt,
            top_k=effective_top_k,
        )
        if result.should_retrieve:
            try:
                result.hits = self.vector_store.search(result.query, k=effective_top_k)
                result.retrieved_context = format_hits_for_prompt(result.hits)
            except Exception as exc:
                result.error = (result.error + " | " if result.error else "") + f"Retrieval failed: {exc}"
                result.retrieved_context = "(Retrieval was requested but ChromaDB returned an error.)"
        else:
            result.retrieved_context = "(The RAG gate decided retrieval was unnecessary for this request.)"
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_decision(request: str, code: str) -> RAGDecision:
        lower = (request or "").lower()
        nonempty = [line for line in (code or "").splitlines() if line.strip()]
        wants_explanation = any(word in lower for word in ("explain", "what does", "walk through"))
        trivial = len(nonempty) <= 8 and not re.search(
            r"(range\([^)]*\+\s*1|=\s*\[\]|=\s*\{\}|SELECT|sqlite3|except\s*:|/|eval\()",
            code or "",
        )
        if wants_explanation and trivial:
            return RAGDecision(
                should_retrieve=False,
                query="",
                reason="Fallback policy: short explanation tasks on trivial snippets can be answered without retrieval.",
            )
        return RAGDecision(
            should_retrieve=True,
            query=AgenticRAGGate._fallback_query(request),
            reason="Fallback policy: review, debugging, style, and testing tasks benefit from curated guidelines.",
        )

    @staticmethod
    def _fallback_query(request: str) -> str:
        lower = (request or "").lower()
        if any(word in lower for word in ("test", "pytest", "unit test")):
            return "pytest edge cases boundary tests assertions"
        if any(word in lower for word in ("style", "lint", "pep", "format")):
            return "PEP 8 naming docstrings formatting clean code"
        if any(word in lower for word in ("debug", "bug", "fix", "error")):
            return "Python debugging mutable defaults off-by-one zero division resource leaks"
        return "Python code review bug patterns style guidelines testing"


__all__ = ["AgenticRAGGate", "RAGDecision", "RAGDecisionResult"]
