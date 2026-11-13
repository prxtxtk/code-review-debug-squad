"""Intent detection and LCEL ``RunnableBranch`` routing.

The Intent Router is itself an LCEL chain:
``INTENT_ROUTER_PROMPT | llm | JsonOutputParser``.

The routed pipeline is assembled with ``RunnableBranch`` so the exact branch
selected for every request is visible and auditable in the Trace Inspector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field, field_validator

from prompts import INTENT_ROUTER_PROMPT, prompt_record

IntentLabel = Literal[
    "debugging",
    "style_review",
    "code_explanation",
    "testing",
    "full_review",
]

VALID_INTENTS: tuple[str, ...] = (
    "debugging",
    "style_review",
    "code_explanation",
    "testing",
    "full_review",
)

INTENT_DESCRIPTIONS: dict[str, str] = {
    "debugging": "Agentic RAG → Bug-Finder Agent → Fixer Agent (with Reflection)",
    "style_review": "Agentic RAG → Style/Lint Agent → Fixer Agent (with Reflection)",
    "code_explanation": "Explainer chain (tools and RAG skipped when trivially answerable)",
    "testing": "Agentic RAG → Test-Writer Agent → Isolated Sandbox",
    "full_review": (
        "Agentic RAG → Bug-Finder + Style/Lint in RunnableParallel → Fixer with "
        "Reflection → Test-Writer → Isolated Sandbox (one retry on failure)"
    ),
}

ROUTE_LABELS: dict[str, str] = {
    "debugging": "Debugging Route",
    "style_review": "Style Review Route",
    "code_explanation": "Explanation Route",
    "testing": "Testing Route",
    "full_review": "Full Squad Route",
}


class IntentDecision(BaseModel):
    """Pydantic structured output for the Intent Router."""

    intent: IntentLabel = Field(description="One of the five supported routes")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="")

    @field_validator("rationale")
    @classmethod
    def clean_rationale(cls, value: str) -> str:
        return (value or "").strip()


@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float
    rationale: str
    source: str = "llm"
    error: str = ""
    hint: str = ""
    prompt_trace: dict[str, Any] = field(default_factory=dict)
    branch_executed: str = ""

    @property
    def route_label(self) -> str:
        return ROUTE_LABELS.get(self.intent, self.intent)

    @property
    def route_description(self) -> str:
        return INTENT_DESCRIPTIONS.get(self.intent, "")

    def to_trace(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "label": self.route_label,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "decision_source": self.source,
            "error": self.error,
            "ui_hint": self.hint,
            "route_description": self.route_description,
            "branch_executed": self.branch_executed,
        }


class IntentRouter:
    """Detect the user's intent with an LCEL structured chain and fallback."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=IntentDecision)
        self.chain = INTENT_ROUTER_PROMPT | llm | self.parser

    def detect(self, request: str, code: str, hint: str = "") -> IntentResult:
        variables = {"request": request, "code": code}
        recorded_prompt = prompt_record("intent_router", INTENT_ROUTER_PROMPT, variables)
        try:
            raw = self.chain.invoke(variables)
            if isinstance(raw, IntentDecision):
                decision = raw
            elif isinstance(raw, dict):
                decision = IntentDecision.model_validate(raw)
            else:
                raise ValueError(f"Unexpected router parser result: {type(raw).__name__}")
            return IntentResult(
                intent=decision.intent,
                confidence=decision.confidence,
                rationale=decision.rationale,
                source="llm",
                hint=hint,
                prompt_trace=recorded_prompt,
            )
        except Exception as exc:
            intent, confidence, rationale = fallback_intent(request, hint)
            return IntentResult(
                intent=intent,
                confidence=confidence,
                rationale=rationale,
                source="fallback",
                error=f"{type(exc).__name__}: {exc}",
                hint=hint,
                prompt_trace=recorded_prompt,
            )


def fallback_intent(request: str, hint: str = "") -> tuple[str, float, str]:
    text = (request or "").lower()
    if any(term in text for term in ("full squad", "full review", "complete review", "review everything", "all agents")):
        return "full_review", 0.85, "Keyword fallback matched a complete-review request."
    if "debug" in text:
        return "debugging", 0.88, "Keyword fallback matched an explicit debugging request."
    if any(term in text for term in ("test", "pytest", "unit test")):
        return "testing", 0.82, "Keyword fallback matched a test-generation request."
    if any(term in text for term in ("explain", "what does", "walk through", "how does")):
        return "code_explanation", 0.82, "Keyword fallback matched an explanation request."
    if any(term in text for term in ("style", "lint", "pep 8", "pep8", "format", "readability")):
        return "style_review", 0.82, "Keyword fallback matched a style request."
    if any(term in text for term in ("bug", "fix", "error", "crash", "fails", "root cause")):
        return "debugging", 0.84, "Keyword fallback matched a debugging request."
    if hint in VALID_INTENTS:
        return hint, 0.60, "The LLM router was unavailable; using the UI intent hint."
    return "full_review", 0.55, "No decisive keyword; defaulting to the complete squad route."


def _intent_matches(state: Any, intent: str) -> bool:
    if not isinstance(state, dict):
        return False
    return state.get("intent") == intent


def build_intent_branch(routes: dict[str, Runnable]) -> RunnableBranch:
    """Assemble the five-way route using LangChain's ``RunnableBranch``.

    ``routes`` must contain a callable/runnable per supported intent.  Each
    branch receives the complete pipeline state dictionary.
    """
    missing = [intent for intent in VALID_INTENTS if intent not in routes]
    if missing:
        raise ValueError(f"Missing routes for: {', '.join(missing)}")
    branches: list[tuple[Callable[[Any], bool], Runnable]] = []
    for intent in VALID_INTENTS:
        condition = RunnableLambda(lambda state, current=intent: _intent_matches(state, current))
        branches.append((condition, routes[intent]))
    # The last route is also the defensive default.
    return RunnableBranch(*branches, routes["full_review"])


def make_route_recorder(intent: str) -> Runnable:
    """Wrap a route so it records exactly which branch was taken.

    Routed functions call this wrapper at entry; the wrapper mutates the trace
    dictionary supplied in state before invoking the implementing coroutine-free
    function.
    """

    def _record(state: dict[str, Any]) -> dict[str, Any]:
        trace = state.setdefault("trace", {})
        trace["route_path"] = f"RunnableBranch → {intent}"
        trace["route_label"] = ROUTE_LABELS[intent]
        trace["route_description"] = INTENT_DESCRIPTIONS[intent]
        return state

    return RunnableLambda(_record)


__all__ = [
    "IntentDecision",
    "IntentResult",
    "IntentRouter",
    "VALID_INTENTS",
    "INTENT_DESCRIPTIONS",
    "ROUTE_LABELS",
    "build_intent_branch",
    "fallback_intent",
    "make_route_recorder",
]
