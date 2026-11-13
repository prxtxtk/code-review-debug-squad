"""Unit tests for intent detection, RunnableBranch routing, and the RAG gate."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from demo_llm import DemoChatModel
from rag_gate import AgenticRAGGate
from router import (
    IntentRouter,
    build_intent_branch,
    fallback_intent,
)
from vector_store import RetrievalHit


@pytest.mark.parametrize(
    ("user_request", "expected"),
    [
        ("This crashes with an empty list — find the bug and fix it.", "debugging"),
        ("Debug this code. Explain the root cause and provide a corrected version.", "debugging"),
        ("Please clean up spacing and make this follow PEP 8.", "style_review"),
        ("Explain what this helper does and when it returns None.", "code_explanation"),
        ("Write pytest tests for the boundary cases and run them.", "testing"),
        ("Review everything: bugs, style, fix it, and run tests.", "full_review"),
    ],
)
def test_intent_router_classifies_all_five_routes(user_request: str, expected: str) -> None:
    router = IntentRouter(DemoChatModel())
    result = router.detect(user_request, "def f():\n    return 1\n")
    assert result.intent == expected
    assert result.source == "llm"
    assert 0.0 <= result.confidence <= 1.0
    assert result.prompt_trace["messages"][0]["role"] == "System"
    assert result.prompt_trace["messages"][-1]["role"] == "User"


def test_router_prompt_is_few_shot_and_structured() -> None:
    router = IntentRouter(DemoChatModel())
    result = router.detect("Debug this", "def f(): pass")
    label = result.prompt_trace["label"]
    techniques = result.prompt_trace["techniques"]
    assert label == "Intent Router"
    assert "Few-Shot Prompting" in techniques
    assert any("JSON" in message["content"] for message in result.prompt_trace["messages"])


def test_runnable_branch_executes_each_route() -> None:
    routes = {
        intent: RunnableLambda(lambda state, current=intent: {**state, "executed": current})
        for intent in (
            "debugging",
            "style_review",
            "code_explanation",
            "testing",
            "full_review",
        )
    }
    branch = build_intent_branch(routes)
    for intent in routes:
        output = branch.invoke({"intent": intent, "trace": {}})
        assert output["executed"] == intent


def test_fallback_intent_uses_hint_when_text_is_ambiguous() -> None:
    intent, confidence, rationale = fallback_intent("please look at this", hint="testing")
    assert intent == "testing"
    assert confidence > 0
    assert "hint" in rationale.lower()


class _FakeStore:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int = 4) -> list[RetrievalHit]:
        self.queries.append(query)
        return [
            RetrievalHit(
                content="Use parameterized SQL queries.",
                source="python_bug_patterns.md",
                title="Bug patterns",
                section="SQL injection",
                distance=0.2,
                similarity=0.8333,
                metadata={"source": "python_bug_patterns.md"},
            )
        ][:k]


def test_rag_gate_skips_trivial_explanation_without_touching_store() -> None:
    store = _FakeStore()
    gate = AgenticRAGGate(DemoChatModel(), store, top_k=3)  # type: ignore[arg-type]
    result = gate.decide(
        "Explain this simple add function.",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )
    assert result.should_retrieve is False
    assert result.query == ""
    assert result.hits == []
    assert store.queries == []
    assert "trivial" in result.reason.lower()
    assert result.to_trace()["retrieval_executed"] is False


def test_rag_gate_retrieves_for_debugging_and_exposes_scores() -> None:
    store = _FakeStore()
    gate = AgenticRAGGate(DemoChatModel(), store, top_k=3)  # type: ignore[arg-type]
    result = gate.decide(
        "Why does this average function fail on an empty list?",
        "def avg(xs):\n    return sum(xs) / len(xs)\n",
    )
    assert result.should_retrieve is True
    assert store.queries
    assert len(result.hits) == 1
    trace = result.to_trace()
    assert trace["hits"][0]["source"] == "python_bug_patterns.md"
    assert 0.0 < trace["hits"][0]["similarity"] <= 1.0
    assert "RETRIEVED 1" in result.retrieved_context
