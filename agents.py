"""LangChain LCEL agents and the ``RunnableBranch`` review orchestrator.

Every agent is a composable LCEL chain, for example::

    BUG_FINDER_PROMPT | llm | StrOutputParser()

The full-review route runs Bug-Finder and Style/Lint in a real
``RunnableParallel``.  Fixer performs a second reflection call, and the Test
Writer's suite is executed by the isolated sandbox.
"""

from __future__ import annotations

import ast
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field, field_validator

from config import AppConfig, llm_identity
from prompts import (
    BUG_FINDER_PROMPT,
    EXPLAINER_PROMPT,
    FIXER_PROMPT,
    FIXER_REFLECTION_PROMPT,
    PROMPT_CATALOG,
    STYLE_AGENT_PROMPT,
    TEST_WRITER_PROMPT,
    prompt_record,
)
from rag_gate import AgenticRAGGate, RAGDecisionResult
from router import (
    IntentResult,
    IntentRouter,
    build_intent_branch,
    make_route_recorder,
)
from sandbox import SandboxResult, environment_description, run_tests
from tools import AnalysisResult, analysis_tool_trace, analyze_code
from vector_store import VectorStoreManager


class AgentError(RuntimeError):
    """Raised when an agent cannot produce a usable result."""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_fenced_code(text: str, *, last: bool = False) -> str:
    matches = FENCED_CODE_RE.findall(text or "")
    if not matches:
        return ""
    return (matches[-1] if last else matches[0]).strip("\n")


def extract_code_after_marker(text: str, marker: str) -> str:
    if marker not in (text or ""):
        return ""
    return extract_fenced_code(text.split(marker, 1)[1])


def strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python|py)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code)
    return code.strip("\n") + ("\n" if code.strip() else "")


def valid_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno}, column {exc.offset})"


@dataclass(slots=True)
class AgentOutput:
    key: str
    label: str
    text: str
    prompt_trace: dict[str, Any]


@dataclass(slots=True)
class TestPlan:
    test_code: str
    rationale: str = ""


class TestPlanModel(BaseModel):
    test_code: str = Field(description="Complete runnable pytest source code")
    rationale: str = Field(default="", description="Coverage summary")

    @field_validator("test_code")
    @classmethod
    def clean_code(cls, value: str) -> str:
        code = strip_code_fences(value)
        if not code.strip():
            raise ValueError("test_code must not be empty")
        return code


@dataclass(slots=True)
class FixResult:
    initial_text: str
    reflection_text: str
    proposed_code: str
    final_code: str
    prompt_traces: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reflection_only: bool = False

    @property
    def reflection_report(self) -> str:
        if not self.reflection_text:
            return ""
        marker = "===FINAL CODE==="
        return self.reflection_text.split(marker, 1)[0].strip()


# ---------------------------------------------------------------------------
# Individual LCEL agents
# ---------------------------------------------------------------------------

class BugFinderAgent:
    """AST-grounded, CoT-driven defect finder."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.chain = BUG_FINDER_PROMPT | llm | StrOutputParser()

    def run(
        self,
        *,
        request: str,
        code: str,
        ast_report: str,
        retrieved_context: str,
    ) -> AgentOutput:
        variables = {
            "request": request,
            "code": code,
            "ast_report": ast_report,
            "retrieved_context": retrieved_context,
        }
        recorded = prompt_record("bug_finder", BUG_FINDER_PROMPT, variables)
        try:
            report = self.chain.invoke(variables)
        except Exception as exc:
            raise AgentError(f"Bug-Finder Agent failed: {type(exc).__name__}: {exc}") from exc
        return AgentOutput("bug_finder", "Bug-Finder Agent", report, recorded)


class StyleAgent:
    """Few-shot, lint-grounded PEP 8 reviewer."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.chain = STYLE_AGENT_PROMPT | llm | StrOutputParser()

    def run(
        self,
        *,
        request: str,
        code: str,
        lint_report: str,
        retrieved_context: str,
    ) -> AgentOutput:
        variables = {
            "request": request,
            "code": code,
            "lint_report": lint_report,
            "retrieved_context": retrieved_context,
        }
        recorded = prompt_record("style_agent", STYLE_AGENT_PROMPT, variables)
        try:
            report = self.chain.invoke(variables)
        except Exception as exc:
            raise AgentError(f"Style/Lint Agent failed: {type(exc).__name__}: {exc}") from exc
        return AgentOutput("style_agent", "Style/Lint Agent", report, recorded)


class FixerAgent:
    """Produce a patch, then critique it in a mandatory reflection call."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.fix_chain = FIXER_PROMPT | llm | StrOutputParser()
        self.reflection_chain = FIXER_REFLECTION_PROMPT | llm | StrOutputParser()

    def run(
        self,
        *,
        request: str,
        code: str,
        bug_report: str,
        style_report: str,
        retrieved_context: str,
    ) -> FixResult:
        variables = {
            "request": request,
            "code": code,
            "bug_report": bug_report,
            "style_report": style_report,
            "retrieved_context": retrieved_context,
        }
        recorded_fix = prompt_record("fixer", FIXER_PROMPT, variables)
        try:
            initial_text = self.fix_chain.invoke(variables)
        except Exception as exc:
            raise AgentError(f"Fixer Agent failed: {type(exc).__name__}: {exc}") from exc
        proposed = extract_code_after_marker(initial_text, "===FIXED CODE===")
        if not proposed:
            proposed = extract_fenced_code(initial_text, last=True)
        if not proposed:
            raise AgentError("Fixer Agent returned no fenced Python code block.")
        proposed = strip_code_fences(proposed)
        valid, error = valid_python(proposed)
        warnings: list[str] = []
        if not valid:
            warnings.append(f"The proposed patch was not valid Python ({error}); the original code is used as the base.")
            proposed = code
        reflected = self.reflect(
            request=request,
            code=code,
            bug_report=bug_report,
            style_report=style_report,
            proposed_code=proposed,
            retrieved_context=retrieved_context,
            test_failure="",
            initial_text=initial_text,
            initial_prompt_trace=recorded_fix,
        )
        reflected.warnings = warnings + reflected.warnings
        return reflected

    def reflect(
        self,
        *,
        request: str,
        code: str,
        bug_report: str,
        style_report: str,
        proposed_code: str,
        retrieved_context: str,
        test_failure: str = "",
        initial_text: str = "",
        initial_prompt_trace: dict[str, Any] | None = None,
    ) -> FixResult:
        variables = {
            "request": request,
            "bug_report": bug_report,
            "style_report": style_report,
            "proposed_code": proposed_code,
            "test_failure": test_failure,
        }
        recorded = prompt_record("fixer_reflection", FIXER_REFLECTION_PROMPT, variables)
        try:
            reflection_text = self.reflection_chain.invoke(variables)
        except Exception as exc:
            raise AgentError(
                f"Fixer reflection pass failed: {type(exc).__name__}: {exc}"
            ) from exc
        final_code = extract_code_after_marker(reflection_text, "===FINAL CODE===")
        if not final_code:
            final_code = extract_fenced_code(reflection_text, last=True)
        warnings: list[str] = []
        if not final_code:
            warnings.append("Reflection returned no fenced code; the proposed patch was retained.")
            final_code = proposed_code
        final_code = strip_code_fences(final_code)
        valid, error = valid_python(final_code)
        if not valid:
            warnings.append(
                f"Reflection produced invalid Python ({error}); falling back to the prior proposed patch."
            )
            final_code = proposed_code
            valid, error = valid_python(final_code)
            if not valid:
                warnings.append("Prior patch was also invalid; the original source is retained.")
                final_code = code
        traces = [trace for trace in (initial_prompt_trace, recorded) if trace]
        return FixResult(
            initial_text=initial_text,
            reflection_text=reflection_text,
            proposed_code=proposed_code,
            final_code=strip_code_fences(final_code),
            prompt_traces=traces,
            warnings=warnings,
            reflection_only=not bool(initial_text),
        )


class TestWriterAgent:
    """Generate a pytest plan as JSON through a structured LCEL chain."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=TestPlanModel)
        self.chain = TEST_WRITER_PROMPT | llm | self.parser

    def run(
        self,
        *,
        request: str,
        fixed_code: str,
        bug_report: str,
        retrieved_context: str,
    ) -> tuple[TestPlan, dict[str, Any]]:
        variables = {
            "request": request,
            "fixed_code": fixed_code,
            "bug_report": bug_report,
            "retrieved_context": retrieved_context,
        }
        recorded = prompt_record("test_writer", TEST_WRITER_PROMPT, variables)
        try:
            raw = self.chain.invoke(variables)
            if isinstance(raw, TestPlanModel):
                plan = TestPlan(test_code=raw.test_code, rationale=raw.rationale)
            elif isinstance(raw, dict):
                parsed = TestPlanModel.model_validate(raw)
                plan = TestPlan(test_code=parsed.test_code, rationale=parsed.rationale)
            else:
                raise ValueError(f"Unexpected test-writer output type: {type(raw).__name__}")
        except Exception as exc:
            raise AgentError(
                f"Test-Writer Agent failed to produce structured tests: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return plan, recorded


class ExplainerAgent:
    """Explain code without running tools or modifying it."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.chain = EXPLAINER_PROMPT | llm | StrOutputParser()

    def run(self, *, request: str, code: str, retrieved_context: str) -> AgentOutput:
        variables = {
            "request": request,
            "code": code,
            "retrieved_context": retrieved_context,
        }
        recorded = prompt_record("explainer", EXPLAINER_PROMPT, variables)
        try:
            text = self.chain.invoke(variables)
        except Exception as exc:
            raise AgentError(f"Explainer Agent failed: {type(exc).__name__}: {exc}") from exc
        return AgentOutput("explainer", "Explainer Agent", text, recorded)


# ---------------------------------------------------------------------------
# Consolidated result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReviewResult:
    run_id: str
    started_at: str
    request: str
    code: str
    intent: IntentResult
    rag: RAGDecisionResult
    trace: dict[str, Any]
    duration_seconds: float = 0.0
    analysis: AnalysisResult | None = None
    bug_report: str = ""
    style_report: str = ""
    explanation: str = ""
    fixed_code: str = ""
    initial_fixed_code: str = ""
    fixer_initial_output: str = ""
    reflection_report: str = ""
    test_plan: TestPlan | None = None
    test_prompt_trace: dict[str, Any] = field(default_factory=dict)
    sandbox_attempts: list[SandboxResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    route_state: dict[str, Any] = field(default_factory=dict)

    @property
    def final_sandbox(self) -> SandboxResult | None:
        return self.sandbox_attempts[-1] if self.sandbox_attempts else None

    @property
    def success(self) -> bool:
        return not self.errors

    @property
    def has_fixed_code(self) -> bool:
        return bool(self.fixed_code.strip())

    @property
    def route_label(self) -> str:
        return self.intent.route_label


def _test_failure_feedback(result: SandboxResult) -> str:
    return (
        f"Status: {result.status_label}\n"
        f"Exit code: {result.exit_code}\n"
        f"STDOUT:\n{result.stdout[-3500:]}\n"
        f"STDERR:\n{result.stderr[-2500:]}"
    )


# ---------------------------------------------------------------------------
# Orchestrator — LCEL RunnableBranch + RunnableParallel
# ---------------------------------------------------------------------------

class CodeReviewSquad:
    """Compose the intent branch and execute the selected review crew."""

    def __init__(
        self,
        llm: Any,
        vector_store: VectorStoreManager,
        config: AppConfig,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.config = config
        self.intent_router = IntentRouter(llm)
        self.rag_gate = AgenticRAGGate(llm, vector_store, top_k=config.top_k)
        self.bug_finder = BugFinderAgent(llm)
        self.style_agent = StyleAgent(llm)
        self.fixer = FixerAgent(llm)
        self.test_writer = TestWriterAgent(llm)
        self.explainer = ExplainerAgent(llm)

        # Explicit LangChain RunnableParallel for the two independent reviewers.
        self.parallel_reviewers = RunnableParallel(
            bug_report=RunnableLambda(self._bug_stage),
            style_report=RunnableLambda(self._style_stage),
        )

        route_functions = {
            "debugging": self._route_debugging,
            "style_review": self._route_style_review,
            "code_explanation": self._route_code_explanation,
            "testing": self._route_testing,
            "full_review": self._route_full_review,
        }
        routes = {
            intent: make_route_recorder(intent) | RunnableLambda(function)
            for intent, function in route_functions.items()
        }
        # The actual LCEL route selection required by the assignment.
        self.route_branch = build_intent_branch(routes)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def review(self, request: str, code: str, intent_hint: str = "") -> ReviewResult:
        started_perf = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = uuid.uuid4().hex[:10]
        trace: dict[str, Any] = {
            "run_id": run_id,
            "started_at": started_at,
            "llm": llm_identity(self.config),
            "model_config": self.config.public_dict(),
            "prompts": [],
            "tool_calls": [],
            "route_path": "RunnableBranch → (not executed)",
            "rag": {},
            "intent": {},
            "errors": [],
            "warnings": [],
            "prompt_catalog": PROMPT_CATALOG,
        }

        intent_result = self.intent_router.detect(request, code, hint=intent_hint)
        trace["intent"] = intent_result.to_trace()
        trace["prompts"].append(intent_result.prompt_trace)

        rag_result = self.rag_gate.decide(request, code)
        trace["rag"] = rag_result.to_trace()
        trace["prompts"].append(rag_result.prompt_trace)

        try:
            kb_status = self.vector_store.status()
        except Exception as exc:
            kb_status = {"ready": False, "error": str(exc)}
        trace["knowledge_base"] = kb_status

        state: dict[str, Any] = {
            "run_id": run_id,
            "request": request,
            "code": code,
            "intent": intent_result.intent,
            "intent_result": intent_result,
            "rag_result": rag_result,
            "retrieved_context": rag_result.retrieved_context,
            "trace": trace,
        }
        errors: list[str] = []
        try:
            final_state = self.route_branch.invoke(state)
            state.update(final_state)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            trace["errors"].append(errors[-1])

        intent_result.branch_executed = str(trace.get("route_path", ""))
        trace["intent"] = intent_result.to_trace()
        trace["llm_call_count"] = max(0, len(trace["prompts"]) - 2)
        trace["agent_call_count"] = trace["llm_call_count"]
        trace["sandbox_environment"] = environment_description() if state.get("sandbox_attempts") else {}
        trace["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
        trace["status"] = "error" if errors else "completed"
        trace["warnings"] = list(state.get("warnings", []))

        result = ReviewResult(
            run_id=run_id,
            started_at=started_at,
            request=request,
            code=code,
            intent=intent_result,
            rag=rag_result,
            trace=trace,
            duration_seconds=time.perf_counter() - started_perf,
            analysis=state.get("analysis"),
            bug_report=state.get("bug_report", ""),
            style_report=state.get("style_report", ""),
            explanation=state.get("explanation", ""),
            fixed_code=state.get("fixed_code", ""),
            initial_fixed_code=state.get("initial_fixed_code", ""),
            fixer_initial_output=state.get("fixer_initial_output", ""),
            reflection_report=state.get("reflection_report", ""),
            test_plan=state.get("test_plan"),
            test_prompt_trace=state.get("test_prompt_trace", {}),
            sandbox_attempts=state.get("sandbox_attempts", []),
            errors=errors,
            warnings=state.get("warnings", []),
            route_state=state,
        )
        return result

    # ------------------------------------------------------------------
    # Shared stages
    # ------------------------------------------------------------------
    @staticmethod
    def _append_prompt(state: dict[str, Any], prompt_trace: dict[str, Any]) -> None:
        if prompt_trace:
            state["trace"]["prompts"].append(prompt_trace)

    def _ensure_analysis(self, state: dict[str, Any]) -> AnalysisResult:
        if "analysis" in state:
            return state["analysis"]
        analysis = analyze_code(state["code"])
        state["analysis"] = analysis
        state["ast_report"] = analysis.report_markdown
        state["lint_report"] = analysis.report_markdown
        state["trace"]["tool_calls"].append(analysis_tool_trace(analysis))
        return analysis

    def _bug_stage(self, state: dict[str, Any]) -> AgentOutput:
        self._ensure_analysis(state)
        return self.bug_finder.run(
            request=state["request"],
            code=state["code"],
            ast_report=state.get("ast_report", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )

    def _style_stage(self, state: dict[str, Any]) -> AgentOutput:
        self._ensure_analysis(state)
        return self.style_agent.run(
            request=state["request"],
            code=state["code"],
            lint_report=state.get("lint_report", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )

    def _run_fix(
        self,
        state: dict[str, Any],
        *,
        bug_report: str,
        style_report: str,
        test_failure: str = "",
        proposed_code: str | None = None,
    ) -> FixResult:
        if proposed_code is None:
            fix = self.fixer.run(
                request=state["request"],
                code=state["code"],
                bug_report=bug_report,
                style_report=style_report,
                retrieved_context=state.get("retrieved_context", ""),
            )
        else:
            fix = self.fixer.reflect(
                request=state["request"],
                code=state["code"],
                bug_report=bug_report,
                style_report=style_report,
                proposed_code=proposed_code,
                retrieved_context=state.get("retrieved_context", ""),
                test_failure=test_failure,
            )
        for prompt_trace in fix.prompt_traces:
            self._append_prompt(state, prompt_trace)
        if fix.warnings:
            state.setdefault("warnings", []).extend(fix.warnings)
            state["trace"]["warnings"].extend(fix.warnings)
        state["fixed_code"] = fix.final_code
        state["reflection_report"] = fix.reflection_report
        if not fix.reflection_only:
            state["initial_fixed_code"] = fix.proposed_code
            state["fixer_initial_output"] = fix.initial_text
        return fix

    def _generate_and_run_tests(
        self,
        state: dict[str, Any],
        *,
        target_code: str,
        bug_report: str,
    ) -> tuple[TestPlan, SandboxResult]:
        plan, prompt_trace = self.test_writer.run(
            request=state["request"],
            fixed_code=target_code,
            bug_report=bug_report,
            retrieved_context=state.get("retrieved_context", ""),
        )
        state["test_plan"] = plan
        state["test_prompt_trace"] = prompt_trace
        self._append_prompt(state, prompt_trace)
        sandbox_result = run_tests(
            target_code,
            plan.test_code,
            timeout=self.config.sandbox_timeout,
            framework="pytest",
        )
        state.setdefault("sandbox_attempts", []).append(sandbox_result)
        record = sandbox_result.trace_record()
        record["name"] = f"Isolated Sandbox — pytest (attempt {len(state['sandbox_attempts'])})"
        state["trace"]["tool_calls"].append(record)
        return plan, sandbox_result

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def _route_debugging(self, state: dict[str, Any]) -> dict[str, Any]:
        self._ensure_analysis(state)
        bug = self.bug_finder.run(
            request=state["request"],
            code=state["code"],
            ast_report=state.get("ast_report", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )
        state["bug_report"] = bug.text
        self._append_prompt(state, bug.prompt_trace)
        fix = self._run_fix(
            state,
            bug_report=bug.text,
            style_report="(Style/Lint pass was not requested in the debugging route.)",
        )
        state["final_fix"] = fix
        return state

    def _route_style_review(self, state: dict[str, Any]) -> dict[str, Any]:
        self._ensure_analysis(state)
        style = self.style_agent.run(
            request=state["request"],
            code=state["code"],
            lint_report=state.get("lint_report", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )
        state["style_report"] = style.text
        self._append_prompt(state, style.prompt_trace)
        fix = self._run_fix(
            state,
            bug_report="(Bug-Finder pass was not requested in the style-review route.)",
            style_report=style.text,
        )
        state["final_fix"] = fix
        return state

    def _route_code_explanation(self, state: dict[str, Any]) -> dict[str, Any]:
        rag = state["rag_result"]
        if not rag.should_retrieve:
            state["trace"]["skipped_tools_reason"] = (
                "The Agentic RAG gate returned should_retrieve=false for this trivial "
                "explanation, and the explanation route intentionally skips AST/lint "
                "and sandbox tools."
            )
        explanation = self.explainer.run(
            request=state["request"],
            code=state["code"],
            retrieved_context=state.get("retrieved_context", ""),
        )
        state["explanation"] = explanation.text
        self._append_prompt(state, explanation.prompt_trace)
        return state

    def _route_testing(self, state: dict[str, Any]) -> dict[str, Any]:
        plan, sandbox_result = self._generate_and_run_tests(
            state,
            target_code=state["code"],
            bug_report="(Testing route: the submitted code is the target; no Bug-Finder pass was requested.)",
        )
        state["test_plan"] = plan
        # Optional single feedback-driven reflection is only part of full_review.
        state["testing_sandbox"] = sandbox_result
        return state

    def _route_full_review(self, state: dict[str, Any]) -> dict[str, Any]:
        self._ensure_analysis(state)
        parallel_results = self.parallel_reviewers.invoke(state)
        bug: AgentOutput = parallel_results["bug_report"]
        style: AgentOutput = parallel_results["style_report"]
        state["bug_report"] = bug.text
        state["style_report"] = style.text
        self._append_prompt(state, bug.prompt_trace)
        self._append_prompt(state, style.prompt_trace)
        state["parallel_reviewers_executed"] = True

        fix = self._run_fix(state, bug_report=bug.text, style_report=style.text)
        state["final_fix"] = fix

        plan, first_sandbox = self._generate_and_run_tests(
            state,
            target_code=fix.final_code,
            bug_report=bug.text,
        )
        state["test_plan"] = plan

        if (
            not first_sandbox.tests_passed
            and not first_sandbox.blocked
            and not first_sandbox.timed_out
            and self.config.max_test_retries > 0
        ):
            state["trace"]["sandbox_retry"] = {
                "attempted": True,
                "reason": "The first isolated pytest run failed; one Fixer reflection retry was permitted.",
            }
            retry_fix = self._run_fix(
                state,
                bug_report=bug.text,
                style_report=style.text,
                test_failure=_test_failure_feedback(first_sandbox),
                proposed_code=fix.final_code,
            )
            state["final_fix"] = retry_fix
            state["fixed_code"] = retry_fix.final_code
            if retry_fix.reflection_report:
                state["reflection_report"] = retry_fix.reflection_report
            _, second_sandbox = self._generate_and_run_tests(
                state,
                target_code=retry_fix.final_code,
                bug_report=bug.text,
            )
            state["retry_sandbox"] = second_sandbox
        else:
            state["trace"]["sandbox_retry"] = {"attempted": False}
        return state


__all__ = [
    "AgentError",
    "AgentOutput",
    "BugFinderAgent",
    "CodeReviewSquad",
    "ExplainerAgent",
    "FixerAgent",
    "ReviewResult",
    "StyleAgent",
    "TestPlan",
    "TestWriterAgent",
    "extract_code_after_marker",
    "extract_fenced_code",
    "strip_code_fences",
    "valid_python",
]
