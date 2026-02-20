"""
AgentTest — Customer Support Triage Agent
==========================================
A complete, runnable showcase of every AgentTest feature using the
pytest plugin.  No API key required — uses a deterministic mock LLM.

Run:
    pytest examples/customer_support/ -v -s

What you will see
-----------------
Feature 1  Recording           — capture every LLM call into a named baseline
Feature 2  Full replay         — re-run agent, compare to baseline (PASS)
Feature 3  Locked replay       — serve all calls from cache, zero live calls
Feature 4  Regression detection— swap in a broken agent, flag divergences
Feature 5  Root-cause analysis — pinpoint the first step that diverged
Feature 6  Cascade detection   — distinguish root cause from downstream effects
Feature 7  assert_no_regression— one-liner assertion helper
Feature 8  agenttest_auto      — single fixture that auto-selects record vs replay
Feature 9  Markers             — @pytest.mark.baseline wires baseline by name

The agent
---------
A two-node LangGraph graph that classifies a customer message and then
generates a routed reply:

    classify_node  →  respond_node

  V1 (healthy) : classifier returns "billing" / "technical" / "general"
  V2 (broken)  : classifier returns "BILLING" / "TECHNICAL" / "GENERAL"
                  (upstream format change breaks the contract)
"""

from __future__ import annotations

import textwrap
from typing import Annotated, List, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentgit.langgraph_callback import langgraph_callback
from agenttest.pytest_plugin.assertions import (
    assert_no_regression,
    assert_step_count,
)


# ══════════════════════════════════════════════════════════════════════════════
# Mock LLM — deterministic, zero API cost
# ══════════════════════════════════════════════════════════════════════════════

class SupportLLM(BaseChatModel):
    """
    Rule-based mock LLM — behaviour is fully deterministic based on the
    system-prompt and user message content, making every run reproducible.

    version="v1"  Healthy agent — classifies in lowercase
    version="v2"  Broken agent  — classifies in UPPERCASE (regression)
    """

    version: str = "v1"

    @property
    def _llm_type(self) -> str:
        return "support-llm"

    @property
    def _identifying_params(self) -> dict:
        return {"version": self.version}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        sys_text  = " ".join(m.content for m in messages if isinstance(m, SystemMessage)).lower()
        user_text = " ".join(m.content for m in messages if isinstance(m, HumanMessage)).lower()

        # ── classify node ──────────────────────────────────────────────────
        if "classify" in sys_text:
            if any(w in user_text for w in ["charge", "refund", "invoice", "payment", "bill", "subscription"]):
                label = "billing"
            elif any(w in user_text for w in ["error", "crash", "bug", "broken", "not working", "fail"]):
                label = "technical"
            else:
                label = "general"

            # V2 regression: returns uppercase, breaking the downstream contract
            content = label.upper() if self.version == "v2" else label

        # ── respond node ───────────────────────────────────────────────────
        elif "billing specialist" in sys_text:
            content = (
                "I can help with your billing issue. "
                "Could you share your account ID so I can review the charges?"
            )
        elif "technical support" in sys_text:
            content = (
                "Let's get this sorted out. "
                "Can you describe the error and the steps that led to it?"
            )
        elif "general support" in sys_text:
            content = (
                "Thanks for reaching out! "
                "I'm happy to help — could you tell me a bit more about what you need?"
            )
        else:
            # V2 fallout: category key not found → generic fallback
            content = (
                "Thank you for contacting us. "
                "Please hold while I connect you with the right team."
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph agent
# ══════════════════════════════════════════════════════════════════════════════

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str          # set by classify_node, read by respond_node


def _build_agent(llm: SupportLLM, callback, session) -> "CompiledGraph":
    """classify → respond"""
    cfg = {
        "callbacks": [callback],
        "configurable": {
            "user_id":    session.user_id,
            "session_id": session.session_id,
        },
    }

    def classify_node(state: SupportState) -> SupportState:
        msgs = [
            SystemMessage(content=(
                "You are a customer-support classifier. "
                "Classify the message as exactly one of: billing, technical, or general. "
                "Reply with ONLY the category word, nothing else."
            )),
            *[m for m in state["messages"] if isinstance(m, HumanMessage)],
        ]
        response = llm.invoke(msgs, config=cfg)
        return {"messages": [response], "category": response.content.strip().lower()}

    def respond_node(state: SupportState) -> SupportState:
        category = state.get("category", "general")
        system_map = {
            "billing":   "You are a billing specialist. Help resolve payment and invoice issues.",
            "technical": "You are a technical support engineer. Diagnose and fix problems.",
            "general":   "You are a general support agent. Answer questions warmly.",
        }
        msgs = [
            SystemMessage(content=system_map.get(category, system_map["general"])),
            *[m for m in state["messages"] if isinstance(m, HumanMessage)],
        ]
        return {"messages": [llm.invoke(msgs, config=cfg)]}

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_node)
    graph.add_node("respond",  respond_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def _build_intercepted_agent(llm, callback, session, replayer) -> "CompiledGraph":
    """
    Same graph but with a gatekeeper middleware injected in front of each
    LLM call.  Used for LOCKED / SELECTIVE replay so live calls are
    short-circuited and served from cache instead.
    """
    mw = replayer.middleware   # [gatekeeper_fn]   (empty list if mode == "full")
    cfg = {
        "callbacks": [callback],
        "configurable": {
            "user_id":    session.user_id,
            "session_id": session.session_id,
        },
    }

    def _invoke(msgs):
        if mw:
            result = mw[0]({"messages": msgs})
            if result.get("_skip_model"):
                return result["messages"][-1]      # ← served from cache
        return llm.invoke(msgs, config=cfg)         # ← live call (fallback)

    def classify_node(state: SupportState) -> SupportState:
        msgs = [
            SystemMessage(content=(
                "You are a customer-support classifier. "
                "Classify the message as exactly one of: billing, technical, or general. "
                "Reply with ONLY the category word, nothing else."
            )),
            *[m for m in state["messages"] if isinstance(m, HumanMessage)],
        ]
        response = _invoke(msgs)
        return {"messages": [response], "category": response.content.strip().lower()}

    def respond_node(state: SupportState) -> SupportState:
        category = state.get("category", "general")
        system_map = {
            "billing":   "You are a billing specialist. Help resolve payment and invoice issues.",
            "technical": "You are a technical support engineer. Diagnose and fix problems.",
            "general":   "You are a general support agent. Answer questions warmly.",
        }
        msgs = [
            SystemMessage(content=system_map.get(category, system_map["general"])),
            *[m for m in state["messages"] if isinstance(m, HumanMessage)],
        ]
        return {"messages": [_invoke(msgs)]}

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_node)
    graph.add_node("respond",  respond_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Test data
# ══════════════════════════════════════════════════════════════════════════════

TICKETS = [
    HumanMessage(content="I was charged twice for my subscription last month — please refund."),
    HumanMessage(content="The app crashes every time I try to export a report."),
    HumanMessage(content="How do I change my notification preferences?"),
]


def _run(graph, tickets=None) -> list[str]:
    """Run all tickets through the graph and return the final response for each."""
    responses = []
    for msg in (tickets or TICKETS):
        result = graph.invoke({"messages": [msg], "category": ""})
        responses.append(result["messages"][-1].content)
    return responses


def _banner(title: str) -> None:
    print()
    print("━" * 64)
    print(f"  {title}")
    print("━" * 64)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — Recording
#  agenttest_record() wraps your agent run and captures every LLM call.
#  set_as_baseline=True promotes the recording to a named baseline on exit.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
def test_01_recording(agenttest_session, agenttest_record):
    """
    FEATURE 1 — Recording
    ---------------------
    Wrap your agent run with agenttest_record() to snapshot every LLM call.
    Pass set_as_baseline=True to promote the recording to a named baseline
    automatically when the context manager exits cleanly.

    Captured per step:
      • request prompt (messages list)
      • response content
      • model / provider / token usage
      • content fingerprint (SHA-256 of response)
      • duration in ms
    """
    _banner("Feature 1 — Recording  (agenttest_record)")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v1   = SupportLLM(version="v1")

    # ── record ────────────────────────────────────────────────────────────
    # The name "support-v1" is used to retrieve the baseline in later tests.
    with agenttest_record(name="support-v1", set_as_baseline=True) as rec:
        graph     = _build_agent(llm_v1, callback, agenttest_session)
        responses = _run(graph)

    # ── verify capture ────────────────────────────────────────────────────
    details = agenttest_session.get_recording_details(rec.recording_id)

    print(f"\n  Recording ID : {rec.recording_id[:16]}...")
    print(f"  Steps captured: {rec.step_count}  (3 tickets × 2 nodes)")
    print()
    for i, d in enumerate(details):
        response_preview = d.response_data.get("content", "")[:60]
        print(f"  [{i}] fingerprint={d.fingerprint[:8]}  "
              f"response='{response_preview}'")

    assert rec.step_count == 6,           "Expected 6 LLM steps (3 tickets × 2 nodes)"
    assert len(details) == 6,             "TestStore should have 6 LLMCallDetail rows"
    assert all(d.fingerprint for d in details), "Every step should have a fingerprint"

    # Billing ticket → billing response
    assert "billing" in responses[0].lower() or "account" in responses[0].lower()
    # Technical ticket → technical response
    assert "error" in responses[1].lower() or "crash" in responses[1].lower() or "sorted" in responses[1].lower()

    print(f"\n  ✓ Baseline 'support-v1' recorded and promoted.")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — Full replay
#  Re-runs the agent with live LLM calls, then compares output to baseline.
#  Use this in CI to prove the agent's behaviour hasn't changed.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_02_full_replay(agenttest_session, agenttest_replay):
    """
    FEATURE 2 — Full replay  (mode='full')
    ----------------------------------------
    agenttest_replay() loads the named baseline, re-runs the agent with
    live LLM calls (same model, same inputs), and automatically compares
    every step to the baseline.

    rep.passed             → True if all steps match
    rep.comparison_result  → full ComparisonResult dataclass
    rep.print_report()     → pretty-printed summary
    """
    _banner("Feature 2 — Full replay  (agenttest_replay, mode='full')")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v1   = SupportLLM(version="v1")

    with agenttest_replay(
        baseline_name="support-v1",
        replay_name="support-v1-full-replay",
        mode="full",
    ) as rep:
        graph = _build_agent(llm_v1, callback, agenttest_session)
        _run(graph)

    rep.print_report()

    r = rep.comparison_result
    print(f"\n  matched_steps  : {r.matched_steps}")
    print(f"  diverged_steps : {r.mismatched_steps}")
    print(f"  total_steps    : {r.total_steps}")

    assert rep.passed,                   "Healthy V1 agent should pass the replay"
    assert r.matched_steps == 6,         "All 6 steps should match"
    assert r.mismatched_steps == 0,      "No divergences expected"


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — Locked replay  (zero live LLM calls)
#  All responses are served from the baseline cache.  No model is charged.
#  Perfect for fast, cost-free CI regression gates.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_03_locked_replay(agenttest_session, agenttest_replay):
    """
    FEATURE 3 — Locked replay  (mode='locked')
    -------------------------------------------
    In locked mode AgentTest installs a gatekeeper middleware in front of
    every LLM call.  Each call is looked up in the baseline cache;
    if found the cached response is returned immediately and the real
    model is never called.

    rep.middleware   → [gatekeeper_fn] to inject into your agent
    rep.cache_stats  → {"cache_hits": N, "live_calls": 0, ...}

    Wire the middleware by calling mw[0]({"messages": msgs}) before
    llm.invoke().  If the result has _skip_model=True, use the cached
    response directly.
    """
    _banner("Feature 3 — Locked replay  (agenttest_replay, mode='locked')")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v1   = SupportLLM(version="v1")

    with agenttest_replay(
        baseline_name="support-v1",
        replay_name="support-v1-locked-replay",
        mode="locked",
    ) as rep:
        # _build_intercepted_agent wires rep.middleware into each LLM call
        graph = _build_intercepted_agent(llm_v1, callback, agenttest_session, rep)
        _run(graph)

    stats = rep.cache_stats
    print(f"\n  cache_hits  : {stats['cache_hits']}")
    print(f"  live_calls  : {stats['live_calls']}")
    print(f"  hit_rate    : {stats['cache_hit_rate']:.0%}")
    print(f"  comparison  : {'PASS' if rep.passed else 'FAIL'}")

    assert rep.passed,                 "Locked replay of V1 baseline should pass"
    assert stats["live_calls"]  == 0,  "No live LLM calls in locked mode"
    assert stats["cache_hits"]  == 6,  "All 6 steps served from cache"
    assert stats["cache_hit_rate"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 4 — Regression detection
#  Swap in a broken agent.  AgentTest flags every step that diverged.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_04_regression_detection(agenttest_session, agenttest_replay):
    """
    FEATURE 4 — Regression detection
    ---------------------------------
    Developer ships V2: the classifier now returns uppercase category names
    (BILLING, TECHNICAL, GENERAL) instead of lowercase.

    AgentTest compares each V2 step to the V1 baseline and flags every
    step whose response content diverged.  rep.passed → False.

    Diverged steps have:
      step.status              StepStatus.DIVERGE
      step.similarity_score    < threshold (default 0.95)
      step.diff_summary        human-readable description of what changed
    """
    _banner("Feature 4 — Regression detection  (V2 broken agent)")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v2   = SupportLLM(version="v2")   # ← broken: returns uppercase

    print("\n  V1 classify output :  billing / technical / general")
    print("  V2 classify output :  BILLING / TECHNICAL / GENERAL  ← regression")

    with agenttest_replay(
        baseline_name="support-v1",
        replay_name="support-v2-regression",
        mode="full",
    ) as rep:
        graph = _build_agent(llm_v2, callback, agenttest_session)
        _run(graph)

    r = rep.comparison_result

    print(f"\n  overall   : {'PASS' if r.overall_pass else 'FAIL  ← regression detected'}")
    print(f"  matched   : {r.matched_steps}/{r.total_steps}")
    print(f"  diverged  : {r.mismatched_steps}")
    print(f"  cascade   : {r.cascade_steps}")
    print()

    for sc in r.step_comparisons:
        status_label = {
            "match":   "✓ MATCH  ",
            "diverge": "✗ DIVERGE",
            "cascade": "~ CASCADE",
        }.get(sc.status.value, sc.status.value)
        summary = f"  {sc.diff_summary}" if sc.diff_summary else ""
        print(f"  step {sc.step_index}  {status_label}  score={sc.similarity_score:.2f}{summary}")

    assert not rep.passed,                "V2 agent MUST fail the regression check"
    assert r.mismatched_steps > 0,        "There should be diverged steps"
    assert r.total_steps == 6,            "All steps compared"


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 5 — Root-cause analysis
#  AgentTest identifies the FIRST step that diverged (root cause) and
#  labels subsequent divergences as CASCADE — caused by the root, not
#  independent regressions.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_05_root_cause_and_cascade(agenttest_session, agenttest_replay):
    """
    FEATURE 5 — Root-cause analysis + cascade detection
    ----------------------------------------------------
    When multiple steps diverge, AgentTest uses LCS alignment to identify:

      root cause  — the earliest independent divergence
      cascade     — steps that diverged only because their INPUT changed
                    as a downstream effect of the root cause

    Example in this agent:
      Ticket T-001 classify diverges (root cause: "billing" → "BILLING")
      Ticket T-001 respond  diverges (cascade: wrong system-prompt used)

    rep.comparison_result.root_cause_index  → step index of root cause
    rep.root_cause_summary                  → human-readable description
    """
    _banner("Feature 5 — Root-cause analysis + cascade detection")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v2   = SupportLLM(version="v2")

    with agenttest_replay(
        baseline_name="support-v1",
        replay_name="support-v2-rootcause",
        mode="full",
    ) as rep:
        graph = _build_agent(llm_v2, callback, agenttest_session)
        _run(graph)

    r = rep.comparison_result

    print(f"\n  root_cause_index : {r.root_cause_index}")
    print(f"  root_cause_summary: {rep.root_cause_summary}")
    print()

    diverged = [s for s in r.step_comparisons if s.status.value == "diverge"]
    cascades = [s for s in r.step_comparisons if s.status.value == "cascade"]

    print(f"  diverged steps : {[s.step_index for s in diverged]}")
    print(f"  cascade steps  : {[s.step_index for s in cascades]}")

    assert r.root_cause_index is not None,  "A root cause must be identified"
    assert rep.root_cause_summary,          "Root cause summary must not be empty"

    # Root cause is always a classify step (even-indexed: 0, 2, 4)
    assert r.step_comparisons[r.root_cause_index].step_index % 2 == 0, (
        "Root cause should be a classify step (even index)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 6 & 7 — assert_no_regression + assert_step_count helpers
#  Drop-in assertion helpers for clean, readable tests.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_06_assertion_helpers(agenttest_session, agenttest_replay):
    """
    FEATURE 6 & 7 — assert_no_regression and assert_step_count
    ------------------------------------------------------------
    Import once, use everywhere.  These helpers give clean, descriptive
    failure messages instead of bare 'assert rep.passed'.

    assert_no_regression(rep.comparison_result)
        → AssertionError: "Regression detected: Step 0: ..."

    assert_step_count(rep.comparison_result, exact_steps=6)
        → AssertionError: "Expected exactly 6 steps, got 4"
    """
    _banner("Feature 6 & 7 — assert_no_regression + assert_step_count")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v1   = SupportLLM(version="v1")

    with agenttest_replay(
        baseline_name="support-v1",
        replay_name="support-v1-helpers",
        mode="full",
    ) as rep:
        graph = _build_agent(llm_v1, callback, agenttest_session)
        _run(graph)

    # These helpers raise AssertionError with clear messages on failure
    assert_no_regression(rep.comparison_result)
    assert_step_count(rep.comparison_result, exact_steps=6)

    print(f"\n  assert_no_regression(rep.comparison_result)      ✓")
    print(f"  assert_step_count(rep.comparison_result, exact_steps=6)  ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 8 — @pytest.mark.baseline marker
#  Declare which baseline a test targets directly on the test function.
#  agenttest_replay() reads the marker automatically when baseline_name
#  is omitted — no need to repeat the name inside the test body.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")                       # ← declares the target
def test_07_baseline_marker(agenttest_session, agenttest_replay):
    """
    FEATURE 8 — @pytest.mark.baseline
    ----------------------------------
    Annotate your test with @pytest.mark.baseline("name") and omit
    baseline_name from agenttest_replay().  The fixture reads the marker
    and resolves the baseline automatically.

    Keeps test code DRY — the baseline name lives in one place.
    """
    _banner("Feature 8 — @pytest.mark.baseline marker")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm_v1   = SupportLLM(version="v1")

    # Note: baseline_name is NOT passed — the fixture reads @pytest.mark.baseline
    with agenttest_replay(mode="full") as rep:
        graph = _build_agent(llm_v1, callback, agenttest_session)
        _run(graph)

    print(f"\n  Baseline resolved from marker : 'support-v1'")
    print(f"  Result : {'PASS' if rep.passed else 'FAIL'}")

    assert rep.passed


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 9 — agenttest_auto  (zero boilerplate)
#  The same test function records OR replays depending on CLI flags.
#  Add regression safety to existing tests without changing a single line
#  of agent code.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.agenttest
@pytest.mark.baseline("support-v1-auto")
def test_08_agenttest_auto(agenttest_session, agenttest_auto):
    """
    FEATURE 9 — agenttest_auto  (CLI-flag-driven record / replay)
    --------------------------------------------------------------
    agenttest_auto reads CLI flags and auto-selects the right mode:

      pytest --agenttest --agenttest-record   →  records + sets baseline
      pytest --agenttest                      →  replays, fails on regression
      pytest  (no flags)                      →  normal test, no recording

    The test body is IDENTICAL in all three modes.  No Recorder or
    Replayer imports needed.  Just add `agenttest_auto` to the signature.

    Note: in this self-contained demo we run without --agenttest flags so
    agenttest_auto is None and the test acts as a normal agent test.
    """
    _banner("Feature 9 — agenttest_auto  (zero boilerplate)")

    callback = langgraph_callback(agenttest_session.ag.eventbus)
    llm      = SupportLLM(version="v1")
    graph    = _build_agent(llm, callback, agenttest_session)

    # ── Your normal agent test — unchanged whether recording or replaying ──
    result = graph.invoke({
        "messages": [HumanMessage(content="I was charged twice for my subscription.")],
        "category": "",
    })
    final = result["messages"][-1].content

    # Standard assertions — run in every mode
    assert len(final) > 10
    assert "billing" in final.lower() or "account" in final.lower()

    # ── Explain current mode ──────────────────────────────────────────────
    if agenttest_auto is None:
        print("\n  agenttest_auto=None — running as a plain test (no --agenttest flag)")
        print("  Re-run with --agenttest-record to record a baseline.")
        print("  Re-run with --agenttest          to replay and gate on regressions.")
    elif hasattr(agenttest_auto, "comparison_result"):
        print("\n  agenttest_auto — REPLAY mode")
        print(f"  Result: {'PASS' if agenttest_auto.passed else 'FAIL'}")
    else:
        print("\n  agenttest_auto — RECORD mode")
        print(f"  Recording ID: {agenttest_auto.recording_id}")
