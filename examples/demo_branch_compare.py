"""
AgentGit Demo: Branch, Explore, Compare
========================================

A LangGraph agent explores two different approaches to the same coding problem.
AgentGit checkpoints the state, branches, and compares results side-by-side.

This demonstrates the killer use case: deterministic comparison of agent
strategies from the exact same starting point.

Usage:
    export GOOGLE_API_KEY="your-key"
    python examples/demo_branch_compare.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import AgentGit
from tools.branch_tools import BranchTools
from tools.version_tools import VersionTools
from event import EventType
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END


USER = "developer"
SESSION = "code-task"
DIVIDER = "=" * 64


def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY environment variable")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Initialize AgentGit ─────────────────────────────────────
        ag = AgentGit(tmpdir)
        bt = BranchTools(ag)
        vt = VersionTools(ag)
        callback = ag.get_callback()

        bt.create_branch(USER, SESSION, "main", intent="Main execution")

        # Track tokens across branches
        token_tracker = {"a": 0, "b": 0}

        def track_tokens(event):
            usage = getattr(event, "usage", None)
            if usage and isinstance(usage, dict):
                token_tracker["current"] = token_tracker.get("current", 0) + (usage.get("total_tokens") or 0)

        ag.on(EventType.LLM_CALL_END, track_tokens)

        # ── Setup LangGraph ─────────────────────────────────────────
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        def agent_node(state):
            return {"messages": llm.invoke(state["messages"])}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        app = graph.compile()

        config = {
            "callbacks": [callback],
            "configurable": {"user_id": USER, "session_id": SESSION},
            "metadata": {"user_id": USER, "session_id": SESSION},
        }

        conversation = []

        def chat(message):
            """Send a message and get agent response."""
            ag.emit_user_input(USER, SESSION, message)
            conversation.append(("human", message))
            result = app.invoke({"messages": conversation}, config=config)
            ai_msg = result["messages"][-1].content
            conversation.append(("ai", ai_msg))
            return ai_msg

        def print_response(label, text, max_len=600):
            print(f"\n{label}")
            print("-" * 40)
            trimmed = text[:max_len] + ("..." if len(text) > max_len else "")
            print(trimmed)

        # ────────────────────────────────────────────────────────────
        # PHASE 1: Define the problem
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  PHASE 1: Define the problem")
        print(DIVIDER)

        response = chat(
            "I need a Python function called `find_anomalies` that takes a "
            "list of numerical sensor readings and identifies anomalous values. "
            "An anomaly is any reading that deviates more than 2 standard "
            "deviations from the mean. Return a list of (index, value) tuples. "
            "Just acknowledge the task and summarize your understanding. "
            "Do NOT write code yet."
        )
        print_response("Agent acknowledges:", response)

        # ────────────────────────────────────────────────────────────
        # CHECKPOINT: Save state at the decision point
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  CHECKPOINT: Saving state before implementation")
        print(DIVIDER)

        checkpoint = ag.checkpoint(
            USER, SESSION, "pre-implementation",
            agent_memory={"task": "anomaly detection function"},
            conversation_history=list(conversation),
            label="Before choosing approach",
        )
        checkpoint_node = ag.get_active_branch(USER, SESSION).head_node_id
        saved_conversation = list(conversation)

        print(f"  Saved at node:        {checkpoint_node}")
        print(f"  Conversation length:  {len(conversation)} messages")
        print(f"  Checkpoint hash:      {checkpoint.hash[:12]}...")
        print(f"  Git SHA:              {checkpoint.filesystem_ref[:12]}...")
        print()
        print("  Agent state is frozen. Two branches will now diverge")
        print("  from this exact point.")

        # ────────────────────────────────────────────────────────────
        # BRANCH A: Statistics-heavy approach
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  BRANCH A: Pure statistics approach (numpy/scipy)")
        print(DIVIDER)

        branch_a_id = bt.create_branch(
            USER, SESSION, "approach-statistics",
            from_node=int(checkpoint_node),
            intent="Use numpy/scipy statistical methods",
        )
        token_tracker["current"] = 0
        start_a = time.time()

        response_a = chat(
            "Write find_anomalies using numpy and scipy. "
            "Use scipy.stats.zscore for z-score calculation. "
            "Include proper imports, type hints, and a docstring. "
            "Make it production-ready with edge case handling. "
            "Show the complete implementation."
        )
        time_a = time.time() - start_a
        token_tracker["a"] = token_tracker.get("current", 0)

        print_response("Branch A result:", response_a)

        stats_a = bt.get_branch_stats(USER, SESSION, "approach-statistics")
        nodes_a = ag.get_branch_nodes(USER, SESSION, branch_a_id)

        # ────────────────────────────────────────────────────────────
        # RESTORE: Revert to checkpoint
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  RESTORE: Reverting to checkpoint")
        print(DIVIDER)

        conversation.clear()
        conversation.extend(saved_conversation)
        ag.restore(checkpoint)

        print(f"  Conversation reset to {len(conversation)} messages")
        print(f"  Agent has NO memory of Branch A")
        print(f"  Starting fresh from the same decision point")

        # ────────────────────────────────────────────────────────────
        # BRANCH B: Pure Python approach (no dependencies)
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  BRANCH B: Pure Python approach (zero dependencies)")
        print(DIVIDER)

        bt.switch_branch(USER, SESSION, "main")
        branch_b_id = bt.create_branch(
            USER, SESSION, "approach-pure-python",
            from_node=int(checkpoint_node),
            intent="Use only Python stdlib, no external packages",
        )
        token_tracker["current"] = 0
        start_b = time.time()

        response_b = chat(
            "Write find_anomalies using ONLY Python standard library. "
            "No numpy, no scipy, no pandas. Calculate mean and standard "
            "deviation manually using math.sqrt and sum(). "
            "Include proper type hints and a docstring. "
            "Make it production-ready with edge case handling. "
            "Show the complete implementation."
        )
        time_b = time.time() - start_b
        token_tracker["b"] = token_tracker.get("current", 0)

        print_response("Branch B result:", response_b)

        stats_b = bt.get_branch_stats(USER, SESSION, "approach-pure-python")
        nodes_b = ag.get_branch_nodes(USER, SESSION, branch_b_id)

        # ────────────────────────────────────────────────────────────
        # COMPARE: Side-by-side analysis
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  COMPARISON: Branch A vs Branch B")
        print(DIVIDER)

        col_w = 20
        lbl_w = 28
        print(f"\n  {'Metric':<{lbl_w}} {'Statistics':>{col_w}} {'Pure Python':>{col_w}}")
        print(f"  {'_'*lbl_w} {'_'*col_w} {'_'*col_w}")
        print(f"  {'Branch name':<{lbl_w}} {'approach-statistics':>{col_w}} {'approach-pure-python':>{col_w}}")
        print(f"  {'DAG nodes':<{lbl_w}} {len(nodes_a):>{col_w}} {len(nodes_b):>{col_w}}")
        print(f"  {'LLM response time':<{lbl_w}} {f'{time_a:.1f}s':>{col_w}} {f'{time_b:.1f}s':>{col_w}}")
        print(f"  {'Response length (chars)':<{lbl_w}} {len(response_a):>{col_w}} {len(response_b):>{col_w}}")
        print(f"  {'External dependencies':<{lbl_w}} {'numpy, scipy':>{col_w}} {'none':>{col_w}}")

        # Action type breakdown
        types_a = Counter(n.action_type.value for n in nodes_a)
        types_b = Counter(n.action_type.value for n in nodes_b)
        all_types = sorted(set(list(types_a.keys()) + list(types_b.keys())))

        print(f"\n  Action breakdown:")
        for t in all_types:
            print(f"    {t:<{lbl_w-2}} {types_a.get(t, 0):>{col_w}} {types_b.get(t, 0):>{col_w}}")

        # DAG structure
        print(f"\n  DAG Structure:")
        print(f"    Shared checkpoint:   node {checkpoint_node}")
        if nodes_a:
            print(f"    Branch A range:      node {nodes_a[0].id} -> {nodes_a[-1].id}")
        if nodes_b:
            print(f"    Branch B range:      node {nodes_b[0].id} -> {nodes_b[-1].id}")
        print(f"    Branches diverge from the same point - results are")
        print(f"    directly comparable with no cross-contamination.")

        # All branches summary
        print(f"\n  All branches in session:")
        for b in bt.list_branches(USER, SESSION):
            node_count = len(ag.get_branch_nodes(USER, SESSION, b.branch_id))
            print(f"    {b.name:<28} status={b.status.value:<12} nodes={node_count}")

        # History / lineage check
        if nodes_a and nodes_b:
            history_a = ag.get_history(USER, SESSION, int(nodes_a[-1].id))
            history_b = ag.get_history(USER, SESSION, int(nodes_b[-1].id))
            shared = set(n.id for n in history_a) & set(n.id for n in history_b)
            print(f"\n  Lineage:")
            print(f"    Branch A full path:  {len(history_a)} nodes from root")
            print(f"    Branch B full path:  {len(history_b)} nodes from root")
            print(f"    Shared ancestors:    {len(shared)} nodes")

        # ────────────────────────────────────────────────────────────
        # VERDICT
        # ────────────────────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("  WHAT JUST HAPPENED")
        print(DIVIDER)
        print("""
  1. Agent received a coding task
  2. AgentGit saved a checkpoint BEFORE the agent chose an approach
  3. Branch A: Agent implemented using numpy/scipy
  4. State restored to checkpoint (agent forgot Branch A entirely)
  5. Branch B: Agent implemented using pure Python
  6. Both branches compared from identical starting state

  This is impossible with a linear conversation:
    - No context contamination between approaches
    - Both branches started with the exact same knowledge
    - Results are directly comparable
    - Full execution history preserved in the DAG

  Use cases:
    - A/B test different agent strategies
    - Compare LLM models on the same task
    - Explore risky approaches safely
    - Human-in-the-loop: reject and retry from checkpoint
        """)

        ag.close()
        print("  Done.\n")


if __name__ == "__main__":
    main()
