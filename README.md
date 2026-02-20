# agentgraph

> Your AI agent tried 3 approaches and failed. Which one almost worked?

agentgraph gives you **git-like execution tracking** and **record-replay regression testing** for LangGraph agents — without changing your agent code.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/built%20for-LangGraph-orange)](https://github.com/langchain-ai/langgraph)

---

Two packages. One purpose.

| Package | What it does |
|---------|-------------|
| **agentgit** | Records every LLM call, tool invocation, and agent decision into a persistent execution graph — queryable, branchable, restorable |
| **agenttest** | Sits on top of agentgit. Record your agent's correct behaviour once, then automatically flag regressions on every code change |

---

## See it in action

```python
from agentgit import AgentGit
from agentgit.langgraph_callback import langgraph_callback

ag = AgentGit(project_dir=".")
callback = langgraph_callback(ag.eventbus)

# Your existing LangGraph agent — nothing changes
result = graph.invoke(
    {"messages": [HumanMessage(content="Refund my order")]},
    config={
        "callbacks": [callback],
        "configurable": {"user_id": "alice", "session_id": "support-001"}
    }
)

# Every step is now recorded
nodes = ag.get_branch_nodes("alice", "support-001", branch_id=1)
for node in nodes:
    print(f"[{node.action_type}] tokens={node.token_count}")
# [LLM_CALL]  tokens=340
# [LLM_CALL]  tokens=512
```

Add regression testing with three more lines:

```python
from agenttest.session import AgentTestSession
from agenttest.recorder import Recorder

session = AgentTestSession.standalone(project_dir=".")

with Recorder(session, name="support-flow", set_as_baseline=True):
    graph.invoke(...)   # captured — this is now your baseline
```

On every future run, agenttest compares against that baseline and tells you exactly which step diverged first.

---

## Setup

```bash
git clone https://github.com/rohithputha/agentgraph.git
cd agentgraph

pip install -e .            # agentgit
pip install -e agenttest/   # agenttest + pytest plugin + CLI
```

**Requirements:** Python 3.8+, LangChain Core, LangGraph, SQLite (built-in), Git

---

## agentgit

### How it works

A LangChain callback handler intercepts every event fired by LangGraph and writes it as a node in a SQLite DAG. No agent code changes. No wrappers. One line in the `config`.

```
LangGraph graph
     │
     │  callbacks=[langgraph_callback(ag.eventbus)]
     ▼
  agentgit
     │
     ├── eventbus (pub/sub)
     ├── tracer   (event → DAG node)
     └── dag_store (SQLite)
              │
              ├── nodes     (every LLM call, tool call, agent turn)
              ├── branches  (execution paths)
              └── checkpoints (state snapshots)
```

### Query execution history

```python
# Full execution history for a session
nodes = ag.get_branch_nodes("alice", "support-001", branch_id=1)
for node in nodes:
    print(f"[{node.action_type}] {node.content[:80]}")
# [LLM_CALL]   classify: {"category": "billing"}
# [LLM_CALL]   respond:  {"content": "I can help with your billing issue..."}

# Get the path from root to any node
history = ag.get_history("alice", "support-001", node_id=10)

# Inspect a single node
node = ag.get_node("alice", "support-001", node_id=5)
print(node.token_count, node.duration_ms)
# 340  1203
```

### Branch execution paths

Like git branches, but for agent runs. Fork at any node to explore a different prompt, tool, or model — without losing the original path.

```python
from agentgit.tools.branch_tools import BranchTools

bt = BranchTools(ag)

# Fork the current execution
bt.create_branch("alice", "support-001", "experiment-gpt4")
bt.switch_branch("alice", "support-001", "experiment-gpt4")

# Run agent with different config...

stats = bt.get_branch_stats("alice", "support-001", "experiment-gpt4")
# {"node_count": 8, "tokens_used": 1240, "time_elapsed": 3.1}

# Compare branches, then abandon the worse one
bt.abandon_branch("alice", "support-001", "experiment-gpt4", reason="Higher cost, same quality")
```

### Checkpoint and restore

```python
from agentgit.tools.version_tools import VersionTools

vt = VersionTools(ag)

# Snapshot agent state (stored as a git commit)
cp_hash = vt.create_checkpoint(
    "alice", "support-001",
    name="before-prompt-change",
    agent_memory={"context": "..."},
    conversation_history=[...],
)

# Something broke — go back
vt.restore_checkpoint("alice", "support-001", cp_hash)
```

### Subscribe to events

```python
from agentgit.event import EventType

ag.eventbus.subscribe(EventType.LLM_CALL_END,
    lambda e: print(f"  {e.model} replied in {e.duration_ms}ms — {e.usage} tokens"))

ag.eventbus.subscribe(EventType.TOOL_CALL_END,
    lambda e: print(f"  tool '{e.tool_name}' → {e.content[:60]}"))
```

| Event | Fires when |
|-------|-----------|
| `LLM_CALL_START` | LLM request begins |
| `LLM_CALL_END` | LLM response received |
| `LLM_ERROR` | LLM call fails |
| `TOOL_CALL_START` | Tool execution begins |
| `TOOL_CALL_END` | Tool completes |
| `AGENT_TURN_END` | Agent turn completes |

---

## agenttest

agenttest wraps agentgit. `AgentTestSession` shares the same SQLite connection as `AgentGit`, so recordings, baselines, and comparisons live alongside the execution DAG — no separate database.

### The three-step workflow

```
1. Record   →  run agent once, capture every LLM call as a baseline
2. Replay   →  re-run agent, compare every step against baseline
3. Detect   →  flag divergences, identify root cause, label cascade effects
```

### pytest plugin — recommended

Install agenttest and the fixtures are available in every test automatically.

#### Record a baseline

```python
# test_support_agent.py
import pytest
from agentgit.langgraph_callback import langgraph_callback

@pytest.mark.agenttest
def test_record(agenttest_session, agenttest_record):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    with agenttest_record(name="support-v1", set_as_baseline=True) as rec:
        graph = build_agent(callback, agenttest_session)
        graph.invoke({"messages": [HumanMessage(content="Refund my order")]})

    assert rec.step_count == 2  # classify + respond
```

```bash
pytest test_support_agent.py --agenttest-record --agenttest -v
```

#### Replay on every CI run

```python
@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_regression(agenttest_session, agenttest_replay):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    with agenttest_replay(baseline_name="support-v1", mode="full") as rep:
        graph = build_agent(callback, agenttest_session)
        graph.invoke({"messages": [HumanMessage(content="Refund my order")]})

    assert rep.passed, f"Regression: {rep.root_cause_summary}"
    # AssertionError: "Regression: Step 0: similarity 0.00 below threshold 0.85"
```

```bash
pytest test_support_agent.py --agenttest -v
```

#### Zero-boilerplate: `agenttest_auto`

One fixture. The same test body records OR replays depending on which CLI flag you pass.

```python
@pytest.mark.agenttest
@pytest.mark.baseline("support-v1")
def test_auto(agenttest_session, agenttest_auto):
    callback = langgraph_callback(agenttest_session.ag.eventbus)
    graph = build_agent(callback, agenttest_session)

    # Unchanged agent test logic — works in record mode, replay mode, or neither
    result = graph.invoke({"messages": [HumanMessage(content="Refund my order")]})
    assert "billing" in result["messages"][-1].content.lower()

# pytest ... --agenttest --agenttest-record   →  records + sets baseline
# pytest ... --agenttest                      →  replays, fails on regression
# pytest ...                                  →  plain test, no recording
```

### Replay modes

| Mode | Live LLM calls | Best for |
|------|---------------|---------|
| `full` | Yes — same model, same inputs | Full regression gate, catches model drift |
| `locked` | No — 100% from cache | Fast, zero-cost CI |
| `selective` | Partial — cache hits where possible | Changed-step-only validation |

**Locked mode** — no API calls, no cost:

```python
with agenttest_replay(baseline_name="support-v1", mode="locked") as rep:
    graph = build_intercepted_agent(callback, agenttest_session, rep.middleware)
    graph.invoke(...)

print(rep.cache_stats)
# {"cache_hits": 6, "live_calls": 0, "cache_hit_rate": 1.0}
```

### Regression detection and root-cause analysis

When steps diverge, agenttest does more than flag them. It finds the **root cause** — the first step that independently broke — and labels all downstream failures as **cascade**, so you're not buried in noise.

```
V1 baseline          V2 replay (bug: classifier returns UPPERCASE)
────────────         ────────────────────────────────────────────
step 0  "billing"    step 0  "BILLING"   ← DIVERGE  (root cause)
step 1  "I can..."   step 1  "I can..."  ← MATCH
step 2  "technical"  step 2  "TECHNICAL" ← CASCADE  (caused by step 0)
step 3  "Let's..."   step 3  "Let's..."  ← MATCH
step 4  "general"    step 4  "GENERAL"   ← CASCADE  (caused by step 0)
step 5  "Thanks..."  step 5  "Thanks..." ← MATCH
```

One root cause. Two cascades. Fix step 0, everything else resolves.

### Assertion helpers

```python
from agenttest.pytest_plugin.assertions import assert_no_regression, assert_step_count

assert_no_regression(rep.comparison_result)
# AssertionError: "Regression detected: Step 0: similarity 0.00 below threshold 0.85"

assert_step_count(rep.comparison_result, exact_steps=6)
# AssertionError: "Expected exactly 6 steps, got 4"
```

### CLI

```bash
agenttest list                          # all recordings
agenttest show rec_abc123               # recording + LLM steps
agenttest baseline list                 # all baselines
agenttest baseline set my-base rec_abc  # promote a recording to baseline
agenttest history                       # all comparison runs
agenttest history --failed              # only failed comparisons
agenttest diff cmp_xyz789               # step-by-step breakdown of a comparison
```

---

## Run the demo

A fully working example ships with the repo. No API key needed — uses a deterministic mock LLM.

```bash
pytest examples/customer_support/ -v -s
```

It runs 8 tests back to back, each demonstrating one feature:

```
test_01_recording              6 LLM steps captured, baseline promoted
test_02_full_replay            6/6 steps matched — PASS
test_03_locked_replay          6/6 cache hits, 0 live calls, 100% hit rate
test_04_regression_detection   V2 agent (uppercase) — FAIL automatically detected
test_05_root_cause_and_cascade root cause at step 0, steps 2+4 labelled CASCADE
test_06_assertion_helpers      assert_no_regression + assert_step_count
test_07_baseline_marker        @pytest.mark.baseline resolves baseline by name
test_08_agenttest_auto         agenttest_auto — zero-boilerplate record/replay
```

---

## How the two packages connect

```
agentgit                          agenttest
────────                          ─────────
AgentGit                          AgentTestSession
  └── eventbus ──subscribe──────► _on_llm_call_end()
  └── dag_store ◄──shared conn──  TestStore
                                    └── recordings
                                    └── llm_call_details  (one per LLM step)
                                    └── comparisons
                                    └── tags              (baselines)
```

`AgentTestSession` wraps `AgentGit` and shares its SQLite connection. When a recording is active, agenttest's event subscriber writes a `LLMCallDetail` row alongside the normal DAG node — same database, same transaction, no sync needed.

---

## Project structure

```
agentgraph/
├── agentgit/                       # Execution tracking
│   ├── core.py                     # AgentGit — main entry point
│   ├── eventbus.py                 # Pub/sub event bus
│   ├── tracer.py                   # Event → DAG node writer
│   ├── langgraph_callback.py       # LangChain callback handler
│   ├── models/dag.py               # ExecutionNode, Branch, Checkpoint
│   ├── storage/dag_store.py        # SQLite persistence
│   └── tools/                      # BranchTools, VersionTools
│
├── agenttest/                      # Regression testing
│   ├── session.py                  # AgentTestSession
│   ├── recorder.py                 # Recorder context manager
│   ├── replayer.py                 # Replayer context manager
│   ├── comparator.py               # LCS-based comparison engine
│   ├── cascade.py                  # Cascade detection
│   ├── fingerprint.py              # Response fingerprinting (SHA-256)
│   ├── interceptors/               # LLM gatekeeper (locked/selective)
│   ├── storage/test_store.py       # Recordings, comparisons, baselines
│   ├── cli/main.py                 # agenttest CLI
│   └── pytest_plugin/              # Fixtures + assertion helpers
│
└── examples/
    └── customer_support/           # Runnable demo, no API key needed
```

---

## Contributing

Contributions are welcome. The codebase has no magic — everything is explicit and easy to follow.

```bash
git clone https://github.com/rohithputha/agentgraph.git
cd agentgraph
pip install -e .
pip install -e agenttest/
pytest examples/customer_support/ -v   # make sure the demo passes
```

Open an issue before starting large changes.

---

## License

Apache-2.0
