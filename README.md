# agentgraph

Behavior regression testing for LLM agents.

`agentgraph` is a monorepo with two packages:

| Package | Purpose |
|---|---|
| `agenttest` | Record/replay behavior regression testing (CLI + pytest plugin) |
| `agentgit` | Execution graph capture (LLM/tool events stored in SQLite DAG) |

If your goal is "catch behavior drift before merge", focus on `agenttest`.

## What this gives you

LLM agents are non-deterministic, so exact-output assertions are noisy.
`agenttest` turns this into a baseline consistency workflow:

1. Record a known-good run.
2. Replay after code changes.
3. Detect drift as `PASS`, `REGRESSION`, or `DELTA`.
4. Accept intentional changes explicitly and commit updated baseline.

This is a consistency framework, not a correctness oracle.

## Quick Start (CLI-first)

### 1) Install

```bash
# from repo root
pip install -e .
pip install -e agenttest/
```

### 2) Write one marked test

```python
import pytest
from agentgit.langgraph_callback import langgraph_callback

@pytest.mark.agenttest
@pytest.mark.baseline("support-refund")
def test_support_refund(agenttest_session):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    graph = build_agent(callback, agenttest_session)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Refund my order"}]},
        config={
            "callbacks": [callback],
            "configurable": {
                "user_id": "pytest",
                "session_id": "test-session",
            },
        },
    )

    assert result is not None
```

If you do not want pytest tests for experimentation, define standalone scenarios in `agenttest.toml`:

```toml
[agenttest]
default_replay_mode = "locked"

[[scenarios]]
name = "refund_flow"
entrypoint = "myapp.scenarios:run_refund_flow"
expects_llm = true
input = { query = "Refund my order" }
```

### 3) Record baseline once

```bash
# pytest backend
agenttest record --name=test_support_refund

# standalone scenario backend (no pytest test required)
agenttest record --scenario=refund_flow
```

### 4) Replay in CI/local

```bash
# strict regression guard (default path)
agenttest replay --mode=locked

# when intentionally changing prompts/graph
agenttest replay --mode=selective test_support_refund

# standalone scenario replay
agenttest replay --scenario=refund_flow --mode=locked

# accept reviewed replay as new baseline
agenttest accept test_support_refund
```

## Daily Workflow

### Case A: "I only refactored code"

```bash
agenttest replay --mode=locked
```

Expected: `PASS`.

### Case B: "I intentionally changed prompt / route / tool usage"

```bash
agenttest replay --mode=selective <test-filter>
agenttest diff <comparison_id>
agenttest accept <test-filter>
```

Expected: `DELTA` during review, then baseline update.

### Case C: "Large redesign or baseline rebuild"

```bash
agenttest record --name=<pytest-k-filter>
agenttest set-baseline <baseline_name>
```

Expected: full re-record.

## Replay Modes

| Mode | What it does | Typical use |
|---|---|---|
| `locked` | Cache-only behavior for intercepted calls; misses fail | CI regression gate |
| `selective` | Cache hits reuse baseline, misses go live and get compared | Intentional behavior evolution |
| `full` | Live re-record of all calls | Baseline rebuild / major redesign |

## Where it works well today

- Pytest-driven LangGraph/LangChain tests using `@pytest.mark.agenttest`.
- Standalone scenario runs via `agenttest record/replay --scenario ...` (no pytest test file needed).
- CLI-first execution in local and CI: `agenttest record/replay/accept/status`.
- Structured behavior history: `agenttest history`, `agenttest diff`.
- Tiered CI split (`always`, `local`, `ci-only`) via `agenttest replay --tier=...`.
- Locked-mode safety for known provider hosts through runtime network guard.

## Current limitations (important)

- In pytest-agent flow, callback wiring is still the most reliable capture path. Standalone scenarios can capture without callback via runtime interception.
- Auto replay wrapping depends on marker/plugin path. Tests without `@pytest.mark.agenttest` are not managed automatically.
- Standalone scenarios avoid pytest markers, but entrypoint functions must be importable (`module:function`).
- Non-standard model invocation paths (bypassing LangChain `BaseChatModel`) may need manual fallback integration.
- Semantic equivalence scoring is currently basic; paraphrase-heavy domains can still produce noise.
- Determinism boundaries for time/UUID/external side effects are not fully productized yet.

## Expected failure modes and what to do

| Symptom | Likely cause | Fix |
|---|---|---|
| Replay captures 0 steps | Callback/context not wired | Ensure callback + stable `configurable.user_id/session_id` |
| Locked replay fails on cache miss | Real behavior changed or non-deterministic input changed | Use selective to review delta, then accept if intentional |
| Selective made live calls you did not expect | Interception path incomplete for that model/tool flow | Use supported integration path, then baseline-update flow |
| CI pass locally but fail in pipeline | Tier mismatch or missing baseline commit | Run same `agenttest replay --tier=...` locally and commit `.agentgit/` |

## CI/CD Usage

Use CLI commands directly in workflows:

```bash
agenttest replay --mode=locked --tier=always
agenttest replay --mode=locked --tier=local
agenttest replay --mode=selective --tier=ci-only
agenttest ci post-comment --pr=<number>
```

Reference workflow: `.github/workflows/agenttest.yml`.

## Commands

```bash
agenttest replay --mode=locked [test_filter]
agenttest replay --scenario=<scenario_name> --mode=locked
agenttest record --name=<pytest-k-filter>          # pytest backend
agenttest record --scenario=<scenario_name>        # standalone backend
agenttest accept [test_name]
agenttest status
agenttest history
agenttest diff <comparison_id>
agenttest list
agenttest show <recording_id>
agenttest baseline list
agenttest baseline set <name> <recording_id>
agenttest pull-baseline --from-run <run_id>
```

## Integration checklist

- Mark test with `@pytest.mark.agenttest`.
- Use `@pytest.mark.baseline("...")` for stable baseline naming.
- Pass `langgraph_callback(agenttest_session.ag.eventbus)` in graph calls.
- Set stable `configurable.user_id` + `configurable.session_id`.
- Commit `.agentgit/` when accepting intentional behavior changes.

## In Pipeline

Planned improvements currently in progress:

- Tool stubbing support: model behavior under different tool categories and replay modes.
- Semantic similarity scoring improvements by mode (to reduce paraphrase noise and improve intent detection).
- Hardening selective/locked behavior across more graph wiring and edge cases.
- Oracle support: explicit determinism boundaries (time, UUID, external volatility) and documented breakage behavior.

## Repository layout

- `agenttest/`: regression testing framework (CLI, plugin, replayer, comparator)
- `agentgit/`: DAG/event storage and callback plumbing
- `.agentgit/`: baseline and run artifacts (commit intentionally when accepting behavior changes)
- `tests/`: automated test suite for guarantees and integration behavior

## License

Apache-2.0
