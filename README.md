# agentgraph

Behavior regression testing for LLM agents.

`agentgraph` is a monorepo with two packages:

| Package | Purpose |
|---|---|
| `agenttest` | Record/replay behavior regression testing (CLI + pytest plugin) |
| `agentgit` | Execution graph capture (LLM/tool events stored in SQLite DAG) |

If you care about catching agent behavior drift in CI, `agenttest` is the primary product.

## Why this exists

LLM agents are non-deterministic. Traditional "assert exact output" tests become noisy and brittle.

`agenttest` solves this by testing **consistency against an approved baseline**:

1. Record a baseline run once.
2. Replay on every change.
3. Fail when behavior drifts from what was approved.

This is a **consistency framework**, not a correctness oracle.

## Guarantees and Non-Guarantees

### What `agenttest` guarantees

- Structural drift is detected: added/removed/reordered LLM steps and route changes surface in replay.
- Locked replay can run with zero live provider calls for intercepted paths.
- Deltas/regressions are visible in comparison history and `.agentgit/` artifacts.
- CLI-first workflow for CI/CD: `record`, `replay`, `accept`, `status`.

### What it does not guarantee

- It does not prove the baseline is correct.
- It does not prove production outputs are globally correct for all unseen inputs.
- Semantic comparison is not perfect for all language/domain edge cases.

## How it works

```text
Your test (pytest + @agenttest marker)
  -> graph.invoke(..., config={callbacks:[langgraph_callback(...)]})
  -> agentgit callback emits LLM/tool events
  -> agenttest stores per-step call details (prompt/response/fingerprint/cache-hit)

Replay:
  -> gatekeeper/runtime interception checks baseline cache
  -> locked/selective/full mode decides cached vs live
  -> comparator aligns baseline vs replay and reports PASS / REGRESSION / DELTA
```

## Quickstart (CLI-first)

### 1. Install

```bash
# from repo root
pip install -e .
pip install -e agenttest/
```

### 2. Write one marked test

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

### 3. Record baseline once

```bash
agenttest record --name=test_support_refund
```

### 4. Replay in CI/local

```bash
# strict regression guard (default CI path)
agenttest replay --mode=locked

# when intentionally changing prompts/graph
agenttest replay --mode=selective test_support_refund

# accept reviewed replay as new baseline
agenttest accept test_support_refund
```

## Replay modes

| Mode | Behavior |
|---|---|
| `locked` | Cache-only for intercepted LLM calls; miss becomes failure path |
| `selective` | Cached calls reuse baseline; misses go live and are compared |
| `full` | Live re-record of all calls |

Use `locked` for normal CI regression checks, `selective` when evolving behavior, `full` for major redesigns or baseline rebuilds.

## CI/CD flow

`agenttest` CLI is designed to be the CI entrypoint.

```bash
agenttest replay --mode=locked --tier=always
agenttest replay --mode=locked --tier=local
agenttest replay --mode=selective --tier=ci-only
agenttest ci post-comment --pr=<number>
```

Example workflow file is at `.github/workflows/agenttest.yml`.

## Commands

```bash
agenttest replay --mode=locked [test_filter]
agenttest record --name=<pytest-k-filter>
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

## Integration requirements

For reliable recording/replay in LangGraph tests:

- Mark tests with `@pytest.mark.agenttest`.
- Pass `langgraph_callback(agenttest_session.ag.eventbus)` in invocation callbacks.
- Provide stable `configurable.user_id` and `configurable.session_id` in `config`.

For non-standard model paths that bypass LangChain `BaseChatModel`, manual wrapping (`replayer.wrap_model(...)`) may still be required.

## Repository layout

- `agenttest/`: regression testing framework (CLI, plugin, replayer, comparator)
- `agentgit/`: DAG/event storage and callback plumbing
- `.agentgit/`: baseline and run artifacts (commit this intentionally when accepting behavior changes)
- `tests/`: automated test suite for guarantees and integration behavior

## License

Apache-2.0
