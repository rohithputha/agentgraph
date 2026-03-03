# agenttest

Record/replay behavior regression testing for LLM agents.

`agenttest` lets you approve agent behavior once, then enforce that behavior in CI.
It is built on top of `agentgit` and stores baselines/comparisons in `.agentgit/`.

## Core idea

`agenttest` validates **consistency with approved baselines**.

- It detects drift from what you already reviewed.
- It does not claim the baseline itself is universally correct.

## CLI-first workflow

### 1) Pick execution style

Pytest style:

```python
import pytest
from agentgit.langgraph_callback import langgraph_callback

@pytest.mark.agenttest
@pytest.mark.baseline("my-agent-baseline")
def test_my_agent(agenttest_session):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    graph = build_agent(callback, agenttest_session)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Plan my trip"}]},
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

Standalone scenario style (no pytest test required):

```toml
[agenttest]
default_replay_mode = "locked"

[[scenarios]]
name = "trip_flow"
entrypoint = "myapp.scenarios:trip_flow"
expects_llm = true
input = { query = "Plan my trip" }
```

Where `myapp.scenarios:trip_flow` is an importable callable that runs the agent.

### 2) Record baseline once

```bash
# pytest backend
agenttest record --name=test_my_agent

# standalone scenario backend
agenttest record --scenario=trip_flow
```

### 3) Replay on changes

```bash
# strict regression mode
agenttest replay --mode=locked

# intentional change mode
agenttest replay --mode=selective test_my_agent

# standalone scenario replay
agenttest replay --scenario=trip_flow --mode=locked
```

### 4) Accept reviewed change

```bash
agenttest accept test_my_agent
```

## Modes

| Mode | What happens |
|---|---|
| `locked` | Intercepted LLM calls must hit cache; no live provider call should be needed |
| `selective` | Cache hits use baseline; misses go live and are compared |
| `full` | Re-record everything live |

## Result semantics

| Outcome | Meaning |
|---|---|
| `PASS` | Replay matched approved behavior |
| `REGRESSION` | Existing approved behavior changed unexpectedly |
| `DELTA` | New/changed behavior detected and needs explicit acceptance |

## How it works internally

1. **Recording**: callback events capture LLM step metadata and responses.
2. **Fingerprinting**: structural signature (provider/method/model/roles/tools) supports alignment.
3. **Replay interception**: gatekeeper/runtime checks cache for each LLM call.
4. **Comparison**: baseline and replay steps are aligned and scored.
5. **Storage**: results are written to `.agentgit/` for audit/history.

## CI usage

Use `agenttest` commands directly in workflows:

```bash
agenttest replay --mode=locked --tier=always
agenttest replay --mode=locked --tier=local
agenttest replay --mode=selective --tier=ci-only
agenttest ci post-comment --pr=<number>
```

See `.github/workflows/agenttest.yml` for an end-to-end reference.

## Commands

```bash
agenttest replay --mode=locked [test_filter]
agenttest replay --scenario=<scenario_name> --mode=locked
agenttest record --name=<pytest-k-filter>
agenttest record --scenario=<scenario_name>
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

## Requirements for reliable runs

- Use `@pytest.mark.agenttest` for tests that should be auto-wrapped.
- Pass `langgraph_callback(agenttest_session.ag.eventbus)` in callbacks.
- Set stable `configurable.user_id` / `configurable.session_id` in invoke config.

For standalone scenarios, make sure `entrypoint = "module:function"` is importable from the current environment.
For custom/non-LangChain model paths, manual model wrapping can still be used as a fallback.

## License

Apache-2.0
