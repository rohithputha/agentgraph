# agenttest

**Record-replay regression testing for LangGraph agents.**

agenttest captures every LLM call your agent makes during a run, stores it as a named baseline, and automatically detects regressions when you replay against that baseline. Think of it as pytest for agent behaviour — not just individual functions.

Built on top of [agentgit](../README.md). Shares its SQLite database. No separate infrastructure needed.

---

## Features

- **Record** — capture every LLM call (prompt, response, fingerprint, token usage) into a named baseline
- **Replay** — re-run your agent and compare every step against the baseline
- **Three replay modes** — `full` (live calls), `locked` (100% cache, zero cost), `selective` (partial cache)
- **Root-cause analysis** — identifies the first step that independently broke vs downstream cascade effects
- **pytest plugin** — auto-wraps `@pytest.mark.agenttest` tests from CLI (`--agenttest`) with no `Replayer` boilerplate
- **CLI** — inspect recordings, baselines, and comparison history from the terminal
- **Assertion helpers** — `assert_no_regression()`, `assert_step_count()`

---

## Quickstart

```python
# test_my_agent.py
import pytest
from agentgit.langgraph_callback import langgraph_callback

@pytest.mark.agenttest
@pytest.mark.baseline("my-baseline")
def test_my_agent(agenttest_session):
    callback = langgraph_callback(agenttest_session.ag.eventbus)
    graph = build_your_agent(callback, agenttest_session)
    result = graph.invoke({"messages": [...]})
    assert result is not None
```

```bash
# Record once
pytest test_my_agent.py::test_my_agent --agenttest --agenttest-record

# Replay on every CI run
pytest test_my_agent.py::test_my_agent --agenttest --agenttest-mode=locked
```

For custom or non-standard models, `replayer.wrap_model(llm)` remains available as a fallback.

---

## CLI

```bash
agenttest list                          # all recordings
agenttest show <recording_id>           # recording detail + LLM steps
agenttest baseline list                 # all baselines
agenttest baseline set <name> <rec_id>  # promote recording to baseline
agenttest history                       # all comparison runs
agenttest history --failed              # only failed comparisons
agenttest diff <comparison_id>          # step-by-step breakdown
```

---

## Running the demo

A fully working example ships with the repo. No API key required.

```bash
pytest examples/customer_support/ -v -s
```

---

## License

Apache-2.0
