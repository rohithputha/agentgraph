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
- **pytest plugin** — `agenttest_record`, `agenttest_replay`, `agenttest_auto` fixtures available automatically
- **CLI** — inspect recordings, baselines, and comparison history from the terminal
- **Assertion helpers** — `assert_no_regression()`, `assert_step_count()`

---

## Quickstart

```python
# conftest.py — shared session for your test suite
import pytest
from agenttest.session import AgentTestSession

@pytest.fixture(scope="session")
def agenttest_session(tmp_path_factory):
    session = AgentTestSession.standalone(
        project_dir=str(tmp_path_factory.mktemp("agenttest")),
        user_id="ci",
        session_id="my-agent",
    )
    yield session
    session.close()
```

```python
# test_my_agent.py
import pytest
from agentgit.langgraph_callback import langgraph_callback
from agenttest.pytest_plugin.assertions import assert_no_regression

@pytest.mark.agenttest
def test_record(agenttest_session, agenttest_record):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    with agenttest_record(name="my-baseline", set_as_baseline=True) as rec:
        graph = build_your_agent(callback, agenttest_session)
        graph.invoke({"messages": [...]})

    assert rec.step_count > 0

@pytest.mark.agenttest
@pytest.mark.baseline("my-baseline")
def test_regression(agenttest_session, agenttest_replay):
    callback = langgraph_callback(agenttest_session.ag.eventbus)

    with agenttest_replay(baseline_name="my-baseline", mode="full") as rep:
        graph = build_your_agent(callback, agenttest_session)
        graph.invoke({"messages": [...]})

    assert_no_regression(rep.comparison_result)
```

```bash
# Record once
pytest test_my_agent.py::test_record --agenttest-record --agenttest

# Replay on every CI run
pytest test_my_agent.py::test_regression --agenttest
```

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
