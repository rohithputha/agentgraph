"""
Pytest plugin for AgentTest.

Usage:
    pytest --agenttest
    pytest --agenttest-mode=locked
    pytest --agenttest-record
"""

pytest_configure = None
pytest_addoption = None
agenttest_session = None
agenttest_record = None
agenttest_replay = None

try:
    from agenttest.pytest_plugin.plugin import (
        pytest_configure,
        pytest_addoption,
        agenttest_session,
        agenttest_record,
        agenttest_replay,
    )
except Exception:
    # Keep assertions importable when pytest is not installed.
    pass

from agenttest.pytest_plugin.assertions import (
    assert_no_regression,
    assert_step_count,
    assert_no_new_errors,
)

__all__ = [
    "pytest_configure",
    "pytest_addoption",
    "agenttest_session",
    "agenttest_record",
    "agenttest_replay",
    "assert_no_regression",
    "assert_step_count",
    "assert_no_new_errors",
]
