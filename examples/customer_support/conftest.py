"""
Shared fixtures for the Customer Support AgentTest demo.

Provides a single AgentTestSession that persists for the entire pytest
session so all tests share the same recordings database.  This lets
later tests replay against baselines recorded by earlier tests.

Run the full demo:
    pytest examples/customer_support/ -v -s

Run just one feature:
    pytest examples/customer_support/ -v -s -k "locked"
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agenttest.session import AgentTestSession


# ─── shared project dir ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_dir(tmp_path_factory):
    """One temp directory for the whole pytest session."""
    return str(tmp_path_factory.mktemp("agenttest-cs-demo"))


# ─── override the plugin's agenttest_session with a session-scoped version ──

@pytest.fixture(scope="session")
def agenttest_session(project_dir):
    """
    Session-scoped AgentTestSession shared by all tests.

    Overrides the plugin's default fixture so that every test in this
    directory reads from and writes to the same SQLite database.
    That means a baseline recorded in test_01 is immediately visible
    to the Replayer in test_02.
    """
    session = AgentTestSession.standalone(
        project_dir=project_dir,
        user_id="demo",
        session_id="customer-support",
    )
    yield session
    session.close()
