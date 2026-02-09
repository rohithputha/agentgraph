"""
AgentTest pytest plugin
"""
import pytest


def pytest_addoption(parser):
    """Add agenttest CLI options to pytest"""
    group = parser.getgroup("agenttest")
    group.addoption(
        "--agenttest",
        action="store_true",
        help="Enable agenttest recording/replay",
    )
    group.addoption(
        "--agenttest-mode",
        default="selective",
        help="Replay mode: locked, selective, or full",
    )


def pytest_configure(config):
    """Register agenttest marker"""
    config.addinivalue_line(
        "markers",
        "agenttest: mark test as using agenttest recording/replay",
    )


@pytest.fixture
def agenttest(request):
    """Fixture for agenttest functionality"""
    # Implementation pending
    class AgentTestFixture:
        def record(self, name):
            raise NotImplementedError("AgentTest recording not yet implemented")

        def replay(self, baseline, mode="selective"):
            raise NotImplementedError("AgentTest replay not yet implemented")

    return AgentTestFixture()
