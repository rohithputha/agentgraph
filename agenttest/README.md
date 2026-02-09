# AgentTest

**Record-replay regression testing framework for AI agents**

AgentTest is a testing framework that captures LLM calls during agent execution and replays them to detect regressions. Think "VCR.py for AI agents."

## Status

🚧 **Under Development** - Core functionality not yet implemented.

## Planned Features

- **Record**: Capture all LLM interactions during agent execution
- **Replay**: Re-run tests using recorded responses (locked/selective/full modes)
- **Compare**: Detect regressions via semantic comparison
- **Pytest Integration**: Native pytest plugin with fixtures and assertions
- **CLI**: Manage recordings and baselines from the command line

## Installation

```bash
pip install -e ./agenttest
```

## Usage

Coming soon...

## Architecture

AgentTest is part of the agentgraph monorepo but is independently installable and does not depend on agentgit.

## License

Apache-2.0
