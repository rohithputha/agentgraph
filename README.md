# AgentGit

**Multi-user DAG-based execution tracking and versioning for LangGraph agents**

AgentGit provides Git-like branching and checkpointing for AI agent workflows, with full multi-user and multi-session support. Track every LLM call, tool invocation, and agent decision in a persistent execution graph that you can explore, branch, and restore.

## Key Features

- **DAG-Based Execution Tracking** - Every agent action becomes a node in a directed acyclic graph
- **Git-Like Branching** - Create execution branches to explore different agent paths
- **Checkpoint & Restore** - Save agent state and restore to any point in history
- **Multi-User Sessions** - Isolated execution contexts per user and session
- **Event-Driven Architecture** - Subscribe to LLM calls, tool executions, and agent turns
- **LangGraph Integration** - Automatic tracking via callback handlers
- **SQLite Persistence** - All execution history stored in local database
- **Git Backend** - Checkpoints stored as git commits for versioning

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rohithputha/agentgit.git
cd agentgit

# Install dependencies
pip install -r requirements.txt

# Set your API key (for examples)
export GOOGLE_API_KEY="your-api-key"
```

### Basic Usage

```python
from core import AgentGit

# Initialize AgentGit
ag = AgentGit(project_dir=".")

# Create a session
user_id = "alice"
session_id = "project-alpha"

# Create main branch
ag.create_branch(user_id, session_id, "main", intent="Main execution")

# Get callback for LangGraph integration
callback = ag.get_callback()

# Use with LangGraph
from langgraph.graph import StateGraph
graph = StateGraph(YourState)
# ... define your graph ...
app = graph.compile()

# All agent activity is automatically tracked!
app.invoke(
    {"messages": [("human", "Hello!")]},
    config={
        "callbacks": [callback],
        "configurable": {"user_id": user_id, "session_id": session_id}
    }
)

# Query execution history
nodes = ag.get_branch_nodes(user_id, session_id, branch_id=1)
for node in nodes:
    print(f"{node.action_type}: {node.content}")

ag.close()
```

## Core Concepts

### Sessions and Users

AgentGit supports **multi-user** and **multi-session** isolation:

- **User ID**: Identifies different users (e.g., "alice", "bob")
- **Session ID**: Identifies different conversation threads (e.g., "project-alpha", "debug-session")

Each (user_id, session_id) pair has its own isolated execution graph.

### Branches

Like Git branches, AgentGit branches let you explore different execution paths:

```python
from tools.branch_tools import BranchTools

bt = BranchTools(ag)

# Create a feature branch from current state
feat_id = bt.create_branch(user_id, session_id, "feature-experiment")

# Switch between branches
bt.switch_branch(user_id, session_id, "feature-experiment")
# ... run agent work ...
bt.switch_branch(user_id, session_id, "main")

# View branch statistics
stats = bt.get_branch_stats(user_id, session_id, "feature-experiment")
# {'node_count': 12, 'tokens_used': 1543, 'time_elapsed': 8.2}
```

### Execution Nodes

Every action creates a node in the DAG:

| Action Type | Description | Triggered By |
|------------|-------------|--------------|
| `USER_INPUT` | User message | `ag.emit_user_input()` |
| `LLM_CALL` | LLM request sent | LangGraph callback |
| `LLM_RESPONSE` | LLM response received | LangGraph callback |
| `TOOL_CALL` | Tool execution | LangGraph callback |
| `CHECKPOINT` | State snapshot | `ag.checkpoint()` |

### Checkpoints

Save and restore agent state at any point:

```python
from tools.version_tools import VersionTools

vt = VersionTools(ag)

# Create checkpoint
checkpoint_hash = vt.create_checkpoint(
    user_id, session_id,
    name="v1.0",
    agent_memory={"context": "..."},
    conversation_history=[...],
    label="Before experiment"
)

# Continue working...
# ... agent runs ...

# Restore previous state
vt.restore_checkpoint(user_id, session_id, checkpoint_hash)

# List all checkpoints
checkpoints = vt.list_checkpoints(user_id, session_id)
```

Checkpoints are stored as **git commits** in `.agentgit/checkpoints/` with full parent chain tracking.

## API Reference

### AgentGit

Main interface for execution tracking:

```python
from core import AgentGit

ag = AgentGit(project_dir=".", agit_dir=".agentgit")
```

#### Session Management

```python
# Emit user input
ag.emit_user_input(user_id, session_id, "Hello!", metadata={})

# Subscribe to events
ag.on(EventType.LLM_CALL_END, lambda event: print(event.text))
ag.on_all(lambda event: print(event.type))
```

#### Branch Operations

```python
# Create branch
branch_id = ag.create_branch(user_id, session_id, "main", intent="", base_node_id=None)

# List branches
branches = ag.list_branches(user_id, session_id, status=BranchStatus.ACTIVE)

# Get active branch
branch = ag.get_active_branch(user_id, session_id)
```

#### Node Inspection

```python
# Peek at node content
content = ag.peek(user_id, session_id, node_id=5)

# Get full node details
node = ag.get_node(user_id, session_id, node_id=5)

# Get execution path from root to node
history = ag.get_history(user_id, session_id, node_id=10)

# Get all nodes in a branch
nodes = ag.get_branch_nodes(user_id, session_id, branch_id=1)
```

#### Checkpointing

```python
# Create checkpoint
checkpoint = ag.checkpoint(
    user_id, session_id,
    name="v1",
    agent_memory={},
    conversation_history=[],
    label="Checkpoint"
)

# Restore checkpoint
ag.restore(checkpoint)
```

#### LangGraph Integration

```python
# Get callback handler
callback = ag.get_callback()

# Use in LangGraph config
config = {
    "callbacks": [callback],
    "configurable": {"user_id": user_id, "session_id": session_id}
}
```

### BranchTools

High-level branch management:

```python
from tools.branch_tools import BranchTools

bt = BranchTools(ag)

# Create branch (auto-detects base node from active branch)
bt.create_branch(user_id, session_id, "feature", from_node=None, intent="")

# Switch branches (marks old as COMPLETED, new as ACTIVE)
bt.switch_branch(user_id, session_id, "feature")

# Get branch by name
branch = bt.get_branch(user_id, session_id, "main")

# Get active branch
active = bt.get_active_branch(user_id, session_id)

# List all branches
branches = bt.list_branches(user_id, session_id, status=None)

# Get branch statistics
stats = bt.get_branch_stats(user_id, session_id, "main")
# Returns: {node_count, tokens_used, time_elapsed, head_node_id, base_node_id}

# Mark branch as abandoned
bt.abandon_branch(user_id, session_id, "feature", reason="Not needed")

# Mark branch as completed
bt.complete_branch(user_id, session_id, "feature", reason="Merged")

# Get all nodes in branch
nodes = bt.get_branch_nodes(user_id, session_id, "main")
```

### VersionTools

Checkpoint lifecycle management:

```python
from tools.version_tools import VersionTools

vt = VersionTools(ag)

# Create checkpoint (returns hash)
cp_hash = vt.create_checkpoint(
    user_id, session_id, "v1",
    agent_memory={},
    conversation_history=[],
    label="Checkpoint"
)

# Restore checkpoint
vt.restore_checkpoint(user_id, session_id, cp_hash)

# List checkpoints
checkpoints = vt.list_checkpoints(user_id, session_id)
# Returns: [{"hash": "abc123", "label": "...", "created_at": ..., "node_id": ...}, ...]

# Get checkpoint details
cp_info = vt.get_checkpoint(user_id, session_id, cp_hash)

# Get checkpoint at specific node
cp_info = vt.get_checkpoint_at_node(user_id, session_id, node_id=5)

# Compare two checkpoints
diff = vt.compare_checkpoints(user_id, session_id, hash1, hash2)
# Returns: {"files_changed": [...], "memory_diff": {...}, "history_diff": {...}}

# Get latest checkpoint
latest = vt.get_latest_checkpoint(user_id, session_id)

# Restore to specific node (creates checkpoint at that node)
vt.restore_to_node(user_id, session_id, node_id=5)
```

## Event System

Subscribe to real-time agent events:

```python
from event import EventType

# Subscribe to specific events
ag.on(EventType.LLM_CALL_START, lambda e: print(f"LLM called: {e.model}"))
ag.on(EventType.LLM_CALL_END, lambda e: print(f"Response: {e.text}"))
ag.on(EventType.TOOL_CALL_END, lambda e: print(f"Tool {e.tool_name}: {e.content}"))

# Subscribe to all events
ag.on_all(lambda e: print(f"Event: {e.type}"))

# Emit custom events
ag.emit(EventType.USER_INPUT, Event(...))
```

### Available Events

| Event Type | Fired When | Key Fields |
|-----------|------------|------------|
| `USER_INPUT` | User message emitted | `content`, `user_id`, `session_id` |
| `LLM_CALL_START` | LLM request begins | `model`, `messages`, `run_id` |
| `LLM_CALL_END` | LLM response received | `text`, `usage`, `duration_ms` |
| `LLM_ERROR` | LLM call fails | `error`, `model` |
| `TOOL_CALL_START` | Tool execution begins | `tool_name`, `tool_args` |
| `TOOL_CALL_END` | Tool completes | `tool_name`, `content`, `duration_ms` |
| `TOOL_ERROR` | Tool fails | `tool_name`, `error` |
| `AGENT_TURN_END` | Agent turn completes | `outputs`, `run_id` |

## Architecture

### Storage Layout

```
project/
└── .agentgit/
    ├── dag.sqlite              # Execution nodes, branches, checkpoints
    ├── checkpoints/            # Git repository for state snapshots
    │   └── .git/
    └── files/                  # Project file snapshots (optional)
```

### Database Schema

**nodes** table:
- `id` - Auto-incrementing node ID
- `user_id`, `session_id` - Session context
- `parent_id` - Previous node in execution chain
- `branch_id` - Which branch this node belongs to
- `action_type` - USER_INPUT, LLM_CALL, TOOL_CALL, etc.
- `content` - JSON payload (messages, tool args, etc.)
- `checkpoint_sha` - Git SHA if this is a checkpoint
- `timestamp`, `duration_ms`, `token_count` - Metrics

**branches** table:
- `branch_id` - Auto-incrementing branch ID
- `user_id`, `session_id` - Session context
- `name` - Branch name (unique per session)
- `head_node_id` - Latest node in branch
- `base_node_id` - Where branch diverged from
- `status` - ACTIVE, COMPLETED, ABANDONED
- `intent` - Why this branch was created

**checkpoints** table:
- `hash` - Checkpoint identifier
- `node_id` - DAG node this checkpoint represents
- `filesystem_ref` - Git commit SHA
- `memory` - Agent memory snapshot (JSON)
- `history` - Conversation history (JSON)
- `files_changed` - Modified files list

### Design Principles

1. **Stateless Core** - No in-memory state tracking. All state in SQLite.
2. **Session Isolation** - Each (user_id, session_id) has independent graph.
3. **Event-Driven** - Components communicate via EventBus pub/sub.
4. **Git Backend** - Checkpoints stored as commits for versioning.
5. **Callback Integration** - Automatic tracking via LangGraph callbacks.

## Examples

See [`examples/test_complex_flow.py`](examples/test_complex_flow.py) for a comprehensive demo including:

- Multi-user, multi-session workflows
- Branch creation and switching
- Checkpoint creation and restoration
- LangGraph integration with tools
- Event subscription and tracking
- Node inspection and history queries

```bash
# Run the comprehensive test
python examples/test_complex_flow.py
```

## Use Cases

1. **Agent Debugging** - Inspect every step of agent execution
2. **A/B Testing** - Run same task on different branches, compare results
3. **State Restoration** - Rewind to any point and try different paths
4. **Multi-User Apps** - Isolated sessions per user with shared codebase
5. **Experiment Tracking** - Version agent behavior with checkpoints
6. **Execution Analysis** - Query DAG for patterns, token usage, timing

## Requirements

- Python 3.8+
- SQLite 3.x
- Git (for checkpoint backend)
- LangChain Core
- LangGraph

See `requirements.txt` for full dependencies.

## Project Structure

```
agentgit/
├── src/
│   ├── core.py                    # AgentGit main interface
│   ├── eventbus.py                # Event pub/sub system
│   ├── event.py                   # Event types and dataclasses
│   ├── tracer.py                  # Event → DAG node converter
│   ├── langgraph_callback.py      # LangChain callback handler
│   ├── models/
│   │   └── dag.py                 # ExecutionNode, Branch, Checkpoint models
│   ├── storage/
│   │   ├── dag_store.py           # SQLite persistence
│   │   ├── checkpoint_store.py    # Git-backed checkpoints
│   │   ├── git_backend.py         # Git operations
│   │   └── schema.sql             # Database schema
│   └── tools/
│       ├── branch_tools.py        # Branch management helpers
│       └── version_tools.py       # Checkpoint helpers
├── examples/
│   └── test_complex_flow.py       # Comprehensive demo
└── requirements.txt
```

## Troubleshooting

### No events firing

- Ensure callback is passed in `config["callbacks"]`
- Verify session context in `config["configurable"]` or `config["metadata"]`
- Check eventbus subscriptions: `ag.on_all(print)` to see all events

### Branch not found

- Call `ag.create_branch()` before using session
- Verify user_id and session_id match exactly
- Check active branch: `ag.get_active_branch(user_id, session_id)`

### Checkpoint restore fails

- Ensure `.agentgit/checkpoints/.git/` exists
- Check checkpoint hash exists: `vt.get_checkpoint(user_id, session_id, hash)`
- Verify git is installed and accessible

### Database locked

- Close previous AgentGit instances: `ag.close()`
- SQLite doesn't support concurrent writes from multiple processes
- Use separate sessions per user/process

## Contributing

Contributions welcome! This framework is designed to be educational and extensible.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT

## Related Projects

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
