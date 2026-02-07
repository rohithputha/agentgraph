"""
AgentGraph Core - LangGraph-integrated interface for agent execution tracking.

Usage:
    from core import AgentGraph
    
    # Initialize
    ag = AgentGraph("/path/to/project")
    
    # Get callback for LangGraph integration
    callback = ag.get_callback()
    
    # Use with LangGraph
    from langgraph.graph import StateGraph
    graph = StateGraph(...)
    app = graph.compile()
    pp.invoke({"input": "..."}, config={"calalbacks": [callback]})
    
    # Peek at node memory
    memory = ag.peek(node_id)
    
    # Clean up
    ag.close()
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable

from storage.dag_store import DagStore
from storage.checkpoint_store import CheckpointStore
from eventbus import Eventbus
from event import Event, EventType
from tracer import Tracer
from langgraph_callback import langgraph_callback
from models.dag import (
    ExecutionNode, Branch, ActionType, CallerType, BranchStatus, Checkpoint
)


class AgentGraph:
    """
    Main interface for AgentGraph - integrates with LangGraph via callbacks.
    
    This class initializes all storage components and provides:
    - LangGraph callback handler for automatic execution tracking
    - Event bus for custom event handling
    - DAG storage for execution history
    - Checkpoint management for state snapshots
    """

    def __init__(self, project_dir: str, agit_dir: str = ".agentgit"):
        """
        Initialize AgentGraph with project and storage directories.
        
        Args:
            project_dir: Path to the project being tracked
            agit_dir: Name of the agentgit directory (default: .agentgit)
        """
        self.project_dir = Path(project_dir).resolve()
        self.agit_path = self.project_dir / agit_dir
        
        # Create .agentgit directory if it doesn't exist
        self.agit_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.eventbus = Eventbus()
        self.dag_store = DagStore(str(self.agit_path / "dag.sqlite"))
        self.checkpoint_store = CheckpointStore(self.agit_path, self.project_dir)
        
        # Initialize tracer (subscribes to eventbus and records nodes)
        self._tracer = Tracer(self)
        
        # Track current state
        self.current_node_id: Optional[int] = None
        self.current_branch_id: Optional[int] = None
        self.current_thread_id: Optional[str] = None

    # ─── LangGraph Integration ─────────────────────────────────────

    def get_callback(self) -> langgraph_callback:
        """
        Get the LangGraph callback handler for automatic tracking.
        
        Use this callback with any LangGraph/LangChain invocation:
        
            callback = ag.get_callback()
            app.invoke(input, config={"callbacks": [callback]})
        """
        return langgraph_callback(self.eventbus)

    # ─── Event Subscription ────────────────────────────────────────

    def on(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to a specific event type."""
        self.eventbus.subscribe(event_type, callback)

    def on_all(self, callback: Callable[[Event], None]):
        """Subscribe to all event types."""
        self.eventbus.subscribe_all(callback)

    def emit(self, event_type: EventType, event: Event):
        """Manually emit an event (e.g., for user input)."""
        self.eventbus.publish(event_type, event)

    def emit_user_input(self, message: str, metadata: Optional[dict] = None):
        """Convenience method to emit a user input event."""
        self.eventbus.publish(
            EventType.USER_INPUT,
            Event(
                type=EventType.USER_INPUT,
                content=message,
                metadata=metadata or {},
                timestamp=datetime.now(),
            )
        )

    # ─── Branch Operations ─────────────────────────────────────────

    def create_branch(
        self,
        name: str,
        intent: str = "",
        base_node_id: Optional[int] = None,
    ) -> int:
        """Create a new branch and set it as current. Returns branch ID."""
        branch = Branch(
            name=name,
            thread_id=name,
            head_node_id=str(base_node_id) if base_node_id else "0",
            base_node_id=str(base_node_id) if base_node_id else "0",
            status=BranchStatus.ACTIVE,
            intent=intent,
            created_by=CallerType.SYSTEM,
            created_at=datetime.now(),
        )
        branch_id = self.dag_store.insert_branch(branch)
        self.current_branch_id = branch_id
        self.current_thread_id = name
        self.current_node_id = base_node_id
        return branch_id

    def switch_branch(self, name: str) -> bool:
        """Switch to an existing branch by name."""
        branch = self.dag_store.get_branch(name)
        if branch:
            row = self.dag_store.conn.execute(
                "SELECT branch_id, head_node_id FROM branches WHERE name = ?", (name,)
            ).fetchone()
            if row:
                self.current_branch_id = row[0]
                self.current_thread_id = name
                self.current_node_id = row[1]
                return True
        return False

    def list_branches(self, status: Optional[BranchStatus] = None) -> List[Branch]:
        """List all branches, optionally filtered by status."""
        return self.dag_store.list_branches(status)

    # ─── Node Operations ───────────────────────────────────────────

    def peek(self, node_id: int) -> Optional[dict]:
        """Peek at the memory (content) for a given node number."""
        return self.dag_store.peek(node_id)

    def get_node(self, node_id: int) -> Optional[ExecutionNode]:
        """Get full node details by ID."""
        return self.dag_store.get_node(node_id)

    def get_history(self, node_id: int) -> List[ExecutionNode]:
        """Get the path from root to a given node."""
        return self.dag_store.get_path_to_root(node_id)

    def get_branch_nodes(self, branch_id: Optional[int] = None) -> List[ExecutionNode]:
        """Get all nodes in a branch. Uses current branch if not specified."""
        bid = branch_id or self.current_branch_id
        if not bid:
            return []
        return self.dag_store.get_branch_nodes(bid)

    # ─── Checkpoint Operations ─────────────────────────────────────

    def checkpoint(
        self,
        name: str,
        agent_memory: dict,
        conversation_history: list,
        label: str = "Checkpoint",
    ) -> Checkpoint:
        """Create a checkpoint of current state."""
        return self.checkpoint_store.create_checkpoint(
            name, agent_memory, conversation_history, label
        )

    def restore(self, checkpoint: Checkpoint):
        """Restore from a checkpoint."""
        self.checkpoint_store.restore_checkpoint(checkpoint)

    # ─── Properties ────────────────────────────────────────────────

    @property
    def store(self) -> DagStore:
        """Access to underlying DAG store (for Tracer compatibility)."""
        return self.dag_store

    @property
    def current_branch(self) -> Optional[Branch]:
        """Get the current active branch."""
        if self.current_branch_id:
            return self.dag_store.get_branch_by_id(self.current_branch_id)
        return None

    # ─── Lifecycle ─────────────────────────────────────────────────

    def close(self):
        """Close database connections."""
        self.dag_store.conn.close()


# ─── Quick Start Helper ────────────────────────────────────────────

def init(project_dir: str = ".") -> AgentGraph:
    """
    Quick initialization helper.
    
    Usage:
        from core import init
        
        ag = init()
        ag.create_branch("main")
        
        # Get callback for LangGraph
        callback = ag.get_callback()
        
        # Use with your LangGraph app
        app.invoke({"input": "Hello"}, config={"callbacks": [callback]})
        
        # Check recorded nodes
        for node in ag.get_branch_nodes():
            print(ag.peek(int(node.id)))
    """
    return AgentGraph(project_dir)
