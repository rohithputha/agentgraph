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
# from langgraph_callback import langgraph_callback  <-- moved inside get_callback
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
        self.checkpoint_store = CheckpointStore(self.agit_path, self.project_dir, self.dag_store)
        
        # Initialize tracer (subscribes to eventbus and records nodes)
        self._tracer = Tracer(self.dag_store)
        self._tracer.eventbus = self.eventbus
        self.eventbus.subscribe_all(self._tracer.handle_event)
        
        # Stateless! No current_node_id, current_branch_id, etc.

    # ─── LangGraph Integration ─────────────────────────────────────

    def get_callback(self) -> 'langgraph_callback':
        """
        Get the LangGraph callback handler for automatic tracking.
        
        Use this callback with any LangGraph/LangChain invocation:
        
            callback = ag.get_callback()
            app.invoke(input, config={"callbacks": [callback]})
        """
        from langgraph_callback import langgraph_callback
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

    def emit_user_input(self, user_id: str, session_id: str, message: str, metadata: Optional[dict] = None):
        """Convenience method to emit a user input event."""
        self.eventbus.publish(
            EventType.USER_INPUT,
            Event(
                type=EventType.USER_INPUT,
                user_id=user_id,
                session_id=session_id,
                content=message,
                metadata=metadata or {},
                timestamp=datetime.now(),
            )
        )

    # ─── Branch Operations ─────────────────────────────────────────

    def create_branch(
        self,
        user_id: str,
        session_id: str,
        name: str,
        intent: str = "",
        base_node_id: Optional[int] = None,
    ) -> int:
        """Create a new branch for a session. Returns branch ID."""
        branch = Branch(
            user_id=user_id,
            session_id=session_id,
            name=name,
            head_node_id=str(base_node_id) if base_node_id else "0",
            base_node_id=str(base_node_id) if base_node_id else "0",
            status=BranchStatus.ACTIVE,
            intent=intent,
            created_by=CallerType.SYSTEM,
            created_at=datetime.now(),
        )
        branch_id = self.dag_store.insert_branch(user_id, session_id, branch)
        return branch_id

    # switch_branch is removed as it's a stateful operation. 
    # Frontend/Client should manage which branch is active by name.
    
    def list_branches(self, user_id: str, session_id: str, status: Optional[BranchStatus] = None) -> List[Branch]:
        """List all branches for a session, optionally filtered by status."""
        return self.dag_store.list_branches(user_id, session_id, status)

    # list_branches duplicate removed

    # ─── Node Operations ───────────────────────────────────────────

    def peek(self, user_id: str, session_id: str, node_id: int) -> Optional[dict]:
        """Peek at the memory (content) for a given node number."""
        return self.dag_store.peek(user_id, session_id, node_id)

    def get_node(self, user_id: str, session_id: str, node_id: int) -> Optional[ExecutionNode]:
        """Get full node details by ID."""
        return self.dag_store.get_node(user_id, session_id, node_id)

    def get_history(self, user_id: str, session_id: str, node_id: int) -> List[ExecutionNode]:
        """Get the path from root to a given node."""
        return self.dag_store.get_path_to_root(user_id, session_id, node_id)

    def get_branch_nodes(self, user_id: str, session_id: str, branch_id: int) -> List[ExecutionNode]:
        """Get all nodes in a branch."""
        return self.dag_store.get_branch_nodes(user_id, session_id, branch_id)

    # ─── Checkpoint Operations ─────────────────────────────────────

    def checkpoint(
        self,
        user_id: str,
        session_id: str,
        name: str,
        agent_memory: dict,
        conversation_history: list,
        label: str = "Checkpoint",
    ) -> Checkpoint:
        """Create a checkpoint of current state."""
        # 1. Create checkpoint logic (git commit, etc.)
        checkpoint = self.checkpoint_store.create_checkpoint(
            user_id, session_id, name, agent_memory, conversation_history, label
        )

        # 2. Record checkpoint as a Node in the DAG
        # We need to find the parent node for this session to link it in the DAG
        # Similar logic to Tracer._create_node
        branch = self.dag_store.get_active_branch(user_id, session_id)
        if branch:
           parent_id = branch.head_node_id
           node = ExecutionNode(
                user_id=user_id,
                session_id=session_id,
                id="0", # Auto-generated
                parent_id=parent_id,
                checkpoint_sha=checkpoint.filesystem_ref, # Store git SHA!
                action_type=ActionType.CHECKPOINT,
                content={"label": label},
                triggered_by=CallerType.SYSTEM,
                caller_context={},
                state_hash=checkpoint.hash,
                timestamp=datetime.now(),
                duration_ms=0,
                token_count=0
           )
           new_id = self.dag_store.insert_node(user_id, session_id, node, branch.branch_id)
           # Update branch head!
           self.dag_store.update_branch_head(user_id, session_id, branch.branch_id, new_id)

           # 3. Persist checkpoint metadata to database
           self.dag_store.insert_checkpoint(checkpoint, new_id)

        return checkpoint

    def restore(self, checkpoint: Checkpoint):
        """Restore from a checkpoint."""
        self.checkpoint_store.restore_checkpoint(checkpoint)

    # ─── Properties ────────────────────────────────────────────────

    @property
    def store(self) -> DagStore:
        """Access to underlying DAG store (for Tracer compatibility)."""
        return self.dag_store

    def get_active_branch(self, user_id: str, session_id: str) -> Optional[Branch]:
        """Get the current active branch for a session."""
        return self.dag_store.get_active_branch(user_id, session_id)

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
        # Session context is required for operations
        user_id = "default"
        session_id = "test-session"
        
        ag.create_branch(user_id, session_id, "main")
        
        # Get callback for LangGraph (handles session context from config via configurable)
        callback = ag.get_callback()
    """
    return AgentGraph(project_dir)
