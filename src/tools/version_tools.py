from typing import Optional, List
from models.dag import Checkpoint


class VersionTools:
    """
    Version control tools for creating and managing checkpoints.
    Provides session-aware checkpoint operations on top of AgentGraph.
    """

    def __init__(self, ag):
        """
        Initialize VersionTools with an AgentGraph instance.

        Args:
            ag: AgentGraph instance to operate on
        """
        self.ag = ag

    def create_checkpoint(
        self,
        user_id: str,
        session_id: str,
        name: str,
        agent_memory: Optional[dict] = None,
        conversation_history: Optional[list] = None,
        label: str = "Checkpoint"
    ) -> str:
        """
        Create a checkpoint snapshot of the current agent state and workspace.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Name for the checkpoint
            agent_memory: Optional agent memory/state to snapshot
            conversation_history: Optional conversation history to include
            label: Optional descriptive label for the checkpoint

        Returns:
            The checkpoint hash (unique identifier)
        """
        checkpoint = self.ag.checkpoint(
            user_id=user_id,
            session_id=session_id,
            name=name,
            agent_memory=agent_memory or {},
            conversation_history=conversation_history or [],
            label=label
        )
        return checkpoint.hash

    def restore_checkpoint(
        self,
        user_id: str,
        session_id: str,
        checkpoint_hash: str
    ) -> bool:
        """
        Restore workspace and state from a checkpoint.

        Args:
            user_id: User identifier
            session_id: Session identifier
            checkpoint_hash: Hash of the checkpoint to restore

        Returns:
            True if restore successful, False otherwise
        """
        # Get checkpoint from store
        checkpoint_row = self.ag.dag_store.get_checkpoint(checkpoint_hash)
        if not checkpoint_row:
            return False

        # Reconstruct Checkpoint object
        checkpoint = Checkpoint(
            hash=checkpoint_row[0],
            agent_memory={},  # Stored separately if needed
            conversation_history=[],  # Stored separately if needed
            filesystem_ref=checkpoint_row[2],
            files_changed=[],
            created_at=checkpoint_row[4],
            compressed=bool(checkpoint_row[5]),
            size_bytes=checkpoint_row[6],
            label=""
        )

        self.ag.restore(checkpoint)
        return True

    def list_checkpoints(self, user_id: str, session_id: str) -> List[dict]:
        """
        List all checkpoints for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            List of checkpoint info dictionaries
        """
        nodes = self.ag.dag_store.get_checkpoint_nodes(user_id, session_id)
        checkpoints = []

        for node in nodes:
            if node.checkpoint_sha:
                checkpoints.append({
                    "node_id": node.id,
                    "sha": node.checkpoint_sha,
                    "label": node.content.get("label", ""),
                    "timestamp": node.timestamp,
                })

        return checkpoints

    def get_checkpoint(self, checkpoint_hash: str) -> Optional[dict]:
        """
        Get checkpoint details by hash.

        Args:
            checkpoint_hash: Hash of the checkpoint

        Returns:
            Dictionary with checkpoint details or None if not found
        """
        checkpoint_row = self.ag.dag_store.get_checkpoint(checkpoint_hash)
        if not checkpoint_row:
            return None

        return {
            "hash": checkpoint_row[0],
            "node_id": checkpoint_row[1],
            "filesystem_ref": checkpoint_row[2],
            "files_changed": checkpoint_row[3],
            "created_at": checkpoint_row[4],
            "compressed": bool(checkpoint_row[5]),
            "size_bytes": checkpoint_row[6],
        }

    def get_checkpoint_at_node(self, user_id: str, session_id: str, node_id: int) -> Optional[str]:
        """
        Get the checkpoint SHA associated with a specific node.

        Args:
            user_id: User identifier
            session_id: Session identifier
            node_id: Node ID to check

        Returns:
            Checkpoint SHA if node has one, None otherwise
        """
        node = self.ag.get_node(user_id, session_id, node_id)
        if node and node.checkpoint_sha:
            return node.checkpoint_sha
        return None

    def compare_checkpoints(
        self,
        checkpoint_hash_1: str,
        checkpoint_hash_2: str
    ) -> Optional[dict]:
        """
        Compare two checkpoints to see what changed.

        Args:
            checkpoint_hash_1: First checkpoint hash
            checkpoint_hash_2: Second checkpoint hash

        Returns:
            Dictionary with comparison details or None if checkpoints not found
        """
        cp1 = self.get_checkpoint(checkpoint_hash_1)
        cp2 = self.get_checkpoint(checkpoint_hash_2)

        if not cp1 or not cp2:
            return None

        return {
            "checkpoint_1": checkpoint_hash_1,
            "checkpoint_2": checkpoint_hash_2,
            "size_diff_bytes": cp2["size_bytes"] - cp1["size_bytes"],
            "time_diff_seconds": (cp2["created_at"] - cp1["created_at"]),
            "files_1": cp1["files_changed"],
            "files_2": cp2["files_changed"],
        }

    def get_latest_checkpoint(self, user_id: str, session_id: str) -> Optional[dict]:
        """
        Get the most recent checkpoint for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Dictionary with latest checkpoint info or None if no checkpoints
        """
        checkpoints = self.list_checkpoints(user_id, session_id)
        if not checkpoints:
            return None

        # Return the most recent (first in list as it's ordered by timestamp DESC)
        return checkpoints[0]

    def restore_to_node(self, user_id: str, session_id: str, node_id: int) -> bool:
        """
        Restore to the checkpoint at a specific node.

        Args:
            user_id: User identifier
            session_id: Session identifier
            node_id: Node ID with checkpoint to restore

        Returns:
            True if restore successful, False otherwise
        """
        checkpoint_sha = self.get_checkpoint_at_node(user_id, session_id, node_id)
        if not checkpoint_sha:
            return False

        # Get the checkpoint details
        checkpoint_row = self.ag.dag_store.get_checkpoint(checkpoint_sha)
        if not checkpoint_row:
            return False

        # Reconstruct and restore
        checkpoint = Checkpoint(
            hash=checkpoint_row[0],
            agent_memory={},
            conversation_history=[],
            filesystem_ref=checkpoint_row[2],
            files_changed=[],
            created_at=checkpoint_row[4],
            compressed=bool(checkpoint_row[5]),
            size_bytes=checkpoint_row[6],
            label=""
        )

        self.ag.checkpoint_store.restore_checkpoint(checkpoint, user_id, session_id)
        return True