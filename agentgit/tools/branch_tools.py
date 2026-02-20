from typing import Optional, List
from datetime import datetime
from ..models.dag import Branch, BranchStatus, CallerType


class BranchTools:
    """
    Branch management tools for creating, switching, and listing branches.
    Provides session-aware branch operations on top of AgentGit.
    """

    def __init__(self, ag):
        """
        Initialize BranchTools with an AgentGit instance.

        Args:
            ag: AgentGit instance to operate on
        """
        self.ag = ag

    def create_branch(
        self,
        user_id: str,
        session_id: str,
        name: str,
        from_node: Optional[int] = None,
        intent: str = ""
    ) -> int:
        """
        Fork a new branch from the specified (or current active branch's head) node.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Name for the new branch
            from_node: Optional node ID to branch from. If None, uses active branch head.
            intent: Optional description of branch purpose

        Returns:
            The newly created branch_id
        """
        # If no base node specified, use the active branch's head
        base_node = from_node
        if base_node is None:
            active_branch = self.ag.get_active_branch(user_id, session_id)
            if active_branch:
                base_node = int(active_branch.head_node_id) if active_branch.head_node_id else None

        return self.ag.create_branch(
            user_id=user_id,
            session_id=session_id,
            name=name,
            intent=intent,
            base_node_id=base_node
        )

    def switch_branch(self, user_id: str, session_id: str, name: str) -> bool:
        """
        Switch to a different branch by marking it as active.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Name of the branch to switch to

        Returns:
            True if switch successful, False if branch not found
        """
        branch = self.ag.dag_store.get_branch(user_id, session_id, name)
        if not branch:
            return False

        # Mark current active branch as not active (if any)
        current_active = self.ag.get_active_branch(user_id, session_id)
        if current_active and current_active.branch_id != branch.branch_id:
            self.ag.dag_store.update_branch_status(
                user_id, session_id, current_active.branch_id, BranchStatus.COMPLETED
            )

        # Mark target branch as active
        self.ag.dag_store.update_branch_status(
            user_id, session_id, branch.branch_id, BranchStatus.ACTIVE
        )
        return True

    def list_branches(
        self,
        user_id: str,
        session_id: str,
        status: Optional[BranchStatus] = None
    ) -> List[Branch]:
        """
        List all branches for a session, optionally filtered by status.

        Args:
            user_id: User identifier
            session_id: Session identifier
            status: Optional status filter (ACTIVE, COMPLETED, ABANDONED, MERGED)

        Returns:
            List of Branch objects
        """
        return self.ag.list_branches(user_id, session_id, status)

    def get_branch(self, user_id: str, session_id: str, name: str) -> Optional[Branch]:
        """
        Get a specific branch by name.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Branch name

        Returns:
            Branch object if found, None otherwise
        """
        return self.ag.dag_store.get_branch(user_id, session_id, name)

    def get_active_branch(self, user_id: str, session_id: str) -> Optional[Branch]:
        """
        Get the currently active branch for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Active Branch object if found, None otherwise
        """
        return self.ag.get_active_branch(user_id, session_id)

    def abandon_branch(self, user_id: str, session_id: str, name: str, reason: str = "") -> bool:
        """
        Mark a branch as abandoned.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Branch name to abandon
            reason: Optional reason for abandonment

        Returns:
            True if successful, False if branch not found
        """
        branch = self.ag.dag_store.get_branch(user_id, session_id, name)
        if not branch:
            return False

        self.ag.dag_store.update_branch_status(
            user_id, session_id, branch.branch_id, BranchStatus.ABANDONED, reason
        )
        return True

    def complete_branch(self, user_id: str, session_id: str, name: str, reason: str = "") -> bool:
        """
        Mark a branch as completed.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Branch name to complete
            reason: Optional completion note

        Returns:
            True if successful, False if branch not found
        """
        branch = self.ag.dag_store.get_branch(user_id, session_id, name)
        if not branch:
            return False

        self.ag.dag_store.update_branch_status(
            user_id, session_id, branch.branch_id, BranchStatus.COMPLETED, reason
        )
        return True

    def get_branch_nodes(self, user_id: str, session_id: str, branch_name: str) -> List:
        """
        Get all execution nodes in a branch.

        Args:
            user_id: User identifier
            session_id: Session identifier
            branch_name: Name of the branch

        Returns:
            List of ExecutionNode objects in the branch
        """
        branch = self.ag.dag_store.get_branch(user_id, session_id, branch_name)
        if not branch:
            return []

        return self.ag.get_branch_nodes(user_id, session_id, branch.branch_id)

    def get_branch_stats(self, user_id: str, session_id: str, name: str) -> Optional[dict]:
        """
        Get statistics for a branch.

        Args:
            user_id: User identifier
            session_id: Session identifier
            name: Branch name

        Returns:
            Dictionary with branch statistics or None if branch not found
        """
        branch = self.ag.dag_store.get_branch(user_id, session_id, name)
        if not branch:
            return None

        nodes = self.ag.get_branch_nodes(user_id, session_id, branch.branch_id)

        return {
            "name": branch.name,
            "status": branch.status.value,
            "intent": branch.intent,
            "node_count": len(nodes),
            "tokens_used": branch.tokens_used,
            "time_elapsed_seconds": branch.time_elapsed_seconds,
            "created_at": branch.created_at,
            "head_node_id": branch.head_node_id,
            "base_node_id": branch.base_node_id,
        }