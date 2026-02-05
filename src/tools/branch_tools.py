from typing import Optional
from datetime import datetime
from models.agit import Agit
from models.dag import Branch, BranchStatus, CallerType


class BranchTool:
    def __init__(self, agit: Agit):
        self.agit = agit
    

    def create_branch(self, name: str, from_node: Optional[int] = None, intent: str = "") -> int:
        """Fork a new branch from the current (or specified) node. Returns the new branch_id."""
        base = from_node or self.agit.current_node_id
        branch = Branch(
            name=name,
            thread_id=name,  # Use name as thread_id for backward compatibility
            head_node_id=str(base),  # Convert to string for model compatibility
            base_node_id=str(base),
            status=BranchStatus.ACTIVE,
            intent=intent,
            created_by=CallerType.HUMAN_CLI,
            created_at=datetime.now(),
        )
        # insert_branch now returns the auto-generated branch_id
        new_branch_id = self.agit.store.insert_branch(branch)
        self.agit.current_branch_id = new_branch_id
        return new_branch_id


    def switch_branch(self, name: str):
        """Switch to a different branch."""
        branch = self.agit.store.get_branch(name)
        if not branch:
            raise ValueError(f"Branch '{name}' not found")
        self.agit.current_branch_id = branch.branch_id
        self.agit.current_node_id = branch.head_node_id
    
    def list_branches(self):
        """List all branches."""
        return self.agit.store.list_branches()
    
    