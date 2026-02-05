

class VersionTools:
    def __init__(self, agit: Agit):
        self.agit = agit
    
    def create_checkpoint(self, name: str, agent_memory: Optional[dict] = None):
        """Snapshot agent state. Returns the checkpoint hash."""
        self
        
