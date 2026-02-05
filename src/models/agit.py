from storage.dag_store import DagStore
from eventbus import Eventbus
class Agit:
    def __init__(self):
        self.eventbus = Eventbus()
        self.store = DagStore()
        self.current_node_id = -1 
        self.current_branch_id = -1


    
