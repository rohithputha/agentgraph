

class eventbus:
    def __init__(self):
        self._subscribers = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type, event):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(event)