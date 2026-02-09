from typing import Callable

from .event import EventType, Event
class Eventbus:
    def __init__(self):
        self._subscribers = {}
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: EventType, event: Event):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(event)

    def subscribe_all(self, callback: Callable[[Event], None]):
        for event_type in EventType:
            self.subscribe(event_type, callback)