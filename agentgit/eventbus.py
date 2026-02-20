from typing import Callable, Optional
import sqlite3
import logging

from .event import EventType, Event

logger = logging.getLogger(__name__)


class Eventbus:
    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize eventbus with optional database connection.

        Args:
            conn: SQLite connection for transactional event processing.
                  If provided, all callbacks for an event run in a single transaction.
                  If None, callbacks run without transaction wrapper.
        """
        self._subscribers = {}
        self.conn = conn

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: EventType, event: Event):
        """
        Publish event to all subscribers with atomic transaction.

        All callbacks run in a single transaction:
        - If all succeed → commit all changes
        - If any fails → rollback all changes

        This ensures all-or-nothing atomicity across all event handlers.
        """
        if event_type not in self._subscribers:
            return

        # No connection - just run callbacks without transaction
        if not self.conn:
            for callback in self._subscribers[event_type]:
                callback(event)
            return

        # With connection - atomic transaction (all-or-nothing)
        try:
            # Execute all callbacks
            for callback in self._subscribers[event_type]:
                callback(event)

            # All succeeded - commit transaction
            self.conn.commit()

        except Exception as e:
            # Any failed - rollback everything
            logger.error(
                f"Event {event_type.value} processing failed, rolling back transaction: {e}",
                exc_info=True
            )
            self.conn.rollback()
            raise

    def subscribe_all(self, callback: Callable[[Event], None]):
        for event_type in EventType:
            self.subscribe(event_type, callback)