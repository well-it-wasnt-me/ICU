"""
Thread-safe runtime event bus bridging background workers with the Textual UI.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable, Optional, Type, TypeVar

from runtime_events import RuntimeEvent

EventT = TypeVar("EventT", bound=RuntimeEvent)


class RuntimeEventBus:
    """
    Lightweight event bus that stores events in a queue and notifies registered listeners.

    The queue allows the UI to poll for new events, while listeners can react immediately
    to specific event types (e.g., for logging or metrics aggregation).
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[RuntimeEvent]" = queue.Queue()
        self._listeners: dict[Type[RuntimeEvent], list[Callable[[RuntimeEvent], None]]] = {}
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        self._queue.put(event)
        listeners = self._listeners.get(type(event), ())
        for listener in list(listeners):
            try:
                listener(event)
            except Exception:
                # Listener failures should not propagate to producers.
                continue

    def subscribe(self, event_type: Type[EventT], listener: Callable[[EventT], None]) -> None:
        with self._lock:
            listeners = self._listeners.setdefault(event_type, [])
            listeners.append(listener)  # type: ignore[arg-type]

    def unsubscribe(self, event_type: Type[EventT], listener: Callable[[EventT], None]) -> None:
        with self._lock:
            listeners = self._listeners.get(event_type)
            if not listeners:
                return
            try:
                listeners.remove(listener)  # type: ignore[arg-type]
            except ValueError:
                pass
            if not listeners:
                self._listeners.pop(event_type, None)

    def poll(self, timeout: Optional[float] = None) -> Optional[RuntimeEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> Iterable[RuntimeEvent]:
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                break
