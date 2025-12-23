import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime_events import DetectionEvent, RuntimeEvent
from tui.event_bus import RuntimeEventBus


def test_event_bus_emit_and_poll():
    bus = RuntimeEventBus()
    event = DetectionEvent(camera_name="Cam", person_name="Alice", confidence=88.0)

    bus.emit(event)
    polled = bus.poll(timeout=0.01)

    assert polled is event
    assert list(bus.drain()) == []


def test_event_bus_subscribe_and_unsubscribe():
    bus = RuntimeEventBus()
    received = []

    def listener(evt: RuntimeEvent) -> None:
        received.append(evt)

    bus.subscribe(DetectionEvent, listener)
    event = DetectionEvent(camera_name="Cam")
    bus.emit(event)

    assert received == [event]

    bus.unsubscribe(DetectionEvent, listener)
    bus.emit(DetectionEvent(camera_name="Cam2"))

    assert received == [event], "Listener should not be called after unsubscribe."
