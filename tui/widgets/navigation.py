"""
Vertical navigation rail with high-level actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button


@dataclass(frozen=True)
class NavigationItem:
    id: str
    label: str


class NavigationSelection(Message):
    """Message emitted when the user selects a navigation item."""

    def __init__(self, item_id: str) -> None:
        super().__init__()
        self.item_id = item_id


class NavigationRail(Widget):
    """
    Simple vertical collection of buttons that behave like a navigation rail.
    """

    items: reactive[Sequence[NavigationItem]] = reactive(tuple())

    def __init__(self, items: Iterable[NavigationItem]) -> None:
        super().__init__()
        self.items = tuple(items)

    def compose(self):
        for item in self.items:
            yield Button(item.label, id=f"nav-{item.id}", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.id:
            return
        item_id = event.button.id.removeprefix("nav-")
        self.post_message(NavigationSelection(item_id))
