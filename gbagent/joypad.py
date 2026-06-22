"""Shared joypad state — intermediary between WebSocket input and environment."""

from __future__ import annotations

from collections.abc import Callable

# Game Boy button names
BUTTON_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]


class JoypadState:
    """Thread-safe joypad state shared between the WebSocket server and the env.

    Exposes a dict-like interface mapping button names to boolean pressed states.
    All mutating methods are compatible with single-writer GIL semantics.
    """

    def __init__(self) -> None:
        self._state: dict[str, bool] = {name: False for name in BUTTON_NAMES}
        self._listeners: list[Callable[[dict[str, bool]], None]] = []

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def is_pressed(self, button: str) -> bool:
        """Return ``True`` if *button* is currently pressed."""
        return self._state.get(button, False)

    @property
    def state(self) -> dict[str, bool]:
        """Return a copy of the current button state."""
        return dict(self._state)

    @property
    def active_buttons(self) -> list[str]:
        """Return names of all currently-pressed buttons."""
        return [name for name, pressed in self._state.items() if pressed]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def press(self, button: str) -> None:
        """Mark a button as pressed."""
        if button in self._state and not self._state[button]:
            self._state[button] = True
            self._notify()

    def release(self, button: str) -> None:
        """Mark a button as released."""
        if button in self._state and self._state[button]:
            self._state[button] = False
            self._notify()

    def reset(self) -> None:
        """Release all buttons."""
        had_any = any(self._state.values())
        for name in BUTTON_NAMES:
            self._state[name] = False
        if had_any:
            self._notify()

    def apply(self, state: dict[str, bool]) -> None:
        """Replace the full state from a dict (e.g. from WS JSON)."""
        changed = False
        for name in BUTTON_NAMES:
            new_val = state.get(name, False)
            if self._state[name] != new_val:
                self._state[name] = new_val
                changed = True
        if changed:
            self._notify()

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def on_change(self, callback: Callable[[dict[str, bool]], None]) -> None:
        """Register a callback invoked whenever the button state changes."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        snapshot = dict(self._state)
        for cb in self._listeners:
            cb(snapshot)

    # ------------------------------------------------------------------
    # Python convenience
    # ------------------------------------------------------------------

    def __getitem__(self, button: str) -> bool:
        return self.is_pressed(button)

    def __repr__(self) -> str:
        active = ", ".join(self.active_buttons) or "none"
        return f"<JoypadState [{active}]>"
