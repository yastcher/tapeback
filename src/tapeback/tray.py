"""System tray icon for tapeback — start/stop recording without a terminal."""

import logging
import threading
from enum import Enum, auto

import pystray
from PIL import Image, ImageDraw

from tapeback import const
from tapeback.pipeline import stop_and_process
from tapeback.recorder import Recorder, detect_devices
from tapeback.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_ICON_SIZE = 64


class TrayState(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


_STATE_COLORS: dict[TrayState, str] = {
    TrayState.IDLE: "#808080",
    TrayState.RECORDING: "#FF0000",
    TrayState.PROCESSING: "#FFA500",
}


def _create_icon(color: str) -> Image.Image:
    """Generate a circle icon with the given color."""
    image = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 4
    draw.ellipse(
        [margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin],
        fill=color,
    )
    return image


def _icon_for_state(state: TrayState) -> Image.Image:
    """Return the icon image for a given tray state."""
    return _create_icon(_STATE_COLORS[state])


class TrayApp:
    """System tray application for tapeback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._recorder = Recorder()
        self._state = TrayState.IDLE
        self._lock = threading.Lock()
        self._icon: pystray.Icon | None = None

    def run(self) -> None:
        """Create and run the system tray icon (blocking)."""
        if self._recorder.is_recording():
            self._state = TrayState.RECORDING

        self._icon = pystray.Icon(
            name="tapeback",
            icon=_icon_for_state(self._state),
            title=self._tooltip(),
            menu=pystray.Menu(
                pystray.MenuItem(
                    "Start Recording",
                    self._on_start,
                    visible=lambda _item: self._state == TrayState.IDLE,
                ),
                pystray.MenuItem(
                    "Stop Recording",
                    self._on_stop,
                    visible=lambda _item: self._state == TrayState.RECORDING,
                ),
                pystray.MenuItem(
                    "Processing...",
                    None,
                    enabled=False,
                    visible=lambda _item: self._state == TrayState.PROCESSING,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Status", self._on_status),
                pystray.MenuItem("Quit", self._on_quit),
            ),
        )
        self._icon.run()

    def _tooltip(self) -> str:
        """Return tooltip text based on current state."""
        if self._state == TrayState.RECORDING:
            session = self._recorder.get_session_info()
            name = session["session_name"] if session else "unknown"
            return f"tapeback: Recording ({name})"
        if self._state == TrayState.PROCESSING:
            return "tapeback: Processing..."
        return "tapeback: Idle"

    def _update_state(self, new_state: TrayState) -> None:
        """Update state, icon, and tooltip."""
        self._state = new_state
        if self._icon:
            self._icon.icon = _icon_for_state(new_state)
            self._icon.title = self._tooltip()

    def _notify(self, message: str) -> None:
        """Send desktop notification via pystray."""
        if self._icon:
            try:
                self._icon.notify(message, title="tapeback")
            except Exception:
                logger.warning("Notification failed: %s", message)

    # --- Menu callbacks ---

    def _on_start(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Handle 'Start Recording' click."""
        with self._lock:
            if self._state != TrayState.IDLE:
                return
            self._update_state(TrayState.RECORDING)
        thread = threading.Thread(target=self._do_start, daemon=True)
        thread.start()

    def _do_start(self) -> None:
        """Start recording in background thread."""
        try:
            detect_devices(self._settings)
            session_name = self._recorder.start(self._settings)
            logger.info("Recording started: %s", session_name)
            self._notify(f"Recording started: {session_name}")
        except Exception as exc:
            logger.exception("Failed to start recording")
            self._notify(f"Error: {exc}")
            with self._lock:
                self._update_state(TrayState.IDLE)

    def _on_stop(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Handle 'Stop Recording' click."""
        with self._lock:
            if self._state != TrayState.RECORDING:
                return
            self._update_state(TrayState.PROCESSING)
        thread = threading.Thread(target=self._do_stop_and_process, daemon=True)
        thread.start()

    def _do_stop_and_process(self) -> None:
        """Stop recording and process in background thread."""
        try:
            md_path = stop_and_process(
                self._recorder,
                self._settings,
                on_status=lambda msg: logger.info(msg),
            )
            logger.info("Saved: %s", md_path)
            self._notify(f"Saved: {md_path.name}")
        except Exception as exc:
            logger.exception("Processing failed")
            self._notify(f"Error: {exc}")
        finally:
            with self._lock:
                self._update_state(TrayState.IDLE)

    def _on_status(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Handle 'Status' click."""
        session = self._recorder.get_session_info()
        if session:
            self._notify(f"Recording: {session['session_name']}\nStarted: {session['started_at']}")
        elif self._state == TrayState.PROCESSING:
            self._notify("Processing transcript...")
        else:
            self._notify("Idle — ready to record")

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Handle 'Quit' click."""
        with self._lock:
            if self._state == TrayState.RECORDING:
                try:
                    self._recorder.stop()
                    self._notify(f"Recording stopped. Audio files saved in {const.TEMP_DIR}/")
                    logger.info("Recording stopped on quit, files preserved")
                except Exception:
                    logger.exception("Error stopping recording on quit")
        if self._icon:
            self._icon.stop()


def run_tray() -> None:
    """Entry point for the tray command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    app = TrayApp(settings)
    app.run()
