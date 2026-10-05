"""System tray icon for tapeback — pure-Python SNI implementation."""

import asyncio
import logging
import threading
from collections.abc import Callable
from enum import Enum, auto

from dbus_next import BusType
from dbus_next.aio import MessageBus

from tapeback._dbusmenu import MENU_OBJECT_PATH, DBusMenu, MenuItem
from tapeback._sni import SNI_OBJECT_PATH, StatusNotifierItem, register_with_watcher
from tapeback._tray_env import TrayEnv, detect_tray_env
from tapeback.pipeline import stop_and_process
from tapeback.recorder import Recorder, detect_devices, session_root
from tapeback.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class TrayState(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


# Freedesktop-standard theme icon names — present on every modern icon theme.
_STATE_ICONS: dict[TrayState, str] = {
    TrayState.IDLE: "audio-input-microphone-symbolic",
    TrayState.RECORDING: "media-record-symbolic",
    TrayState.PROCESSING: "process-working-symbolic",
}

# Stable menu IDs referenced by DBusMenu.Event handler.
_ID_START = 1
_ID_STOP = 2
_ID_PROCESSING = 3
_ID_SEP = 4
_ID_STATUS = 5
_ID_QUIT = 6


def build_menu_items(
    state: TrayState,
    on_start: Callable[[], None] | None = None,
    on_stop: Callable[[], None] | None = None,
    on_status: Callable[[], None] | None = None,
    on_quit: Callable[[], None] | None = None,
) -> list[MenuItem]:
    """Pure layout for current state — extracted so it can be unit-tested."""
    return [
        MenuItem(
            id=_ID_START,
            label="Start Recording",
            visible=(state == TrayState.IDLE),
            on_clicked=on_start,
        ),
        MenuItem(
            id=_ID_STOP,
            label="Stop Recording",
            visible=(state == TrayState.RECORDING),
            on_clicked=on_stop,
        ),
        MenuItem(
            id=_ID_PROCESSING,
            label="Processing...",
            enabled=False,
            visible=(state == TrayState.PROCESSING),
        ),
        MenuItem(id=_ID_SEP, type="separator"),
        MenuItem(id=_ID_STATUS, label="Status", on_clicked=on_status),
        MenuItem(id=_ID_QUIT, label="Quit", on_clicked=on_quit),
    ]


class TrayApp:
    """SNI-driven tray app — replaces the old pystray implementation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._recorder = Recorder()
        self._state = TrayState.IDLE
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._sni: StatusNotifierItem | None = None
        self._menu: DBusMenu | None = None
        self._bus: MessageBus | None = None

    def run(self) -> None:
        """Blocking entry point — runs the asyncio loop until Quit fires."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()

        if self._recorder.is_recording():
            self._state = TrayState.RECORDING

        self._menu = DBusMenu(self._current_menu_items())
        self._sni = StatusNotifierItem(
            app_id="tapeback",
            title="tapeback",
            menu_path=MENU_OBJECT_PATH,
        )
        self._sni.set_icon(_STATE_ICONS[self._state])
        self._sni.set_tooltip("tapeback", self._tooltip_text())
        self._sni.set_callbacks(on_activate=self._on_activate)

        self._bus = await MessageBus(bus_type=BusType.SESSION).connect()
        self._bus.export(SNI_OBJECT_PATH, self._sni)
        self._bus.export(MENU_OBJECT_PATH, self._menu)

        unique_name = self._bus.unique_name
        if unique_name is None:
            # connect() should always populate this; if it didn't, we have no
            # name to register and the tray is unreachable anyway.
            raise RuntimeError("D-Bus connect returned without a unique name")
        try:
            await register_with_watcher(self._bus, unique_name)
            logger.info("Tray registered with StatusNotifierWatcher")
        except Exception:
            logger.exception("Failed to register with StatusNotifierWatcher")
            logger.warning(
                "Tray will not appear. On GNOME install "
                "gnome-shell-extension-appindicator and re-login."
            )

        await self._stopped.wait()
        self._bus.disconnect()

    # --- menu / state ---------------------------------------------------------

    def _current_menu_items(self) -> list[MenuItem]:
        return build_menu_items(
            self._state,
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_status=self._on_status,
            on_quit=self._on_quit,
        )

    def _tooltip_text(self) -> str:
        if self._state == TrayState.RECORDING:
            session = self._recorder.get_session_info()
            name = session["session_name"] if session else "unknown"
            return f"Recording ({name})"
        if self._state == TrayState.PROCESSING:
            return "Processing..."
        return "Idle"

    def _update_state(self, new_state: TrayState) -> None:
        """Called from the loop thread — updates SNI + menu in place."""
        self._state = new_state
        if self._sni is not None:
            self._sni.set_icon(_STATE_ICONS[new_state])
            self._sni.set_tooltip("tapeback", self._tooltip_text())
        if self._menu is not None:
            self._menu.replace_items(self._current_menu_items())

    def _update_state_threadsafe(self, new_state: TrayState) -> None:
        """Schedule a state update from a worker thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._update_state, new_state)

    # --- menu callbacks (fire on asyncio loop thread) ------------------------

    def _on_activate(self) -> None:
        """Left-click on icon: shortcut for start/stop depending on state."""
        with self._lock:
            current = self._state
        if current == TrayState.IDLE:
            self._on_start()
        elif current == TrayState.RECORDING:
            self._on_stop()

    def _on_start(self) -> None:
        with self._lock:
            if self._state != TrayState.IDLE:
                return
            self._update_state(TrayState.RECORDING)
        threading.Thread(target=self._do_start, daemon=True).start()

    def _do_start(self) -> None:
        try:
            detect_devices(self._settings)
            session_name = self._recorder.start(self._settings)
            logger.info("Recording started: %s", session_name)
        except Exception:
            logger.exception("Failed to start recording")
            self._update_state_threadsafe(TrayState.IDLE)

    def _on_stop(self) -> None:
        with self._lock:
            if self._state != TrayState.RECORDING:
                return
            self._update_state(TrayState.PROCESSING)
        threading.Thread(target=self._do_stop_and_process, daemon=True).start()

    def _do_stop_and_process(self) -> None:
        try:
            md_path = stop_and_process(
                self._recorder,
                self._settings,
                on_status=lambda msg: logger.info(msg),
            )
            logger.info("Saved: %s", md_path)
        except Exception:
            logger.exception("Processing failed")
        finally:
            self._update_state_threadsafe(TrayState.IDLE)

    def _on_status(self) -> None:
        session = self._recorder.get_session_info()
        if session:
            logger.info(
                "Recording: %s (started %s)",
                session["session_name"],
                session["started_at"],
            )
        elif self._state == TrayState.PROCESSING:
            logger.info("Processing transcript...")
        else:
            logger.info("Idle — ready to record")

    def _on_quit(self) -> None:
        with self._lock:
            if self._state == TrayState.RECORDING:
                try:
                    self._recorder.stop()
                    logger.info(
                        "Recording stopped on quit, files preserved in %s/",
                        session_root(),
                    )
                except Exception:
                    logger.exception("Error stopping recording on quit")
        if self._stopped is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._stopped.set)


def _warn_if_tray_host_missing(env: TrayEnv) -> None:
    """Emit the AppIndicator hint exactly once via the logger.

    Extracted so that v0.9.5 doesn't double-print like v0.9.4 did (both
    `print(...)` and `logger.warning(...)` ended up in user logs).
    """
    if env.needs_appindicator_hint:
        logger.warning("%s", env.hint_message)


def run_tray() -> None:
    """Entry point for the `tapeback tray` command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    _warn_if_tray_host_missing(detect_tray_env())
    TrayApp(settings).run()
