"""Regression (issue #15): recordings were named in UTC.

The session name becomes the note's file name and its `date` and `time`, so a meeting
at 08:30 in Tokyo was filed as 23:30 on the previous day. Names now follow
TAPEBACK_TIMEZONE, or the machine's own zone when it is unset.
"""

import datetime
import json

import pytest
from pydantic import ValidationError

from tapeback.recorder import Recorder
from tapeback.settings import Settings

# 23:30 UTC: the UTC date and the local date differ, which is the bug's whole effect.
_INSTANT = datetime.datetime(2026, 10, 5, 23, 30, tzinfo=datetime.UTC)


@pytest.mark.parametrize(
    ("zone", "name", "started_at"),
    [
        (None, "2026-10-06_08-30-00", "2026-10-06T08:30:00+09:00"),
        ("", "2026-10-06_08-30-00", "2026-10-06T08:30:00+09:00"),
        # Summer time is still on in Madrid in early October: +02:00, not +01:00.
        ("Europe/Madrid", "2026-10-06_01-30-00", "2026-10-06T01:30:00+02:00"),
        ("UTC", "2026-10-05_23-30-00", "2026-10-05T23:30:00+00:00"),
    ],
    ids=["machine zone", "empty means machine zone", "setting wins", "UTC on request"],
)
def test_session_is_named_in_the_meetings_local_time(
    zone, name, started_at, tmp_path, fake_parecord, system_timezone
):
    system_timezone("Asia/Tokyo")
    settings = Settings(vault_path=tmp_path / "vault", timezone=zone)
    recorder = Recorder(state_dir=tmp_path / "state", clock=lambda: _INSTANT)

    try:
        assert recorder.start(settings) == name
        assert json.loads(recorder.session_file.read_text())["started_at"] == started_at
    finally:
        recorder.stop()


def test_an_unknown_time_zone_is_refused_when_settings_load(tmp_path):
    """At load, with the fix in the message — not at the first recording."""
    with pytest.raises(ValidationError, match=r"unknown time zone 'Mars/Olympus'.*Europe/Madrid"):
        Settings(vault_path=tmp_path, timezone="Mars/Olympus")
