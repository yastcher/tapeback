"""What may stand as a speaker's name in a note: one rule for the user's own name and a
voice profile's."""

import re
from collections.abc import Iterable

from tapeback import const

_NUMBERED_LABEL = re.compile(
    re.escape(const.SPEAKER_LABEL_FMT).replace(r"\{\}", r"\d+"), re.IGNORECASE
)


def speaker_name_problem(name: str, *, reserved: Iterable[str] = ()) -> str | None:
    """Why name cannot stand as a speaker in a note, or None when it can.

    tapeback's own labels are refused, "Other" and "Speaker N" always and `reserved`
    besides: a person under one of them would read as a speaker tapeback did not know.
    """
    if not name or name != name.strip():
        return "it is empty, or starts or ends with a space"
    if any(char in const.SPEAKER_NAME_FORBIDDEN for char in name):
        return "it holds one of * : [ ] , or a line break"
    own = {label.casefold() for label in (const.SPEAKER_OTHER, *reserved)}
    if name.casefold() in own or _NUMBERED_LABEL.fullmatch(name):
        return "it is one of tapeback's own labels"
    return None
