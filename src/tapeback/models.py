"""Domain models — data structures shared across modules."""

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str
    words: list[Word] | None = None
    speaker: str | None = None  # None in phase 1, filled in phase 2


@dataclass
class DiarizationSegment:
    """A continuous speech segment from one speaker (from pyannote)."""

    speaker: str  # "SPEAKER_00", "SPEAKER_01", ...
    start: float  # seconds
    end: float  # seconds


@dataclass
class ActionItem:
    assignee: str
    action: str
    deadline: str | None = None


@dataclass
class Summary:
    brief: str
    action_items: list[ActionItem] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    is_trivial: bool = False
