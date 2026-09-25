"""PII masking for the LLM summarization request.

By default, recording, local Whisper and pyannote all stay on-machine;
`summarizer.summarize()` is the only place a *transcript* crosses to a third party, and
its fallback chain can hand the same text to a second provider when the first fails.
When `TAPEBACK_MASK_PII` is on, this module replaces structured PII with `[LABEL_N]`
placeholders before the text is sent and restores the real values in the parsed result,
so the vault keeps what was actually said while no provider ever sees it.

A second outbound path exists when `TAPEBACK_STT_BACKEND` is a remote backend
(today: `openai`): meeting *audio* is uploaded for speech-to-text. Masking cannot
help there — the raw audio leaves as-is. Prefer the local backend if that is
unacceptable.

Off by default. With masking disabled every method is the identity function — the request
is byte-identical to what it would have been without this module.

Coverage: deterministic regex for email and phone, plus literal terms the user lists in
`TAPEBACK_MASK_TERMS` — which is what actually matters here, since what people say aloud
in a meeting is names, not addresses. Term matching is literal by design: a term is
masked in the exact forms listed and nothing else. Inferring inflected forms needs
morphology, and guessing them would silently rewrite unrelated words. `_RULES` is the
seam for adding detectors.
"""

import functools
import re
import sys

from tapeback import const

type MaskMap = dict[str, str]  # placeholder -> original value

# Email. The local part allows the usual symbols; the host must end in a TLD.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone: Russian (+7 / 8 + 10 digits) and generic international (+<country>...),
# tolerant of spaces, dashes and parentheses. Anchored on a leading +/8 with
# word-boundary look-around so it does not grab bare digit runs (years, ids).
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?![\w])"
    r"|(?<![\w.])\+\d{1,3}[\s\-]?\(?\d{2,4}\)?(?:[\s\-]?\d{2,4}){2,4}(?![\w])"
)
# Email first: its host would otherwise be a candidate for phone matching.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (("EMAIL", _EMAIL_RE), ("PHONE", _PHONE_RE))

_TERM_LABEL = "TERM"
_LABELS: tuple[str, ...] = (*(label for label, _ in _RULES), _TERM_LABEL)

_PLACEHOLDER_RE = re.compile(r"\[(?:" + "|".join(_LABELS) + r")_\d+\]")

# A mask term must not collide with a placeholder label, which the term rule would then
# chew a hole in, nor with a speaker label, which the system prompt tells the model to
# reuse verbatim. Neither is personal data, so dropping such a term protects nothing.
# IGNORECASE because term matching is case-insensitive: "speaker 1" in the list would
# otherwise slip past the guard and go on to mask the real "Speaker 1" labels.
_SPEAKER_LABEL_RE = re.compile(
    re.escape(const.SPEAKER_LABEL_FMT).replace(r"\{\}", r"\d+"), re.IGNORECASE
)
_RESERVED_TERMS = frozenset(
    {const.SPEAKER_YOU.lower(), const.SPEAKER_OTHER.lower(), *(label.lower() for label in _LABELS)}
)


def _parse_terms(raw: str) -> list[str]:
    """Split the comma-separated setting into usable terms, longest first.

    Longest first is what makes a multi-word term win over its own first word once the
    terms become one alternation.
    """
    terms: list[str] = []
    for chunk in raw.split(","):
        term = chunk.strip()
        if not term:
            continue
        if term.lower() in _RESERVED_TERMS or _SPEAKER_LABEL_RE.fullmatch(term):
            print(
                f"Warning: ignoring mask term {term!r} — it collides with a transcript label.",
                file=sys.stderr,
            )
            continue
        terms.append(term)
    # dict.fromkeys deduplicates while keeping the configured order, so the sort below
    # (stable) stays deterministic between runs.
    unique = list(dict.fromkeys(terms))
    unique.sort(key=len, reverse=True)
    return unique


def _term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    """One alternation for all terms, or None when there are none.

    One pattern rather than a pass per term: a single leftmost-longest scan cannot match
    inside a placeholder an earlier term just inserted.
    """
    if not terms:
        return None
    alternation = "|".join(re.escape(term) for term in terms)
    # Look-arounds rather than \b: a term may end in punctuation ("Acme Inc."), where \b
    # would demand a word character exactly where there is none.
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


class Masker:
    """Per-call PII map. Mask the transcript, send it, then unmask each field of the
    parsed response with the same map. One value gets ONE placeholder for the whole call,
    so the model sees a consistent entity and the map stays small.

    Construct one per `summarize` call and discard it — the map holds plaintext PII and
    must never outlive the call or be written anywhere.
    """

    def __init__(self, *, enabled: bool, terms: str = "") -> None:
        self._enabled = enabled
        self._to_placeholder: dict[str, str] = {}  # original -> placeholder
        self._to_original: MaskMap = {}  # placeholder -> original
        self._counts: dict[str, int] = {}  # per-label running index
        self._rules = _RULES
        if enabled:
            # Parsed only when enabled, so a stale term list warns nobody while the
            # feature is off. Terms run last: earlier rules have already consumed the
            # addresses a term could otherwise cut in half.
            pattern = _term_pattern(_parse_terms(terms))
            if pattern is not None:
                self._rules = (*_RULES, (_TERM_LABEL, pattern))

    def mask(self, text: str) -> str:
        """Replace PII with placeholders. Identity when masking is disabled."""
        if not self._enabled or not text:
            return text
        out = text
        for label, pattern in self._rules:
            out = pattern.sub(functools.partial(self._sub, label), out)
        return out

    def _sub(self, label: str, match: re.Match[str]) -> str:
        value = match.group(0)
        existing = self._to_placeholder.get(value)
        if existing is not None:
            return existing  # same value -> same placeholder (consistent + small map)
        self._counts[label] = self._counts.get(label, 0) + 1
        placeholder = f"[{label}_{self._counts[label]}]"
        self._to_placeholder[value] = placeholder
        self._to_original[placeholder] = value
        return placeholder

    @property
    def mapping(self) -> MaskMap:
        return dict(self._to_original)

    @property
    def counts(self) -> dict[str, int]:
        """Distinct values replaced, per label — how much was masked, without disclosing
        what. Occurrences are not counted: a repeated value reuses its placeholder.
        """
        return dict(self._counts)

    def unmask(self, text: str) -> str:
        """Restore the originals this masker replaced."""
        if not text or not self._to_original:
            return text
        out = text
        # Longest placeholder first: the closing bracket already prevents [X_1] from
        # matching inside [X_11], but be explicit and order-independent.
        for placeholder, original in sorted(
            self._to_original.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            out = out.replace(placeholder, original)
        return out

    def residual_placeholders(self, text: str) -> list[str]:
        """Placeholder-shaped fragments left after unmasking — the model invented an
        index or reformatted one, so the real value could not be put back. Empty when
        this masker replaced nothing, since then any such fragment came from the model.
        """
        if not self._to_original:
            return []
        return _PLACEHOLDER_RE.findall(text)
