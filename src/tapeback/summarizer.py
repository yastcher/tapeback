"""Meeting summarization via LLM — extract summary, action items, key decisions."""

import json
import re
from pathlib import Path

import click

from tapeback import _llm
from tapeback._mask import Masker
from tapeback.models import ActionItem, Summary
from tapeback.settings import Settings

_SYSTEM_PROMPT = """\
You analyze meeting transcripts and extract structured information.

You MUST respond with valid JSON only. No markdown, no explanation, no preamble.

Each transcript line is a timecode, a speaker label and what was said. "You" is the \
person who recorded the meeting. The others are "Speaker 1", "Speaker 2" and so on, \
or "Other" when their voices were not told apart. These labels come from audio \
analysis and may be wrong where voices overlap.

JSON schema:
{
  "brief": "2-3 sentence summary of the meeting",
  "action_items": [
    {
      "assignee": "exact speaker label from transcript (e.g. 'You', 'Speaker 1')",
      "action": "what was promised or assigned",
      "deadline": "deadline if mentioned, null otherwise"
    }
  ],
  "key_decisions": ["decision 1", "decision 2"],
  "is_trivial": false
}

Rules:
- Use EXACT speaker labels from the transcript (e.g. "You", "Speaker 1")
- action_items: only explicit commitments. "I'll send the report by Friday" = action item. \
"We should think about it" = NOT an action item.
- key_decisions: only decisions that were agreed upon, not suggestions or open questions
- is_trivial: true ONLY if the meeting had no substantive content \
(small talk, tech issues, scheduling only)
- If is_trivial is true, brief should say so. action_items and key_decisions should be empty.
- Write in the same language as the transcript
- Keep brief concise — 2-3 sentences max, no filler"""

_RETRY_PROMPT = "Respond with valid JSON only. No other text."

# "The same language as the transcript" was not enough: a 4B local model followed the
# language of these English instructions and summarised a Russian meeting in English.
# The note knows its language (Whisper's code), so the prompt names it.
_LANGUAGE_RULE = (
    "- The transcript is in {language} (ISO 639-1): write the brief, the actions and the "
    "decisions in that language, whatever the language of these instructions"
)
_FRONTMATTER = re.compile(r"\A---\n(.*?)^---\n", re.MULTILINE | re.DOTALL)
# A language code as Whisper writes it; anything else in the field is not passed on.
_LANGUAGE_CODE = re.compile(r"^language: ([a-z]{2,3})$", re.MULTILINE)

# A diarized note used to print the raw You/Other transcript and then the diarized
# copy of the same text. Only the copy is sent: the raw half doubled the input.
_RAW_TRANSCRIPT = re.compile(
    r"^## Transcript\n.*?^## Diarized Transcript\n+", re.MULTILINE | re.DOTALL
)


def _strip_markdown_fences(raw: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM response."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = stripped.index("\n")
        stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _parse_response(raw: str) -> Summary:
    """Parse JSON response into Summary dataclass."""
    data = json.loads(_strip_markdown_fences(raw))
    action_items = [
        ActionItem(
            assignee=item["assignee"],
            action=item["action"],
            deadline=item.get("deadline"),
        )
        for item in data.get("action_items", [])
    ]
    return Summary(
        brief=data["brief"],
        action_items=action_items,
        key_decisions=data.get("key_decisions", []),
        is_trivial=data.get("is_trivial", False),
    )


def _summary_texts(summary: Summary) -> list[str]:
    """Every free-text field of a Summary, for scanning."""
    texts = [summary.brief, *summary.key_decisions]
    for item in summary.action_items:
        texts.extend([item.assignee, item.action, item.deadline or ""])
    return texts


def _unmask_summary(summary: Summary, masker: Masker) -> Summary:
    """Put the real PII back into the parsed summary — the vault is local.

    Restoring after JSON parsing rather than on the raw response is deliberate: an
    original value may contain a quote or a backslash, which would corrupt the document
    if substituted back before `json.loads`.
    """
    summary.brief = masker.unmask(summary.brief)
    summary.key_decisions = [masker.unmask(decision) for decision in summary.key_decisions]
    for item in summary.action_items:
        item.assignee = masker.unmask(item.assignee)
        item.action = masker.unmask(item.action)
        if item.deadline is not None:
            item.deadline = masker.unmask(item.deadline)

    residue = sorted(
        {p for text in _summary_texts(summary) for p in masker.residual_placeholders(text)}
    )
    if residue:
        click.echo(
            f"Warning: summary keeps unresolved placeholders ({', '.join(residue)}) — "
            "the model altered them, so those values could not be restored.",
            err=True,
        )
    return summary


def masked_preview(transcript: str, settings: Settings) -> tuple[str, dict[str, int]]:
    """The exact text `summarize` would send, and how many distinct values were masked.

    Nothing is sent and no API key is needed. Exists because the built-in rules match
    nothing on a typical spoken transcript, so "masking is on" is otherwise unfalsifiable
    from the outside.
    """
    masker = Masker(enabled=settings.mask_pii, terms=settings.mask_terms)
    return masker.mask(transcript), masker.counts


def summarize(transcript: str, settings: Settings, *, language: str | None = None) -> Summary:
    """Send transcript to LLM, parse structured response, return Summary.

    With `mask_pii` on, the transcript is masked once and every provider in the fallback
    chain — and the JSON retry — receives that same masked text; the real values are
    restored in the returned Summary. The system prompt holds no user data, only the
    note's language code when it has one, so it is not masked.
    """
    masker = Masker(enabled=settings.mask_pii, terms=settings.mask_terms)
    masked = masker.mask(transcript)
    rule = "\n" + _LANGUAGE_RULE.format(language=language) if language else ""
    with _llm.connected(settings) as ask:
        try:
            raw = ask(_SYSTEM_PROMPT + rule, masked)
            summary = _parse_response(raw)
        except (json.JSONDecodeError, KeyError):
            # Retry once with shorter prompt
            raw = ask(_RETRY_PROMPT + rule, masked)
            try:
                summary = _parse_response(raw)
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(
                    f"Failed to parse LLM response after retry: {raw[:500]}"
                ) from exc
    return _unmask_summary(summary, masker)


def format_summary_markdown(summary: Summary) -> str:
    """Convert Summary to markdown string for insertion into transcript file."""
    lines = ["\n## Summary\n", summary.brief, ""]

    if not summary.is_trivial:
        if summary.action_items:
            lines.append("\n### Action Items\n")
            for item in summary.action_items:
                deadline = f" by {item.deadline}" if item.deadline else ""
                lines.append(f"- [ ] **{item.assignee}:** {item.action}{deadline}")
            lines.append("")

        if summary.key_decisions:
            lines.append("\n### Key Decisions\n")
            for decision in summary.key_decisions:
                lines.append(f"- {decision}")
            lines.append("")

    lines.append("\n---\n")
    return "\n".join(lines)


def note_language(md_content: str) -> str | None:
    """The language code in a note's frontmatter, or None."""
    frontmatter = _FRONTMATTER.match(md_content)
    if frontmatter is None:
        return None
    code = _LANGUAGE_CODE.search(frontmatter.group(1))
    return code.group(1) if code else None


def extract_transcript_from_markdown(md_content: str) -> str:
    """Extract the transcript portion (from '# Meeting ...' onward).

    Skips YAML frontmatter, any existing summary section, and the raw half of an
    older diarized note.
    """
    # Find the transcript header
    match = re.search(r"^# Meeting .+", md_content, re.MULTILINE)
    if match:
        return _RAW_TRANSCRIPT.sub("", md_content[match.start() :], count=1)
    return ""


def inject_summary_into_markdown(existing_md: str, summary_md: str) -> str:
    """Insert or replace summary section in existing transcript markdown.

    Summary goes after YAML frontmatter (after closing ---),
    before the transcript header (# Meeting ...).
    If summary section already exists — replace it.
    """
    # Find frontmatter end (second ---)
    fm_pattern = re.compile(r"^---\n.*?^---\n", re.MULTILINE | re.DOTALL)
    fm_match = fm_pattern.match(existing_md)
    if not fm_match:
        # No frontmatter — prepend summary
        return summary_md + existing_md

    frontmatter = existing_md[: fm_match.end()]
    rest = existing_md[fm_match.end() :]

    # Check if summary section already exists
    transcript_match = re.search(r"^# Meeting .+", rest, re.MULTILINE)
    if transcript_match:
        # Everything between frontmatter end and transcript header gets replaced
        transcript_part = rest[transcript_match.start() :]
        return frontmatter + summary_md + transcript_part

    # No transcript header found — append summary before remaining content
    return frontmatter + summary_md + rest


def maybe_summarize(md_path: Path | str | None, settings: Settings) -> None:
    """Summarize a transcript file if summarization is enabled and API key is available.

    Called from CLI after saving transcript. Failures are non-fatal.
    """
    if md_path is None:
        return

    path = Path(md_path) if isinstance(md_path, str) else md_path

    if not settings.summarize:
        return

    if not _llm.provider_chain(settings):
        click.echo(
            "Warning: No LLM API key set, skipping summarization. "
            "Set TAPEBACK_LLM_API_KEY or a provider-specific API key environment variable.",
            err=True,
        )
        return

    try:
        md_content = path.read_text()
        transcript = extract_transcript_from_markdown(md_content)
        if not transcript.strip():
            click.echo("Warning: No transcript content found, skipping summarization.", err=True)
            return

        # No "Summarizing..." here: the pipeline reports the stage through its status
        # callback, which is what reaches the run record; printing it again doubled it.
        summary = summarize(transcript, settings, language=note_language(md_content))
        summary_md = format_summary_markdown(summary)
        new_content = inject_summary_into_markdown(md_content, summary_md)
        path.write_text(new_content)
        click.echo("Summary added.", err=True)
    except Exception as exc:
        click.echo(f"Warning: Summarization failed: {exc}", err=True)
