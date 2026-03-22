"""Meeting summarization via LLM — extract summary, action items, key decisions."""

import json
import os
import re
import time
from dataclasses import dataclass, field

import click

from meetrec.settings import Settings

_SYSTEM_PROMPT = """\
You analyze meeting transcripts and extract structured information.

You MUST respond with valid JSON only. No markdown, no explanation, no preamble.

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


_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}

_OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def _resolve_api_key(settings: Settings) -> str:
    """Resolve API key: MEETREC_LLM_API_KEY > provider-specific env var."""
    if settings.llm_api_key:
        return settings.llm_api_key

    env_var = _PROVIDER_ENV_VARS.get(settings.llm_provider, "")
    key = os.environ.get(env_var, "") if env_var else ""
    if key:
        return key

    raise RuntimeError(
        f"No API key for {settings.llm_provider}. "
        f"Set MEETREC_LLM_API_KEY or {env_var} environment variable."
    )


def _get_model(settings: Settings) -> str:
    """Return model name: explicit setting or provider default."""
    if settings.llm_model:
        return settings.llm_model
    defaults: dict[str, str] = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.0-flash",
        "openrouter": "google/gemma-3-27b-it:free",
        "deepseek": "deepseek-chat",
        "qwen": "qwen-turbo",
    }
    return defaults[settings.llm_provider]


_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 5  # seconds, doubles each retry: 5, 10, 20


def _call_llm(system_prompt: str, user_message: str, settings: Settings) -> str:
    """Call the configured LLM provider with retry on rate limits.

    Retries up to _MAX_RETRIES times on 429/529 errors with exponential backoff.
    """
    api_key = _resolve_api_key(settings)
    model = _get_model(settings)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _call_llm_once(system_prompt, user_message, settings, api_key, model)
        except Exception as exc:
            status = _get_http_status(exc)
            if status not in (429, 529) or attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_DELAY * (2**attempt)
            click.echo(
                f"Rate limited (HTTP {status}), retrying in {delay}s "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})...",
                err=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")  # pragma: no cover


def _get_http_status(exc: Exception) -> int | None:
    """Extract HTTP status code from provider SDK exceptions."""
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _call_llm_once(
    system_prompt: str,
    user_message: str,
    settings: Settings,
    api_key: str,
    model: str,
) -> str:
    """Single LLM call without retry."""
    if settings.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    # All other providers use OpenAI-compatible Chat Completions API
    import openai

    base_url = _OPENAI_COMPATIBLE_BASE_URLS.get(settings.llm_provider)
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _parse_response(raw: str) -> Summary:
    """Parse JSON response into Summary dataclass."""
    data = json.loads(raw)
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


def summarize(transcript: str, settings: Settings) -> Summary:
    """Send transcript to LLM, parse structured response, return Summary."""
    try:
        raw = _call_llm(_SYSTEM_PROMPT, transcript, settings)
        return _parse_response(raw)
    except json.JSONDecodeError, KeyError:
        # Retry once with shorter prompt
        raw = _call_llm(_RETRY_PROMPT, transcript, settings)
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"Failed to parse LLM response after retry: {raw[:500]}") from exc


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


def extract_transcript_from_markdown(md_content: str) -> str:
    """Extract the transcript portion (from '# Meeting ...' onward).

    Skips YAML frontmatter and any existing summary section.
    """
    # Find the transcript header
    match = re.search(r"^# Meeting .+", md_content, re.MULTILINE)
    if match:
        return md_content[match.start() :]
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


def maybe_summarize(md_path: str | None, settings: Settings) -> None:
    """Summarize a transcript file if summarization is enabled and API key is available.

    Called from CLI after saving transcript. Failures are non-fatal.
    """
    if md_path is None:
        return

    from pathlib import Path

    path = Path(md_path) if isinstance(md_path, str) else md_path

    if not settings.summarize:
        return

    try:
        _resolve_api_key(settings)
    except RuntimeError:
        click.echo(
            "Warning: No LLM API key set, skipping summarization. "
            "Set MEETREC_LLM_API_KEY or ANTHROPIC_API_KEY/OPENAI_API_KEY.",
            err=True,
        )
        return

    try:
        md_content = path.read_text()
        transcript = extract_transcript_from_markdown(md_content)
        if not transcript.strip():
            click.echo("Warning: No transcript content found, skipping summarization.", err=True)
            return

        click.echo("Summarizing...", err=True)
        summary = summarize(transcript, settings)
        summary_md = format_summary_markdown(summary)
        new_content = inject_summary_into_markdown(md_content, summary_md)
        path.write_text(new_content)
        click.echo("Summary added.", err=True)
    except Exception as exc:
        click.echo(f"Warning: Summarization failed: {exc}", err=True)
