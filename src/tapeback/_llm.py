"""How a summary reaches an LLM: providers, keys, the fallback chain, retries, SDK calls.

summarizer.py decides what is sent (prompt, masking, parsing); this module, how. Every
provider is one row of PROVIDERS, which the test isolation reads too: a provider added
here is cut off from the suite's environment the moment it exists.
"""

import ipaddress
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import click

from tapeback import _llama_server, const
from tapeback.settings import Settings

if TYPE_CHECKING:
    import openai

# A provider is retried on HTTP 429/529 this many times, after RETRY_BASE_DELAY seconds
# doubled each time (5, 10, 20), before the chain moves on to the next one.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5


@dataclass(frozen=True)
class Provider:
    """One LLM vendor, as tapeback reaches it."""

    env_var: str  # the vendor's own key variable, read from the environment only
    default_model: str  # used when TAPEBACK_LLM_MODEL is not set
    sdk: Literal["anthropic", "openai"]
    base_url: str | None = None  # an OpenAI-compatible API; None: the SDK's own address


# In fallback order: when the configured provider fails, the others are tried top down.
# Addresses and default models are const.py's, where vendor-dictated values live.
PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider("ANTHROPIC_API_KEY", const.LLM_DEFAULT_MODEL_ANTHROPIC, "anthropic"),
    "groq": Provider("GROQ_API_KEY", const.LLM_DEFAULT_MODEL_GROQ, "openai", const.API_BASE_GROQ),
    "openrouter": Provider(
        "OPENROUTER_API_KEY",
        const.LLM_DEFAULT_MODEL_OPENROUTER,
        "openai",
        const.API_BASE_OPENROUTER,
    ),
    "gemini": Provider(
        "GEMINI_API_KEY", const.LLM_DEFAULT_MODEL_GEMINI, "openai", const.API_BASE_GEMINI
    ),
    "deepseek": Provider(
        "DEEPSEEK_API_KEY", const.LLM_DEFAULT_MODEL_DEEPSEEK, "openai", const.API_BASE_DEEPSEEK
    ),
    "qwen": Provider(
        "DASHSCOPE_API_KEY", const.LLM_DEFAULT_MODEL_QWEN, "openai", const.API_BASE_QWEN
    ),
    "openai": Provider("OPENAI_API_KEY", const.LLM_DEFAULT_MODEL_OPENAI, "openai"),
}

# An OpenAI-compatible API the user names in TAPEBACK_LLM_BASE_URL. Not one of PROVIDERS:
# it has no key variable of its own and no default model, and it stands alone. A failing
# endpoint never falls back to a key the environment happens to hold for one of
# PROVIDERS, and none of them falls back to it.
ENDPOINT = "openai_compatible"

# A llama-server tapeback starts for a summary and stops after it (_llama_server). Like
# ENDPOINT it stands alone: the transcript stays on this machine, so a failure never
# falls back to a cloud provider.
LOCAL = "local"


def endpoint_url(settings: Settings) -> str:
    """The endpoint's address, checked before any text is sent to it.

    Refused: no address, no model, not http(s), or credentials in the URL, which belong
    in TAPEBACK_LLM_API_KEY and would otherwise be echoed with the address. Plain http
    to a host other than this machine goes ahead with a warning: the transcript crosses
    the network unencrypted. Masking applies either way, since a forwarded port can make
    even a loopback address remote.
    """
    url = settings.llm_base_url.strip()
    if not url:
        raise RuntimeError(
            "TAPEBACK_LLM_BASE_URL is not set: the endpoint's address, "
            "such as http://127.0.0.1:8080/v1."
        )
    if not settings.llm_model:
        raise RuntimeError("TAPEBACK_LLM_MODEL is not set: the model the endpoint serves.")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RuntimeError("TAPEBACK_LLM_BASE_URL is not an http(s) address.")
    if parts.username or parts.password:
        raise RuntimeError(
            f"TAPEBACK_LLM_BASE_URL for {parts.hostname} carries credentials; "
            "put the key in TAPEBACK_LLM_API_KEY."
        )
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        click.echo(f"Warning: the transcript goes to {parts.hostname} over plain http.", err=True)
    return url


def endpoint_address(settings: Settings) -> str:
    """host:port of the endpoint, for display: never its path, query or credentials."""
    parts = urlsplit(settings.llm_base_url.strip())
    if not parts.hostname:
        return "no address"
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port else host


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def api_key(provider: str, settings: Settings) -> str:
    """The key for provider, or "" when there is none.

    TAPEBACK_LLM_API_KEY belongs to the configured provider; every cloud provider also
    reads its own variable from the environment. The endpoint reads nothing else.
    """
    if provider == settings.llm_provider:
        key = settings.llm_api_key.get_secret_value()
        if key:
            return key
    if provider == ENDPOINT:
        return ""
    return os.environ.get(PROVIDERS[provider].env_var, "")


def has_api_key(settings: Settings) -> bool:
    """Whether the configured provider has a key; the key itself never leaves here."""
    return bool(api_key(settings.llm_provider, settings))


def provider_chain(settings: Settings) -> list[tuple[str, str, str]]:
    """The (provider, key, model) list to try, the configured provider first.

    Providers without a key are skipped. An explicit TAPEBACK_LLM_MODEL applies to the
    configured provider only. The endpoint is a chain of its own, keyed or not.
    """
    if settings.llm_provider == ENDPOINT:
        return [(ENDPOINT, api_key(ENDPOINT, settings), settings.llm_model)]
    if settings.llm_provider == LOCAL:
        return [(LOCAL, "", _llama_server.model_name(settings))]

    chain: list[tuple[str, str, str]] = []

    primary = settings.llm_provider
    primary_key = api_key(primary, settings)
    if primary_key:
        chain.append((primary, primary_key, settings.llm_model or PROVIDERS[primary].default_model))

    for provider, spec in PROVIDERS.items():
        if provider == primary:
            continue
        key = api_key(provider, settings)
        if key:
            chain.append((provider, key, spec.default_model))

    return chain


@contextmanager
def connected(settings: Settings) -> Iterator[Callable[[str, str], str]]:
    """Ask as often as the block needs: for `local`, all through one server start.

    A summary asks twice when its first answer is not JSON; a local model is loaded
    once for both.
    """
    if settings.llm_provider != LOCAL:
        yield lambda system, user: call(system, user, settings)
        return
    with _llama_server.running(settings) as server:
        client = _openai_client(server.key, server.url, settings.llm_local_timeout, retries=False)
        yield lambda system, user: _chat(client, server.model, system, user)


def call(system_prompt: str, user_message: str, settings: Settings) -> str:
    """Send one request through the chain, or to a server started for it.

    Each provider is retried on rate limits before the next one is tried.
    """
    if settings.llm_provider == LOCAL:
        with connected(settings) as ask:
            return ask(system_prompt, user_message)
    chain = provider_chain(settings)
    if not chain:
        raise RuntimeError(
            "No LLM providers available. "
            "Set TAPEBACK_LLM_API_KEY or a provider-specific API key environment variable."
        )
    if settings.llm_provider == ENDPOINT:
        endpoint_url(settings)  # refused or warned before anything is sent

    last_exc: Exception = RuntimeError("No providers attempted")
    for i, (provider, key, model) in enumerate(chain):
        try:
            return _call_with_retry(system_prompt, user_message, provider, key, model, settings)
        except Exception as exc:
            last_exc = exc
            if i < len(chain) - 1:
                click.echo(
                    f"Provider {provider} failed: {exc}. Trying next provider...",
                    err=True,
                )

    raise last_exc


def _call_with_retry(
    system_prompt: str,
    user_message: str,
    provider: str,
    key: str,
    model: str,
    settings: Settings,
) -> str:
    """Call a single provider, retrying on rate limits (429/529)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _call_once(system_prompt, user_message, provider, key, model, settings)
        except Exception as exc:
            status = _http_status(exc)
            if status not in (429, 529) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            click.echo(
                f"Rate limited (HTTP {status}), retrying in {delay}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                err=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")  # pragma: no cover


def _http_status(exc: Exception) -> int | None:
    """Extract HTTP status code from provider SDK exceptions."""
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _call_once(
    system_prompt: str,
    user_message: str,
    provider: str,
    key: str,
    model: str,
    settings: Settings,
) -> str:
    """Single LLM call without retry."""
    if provider != ENDPOINT and PROVIDERS[provider].sdk == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install with: uv pip install tapeback[llm]"
            ) from None

        ant_client = anthropic.Anthropic(api_key=key, timeout=settings.llm_timeout)
        ant_response = ant_client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        # Current models think adaptively, so a thinking block can come before the text.
        blocks = ant_response.content
        texts = [block.text for block in blocks if isinstance(block, anthropic.types.TextBlock)]
        if not texts:
            kinds = ", ".join(block.type for block in blocks)
            raise RuntimeError(f"anthropic reply holds no text block, only: {kinds or 'nothing'}")
        return "".join(texts)

    if provider == ENDPOINT:
        url = settings.llm_base_url.strip()
        client = _openai_client(key, url, settings.llm_timeout, retries=False)
    else:
        url = PROVIDERS[provider].base_url
        client = _openai_client(key, url, settings.llm_timeout, retries=True)
    return _chat(client, model, system_prompt, user_message)


def _openai_client(
    key: str, base_url: str | None, timeout: float, *, retries: bool
) -> "openai.OpenAI":
    """An OpenAI SDK client. Without retries for an endpoint or the local server: each
    retry would send the whole transcript again, billed or not."""
    try:
        import openai
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Install with: uv pip install tapeback[llm]"
        ) from None
    if retries:
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout)
    return openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)


def _chat(client: "openai.OpenAI", model: str, system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
