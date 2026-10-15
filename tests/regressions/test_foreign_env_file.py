"""Regression: tapeback died in any directory whose .env belongs to another app.

Settings reads ~/.config/tapeback/.env and the .env of the directory it runs in.
pydantic-settings forbids unknown keys by default, so every key of another
project's .env became a validation error, and the error showed its value:
`tapeback start` in a web project's directory printed that project's bot token and
private keys. The README's own LLM setup failed the same way: it put
GEMINI_API_KEY into ~/.config/tapeback/.env.

A key tapeback does not own is ignored: never validated, never shown.
"""

from tapeback.cli import cli

FOREIGN_SECRET = "fake-bot-token-0123456789"
PROVIDER_KEY = "fake-gemini-key-0123456789"


def test_another_apps_env_in_the_working_directory_is_ignored(runner, real_env_files):
    _, working_dir = real_env_files
    (working_dir / ".env").write_text(f"OTHERAPP_BOT_TOKEN={FOREIGN_SECRET}\nOTHERAPP_PORT=5051\n")

    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0, result.exception
    assert FOREIGN_SECRET not in result.output
    assert FOREIGN_SECRET not in repr(result.exception)


def test_a_provider_variable_in_the_user_config_no_longer_kills_startup(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text(f"TAPEBACK_LLM_PROVIDER=gemini\nGEMINI_API_KEY={PROVIDER_KEY}\n")

    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0, result.exception
    assert PROVIDER_KEY not in result.output
    assert PROVIDER_KEY not in repr(result.exception)
