"""`tapeback setup`: diarization and the summary switched on and off in the user's file.

The file is real (`real_env_files`): what setup writes, keeps and reports is read back
from disk. Only the terminal is faked, where a test needs a person to answer.
"""

import os
import stat

import pytest

from tapeback import cli as cli_module
from tapeback.cli import cli


def _lines(result) -> list[str]:
    return result.output.splitlines()


def test_a_flag_writes_the_user_file_and_keeps_the_rest_of_it(runner, real_env_files):
    user_config, _ = real_env_files
    original = '# mine\nTAPEBACK_HF_TOKEN="hf_fake"\n\nexport TAPEBACK_LANGUAGE=ru\n'
    user_config.write_text(original)
    user_config.chmod(0o640)

    result = runner.invoke(cli, ["setup", "--no-diarize"])

    assert result.exit_code == 0, result.output
    assert user_config.read_text() == original + "TAPEBACK_DIARIZE=false\n"
    assert stat.S_IMODE(user_config.stat().st_mode) == 0o640
    assert _lines(result)[:3] == [
        f"Wrote TAPEBACK_DIARIZE=false to {user_config}",
        f"Diarization: off ({user_config})",
        "Summary: on (default)",
    ]


def test_a_value_the_file_already_holds_is_not_written_again(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text("TAPEBACK_DIARIZE=false\n")
    inode = user_config.stat().st_ino

    result = runner.invoke(cli, ["setup", "--no-diarize"])

    assert result.exit_code == 0, result.output
    assert user_config.stat().st_ino == inode
    assert _lines(result)[0] == f"Diarization: off ({user_config})"


@pytest.mark.parametrize("where", ["environment", "./.env"])
def test_the_report_names_what_overrides_the_user_file(runner, real_env_files, monkeypatch, where):
    user_config, working_dir = real_env_files
    if where == "environment":
        monkeypatch.setenv("TAPEBACK_SUMMARIZE", "false")
    else:
        (working_dir / ".env").write_text("TAPEBACK_SUMMARIZE=false\n")

    result = runner.invoke(cli, ["setup", "--summarize"])

    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "TAPEBACK_SUMMARIZE=true\n"
    assert f"Summary: off ({where}, overriding {user_config})" in _lines(result)


def test_without_a_terminal_setup_only_reports(runner, real_env_files):
    user_config, _ = real_env_files

    result = runner.invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    assert user_config.read_text() == ""
    assert _lines(result)[:2] == ["Diarization: on (default)", "Summary: on (default)"]


def test_in_a_terminal_setup_asks_and_writes_only_a_changed_answer(
    runner, real_env_files, monkeypatch
):
    user_config, _ = real_env_files
    monkeypatch.setattr(cli_module, "_stdin_is_terminal", lambda: True)

    # Diarization off; the summary question answered with its default.
    result = runner.invoke(cli, ["setup"], input="n\n\n")

    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "TAPEBACK_DIARIZE=false\n"
    assert "Diarize speakers? [Y/n]: n" in _lines(result)
    assert "Summarize meetings with an LLM? [Y/n]: " in _lines(result)


def test_no_diarization_question_when_the_stt_model_labels_speakers(
    runner, real_env_files, monkeypatch
):
    user_config, _ = real_env_files
    user_config.write_text(
        "TAPEBACK_STT_BACKEND=openai\nTAPEBACK_STT_MODEL=gpt-4o-transcribe-diarize\n"
    )
    monkeypatch.setattr(cli_module, "_stdin_is_terminal", lambda: True)

    result = runner.invoke(cli, ["setup"], input="\n")

    assert result.exit_code == 0, result.output
    assert not any(line.startswith("Diarize speakers?") for line in _lines(result))
    assert "Diarization: speakers come from the STT model gpt-4o-transcribe-diarize" in _lines(
        result
    )


def test_a_new_user_file_is_private(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.unlink()
    user_config.parent.rmdir()

    result = runner.invoke(cli, ["setup", "--no-summarize"])

    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "TAPEBACK_SUMMARIZE=false\n"
    assert stat.S_IMODE(user_config.stat().st_mode) == 0o600
    assert stat.S_IMODE(user_config.parent.stat().st_mode) == 0o700


def test_a_symlinked_user_file_stays_a_symlink(runner, real_env_files, tmp_path):
    """Dotfile managers link the file; setup writes where the link points."""
    user_config, _ = real_env_files
    kept = tmp_path / "dotfiles" / "tapeback.env"
    kept.parent.mkdir()
    kept.write_text("TAPEBACK_DIARIZE=true\n")
    user_config.unlink()
    user_config.symlink_to(kept)

    result = runner.invoke(cli, ["setup", "--no-diarize"])

    assert result.exit_code == 0, result.output
    assert user_config.is_symlink()
    assert kept.read_text() == "TAPEBACK_DIARIZE=false\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_a_file_setup_cannot_write_gets_the_lines_to_add_by_hand(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.parent.chmod(0o500)
    try:
        result = runner.invoke(cli, ["setup", "--no-diarize", "--no-summarize"])
    finally:
        user_config.parent.chmod(0o700)

    assert result.exit_code == 1
    assert user_config.read_text() == ""
    assert _lines(result)[-2:] == ["TAPEBACK_DIARIZE=false", "TAPEBACK_SUMMARIZE=false"]


def test_hints_name_what_is_missing(runner, real_env_files):
    result = runner.invoke(cli, ["setup", "--diarize", "--summarize"])

    assert result.exit_code == 0, result.output
    assert _lines(result)[-3:] == [
        "Hint: diarization needs TAPEBACK_HF_TOKEN; see Speaker diarization in the README.",
        "Hint: summaries need an LLM key in TAPEBACK_LLM_API_KEY; "
        "see LLM summarization in the README.",
        "Hint: a running tray keeps its old settings until it is restarted.",
    ]


def test_no_hints_when_nothing_is_missing_or_written(runner, real_env_files):
    user_config, _ = real_env_files
    user_config.write_text("TAPEBACK_HF_TOKEN=hf_fake\nTAPEBACK_LLM_API_KEY=fake-llm-key\n")

    result = runner.invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    assert not any(line.startswith("Hint:") for line in _lines(result))
