"""scripts/gate.sh must run exactly what CI runs, and never report green over a failure."""

import re

from tests.fixtures import REPO_ROOT

GATE_COMMANDS = [
    "python3 scripts/check-workflow-pins.py",
    "uv lock --locked",
    "uv run ruff check",
    "uv run ruff format --check",
    "uv run ty check",
    "uv run pytest --tb=short",
]


def test_gate_runs_every_check_in_order_and_passes(run_gate):
    result, commands = run_gate()

    assert commands == GATE_COMMANDS
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith("all gates passed")


def test_one_failed_check_fails_the_gate_without_skipping_the_rest(run_gate):
    result, commands = run_gate(fail="uv run ty check")

    assert commands == GATE_COMMANDS
    assert result.returncode == 1
    assert "all gates passed" not in result.stdout
    assert result.stdout.rstrip().endswith("failed:\n  ty")


def test_gate_refuses_arguments_instead_of_running_a_part(run_gate):
    result, commands = run_gate("pytest")

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert commands == []


def test_gate_mirrors_the_ci_job():
    """CI's checks, with its install steps left out: apt is the runner's business,
    and `uv sync --locked` becomes `uv lock --locked` — the same lockfile question
    without rewriting the developer's environment."""
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    runs = re.findall(r"^\s+run: (.+)$", ci, re.M)

    mirrored = [
        "uv lock --locked" if run.startswith("uv sync --locked") else run
        for run in runs
        if not run.startswith("sudo apt-get")
    ]

    assert mirrored == GATE_COMMANDS
