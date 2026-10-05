"""scripts/gate.sh must run exactly what CI runs, and never report green over a failure."""

import re

import pytest

from tests.fixtures import REPO_ROOT

CI_CHECKS = [
    "python3 scripts/check-workflow-pins.py",
    "uv lock --locked",
    "uv run ruff check",
    "uv run ruff format --check",
    "uv run ty check",
    "uv run pytest --tb=short",
]
DEB_SMOKE = "deb-smoke.sh"
ARCH_SMOKE = "docker run --rm -v <root>:/src:ro archlinux:latest /src/scripts/arch-smoke.sh"


def _workflow(name: str) -> str:
    return (REPO_ROOT / ".github" / "workflows" / name).read_text()


def _gate_list(name: str) -> list[str]:
    """A bash array from gate.sh, e.g. DEB_PATHS=("a" "b"), possibly over several lines."""
    gate = (REPO_ROOT / "scripts" / "gate.sh").read_text()
    match = re.search(rf"^{name}=\((?P<items>[^)]*)\)", gate, re.M)
    assert match is not None
    return re.findall(r'"([^"]+)"', match["items"])


def _workflow_paths(name: str) -> list[str]:
    match = re.search(r"^\s+paths:\n(?P<items>(?:\s+- .+\n)+)", _workflow(name), re.M)
    assert match is not None
    return [item.strip('"') for item in re.findall(r"- (.+)", match["items"])]


def test_gate_runs_the_ci_checks_in_order_and_passes(run_gate):
    result, commands = run_gate(changed=["README.md"])

    assert commands == CI_CHECKS
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith("all gates passed")


def test_one_failed_check_fails_the_gate_without_skipping_the_rest(run_gate):
    result, commands = run_gate(changed=["README.md"], fail="uv run ty check")

    assert commands == CI_CHECKS
    assert result.returncode == 1
    assert "all gates passed" not in result.stdout
    assert result.stdout.rstrip().endswith("failed:\n  ty")


@pytest.mark.parametrize(
    ("changed", "stands"),
    [
        (["src/tapeback/cli.py"], [DEB_SMOKE, ARCH_SMOKE]),
        (["packaging/tapeback-llm/PKGBUILD"], [DEB_SMOKE, ARCH_SMOKE]),
        (["scripts/build-deb.sh"], [DEB_SMOKE]),
        (["scripts/arch-smoke.sh"], [ARCH_SMOKE]),
        (["docs/release-testing.md", "tests/test_cli.py"], []),
    ],
    ids=["source", "packaging", "deb only", "arch only", "neither"],
)
def test_stands_run_when_the_diff_touches_their_paths(run_gate, changed, stands):
    result, commands = run_gate(changed=changed)

    assert commands == CI_CHECKS + stands
    assert result.returncode == 0


def test_all_runs_every_stand_whatever_the_diff(run_gate):
    _, commands = run_gate("--all", changed=["README.md"])

    assert commands == [*CI_CHECKS, DEB_SMOKE, ARCH_SMOKE]


def test_without_a_merge_base_every_stand_runs(run_gate):
    """Nothing can be ruled out without a base, so nothing is."""
    _, commands = run_gate(changed=None)

    assert commands == [*CI_CHECKS, DEB_SMOKE, ARCH_SMOKE]


def test_failed_stand_fails_the_gate_and_the_next_stand_still_runs(run_gate):
    # The stub matches its basename and arguments; deb-smoke.sh takes none.
    result, commands = run_gate(changed=["src/tapeback/cli.py"], fail="deb-smoke.sh ")

    assert commands == [*CI_CHECKS, DEB_SMOKE, ARCH_SMOKE]
    assert result.returncode == 1
    assert result.stdout.rstrip().endswith("failed:\n  .deb smoke")


@pytest.mark.parametrize("args", [["pytest"], ["--all", "--all"]])
def test_gate_refuses_anything_but_all(run_gate, args):
    result, commands = run_gate(*args)

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert commands == []


def test_gate_mirrors_the_ci_job():
    """CI's checks, with its install steps left out: apt is the runner's business,
    and `uv sync --locked` becomes `uv lock --locked` — the same lockfile question
    without rewriting the developer's environment."""
    runs = re.findall(r"^\s+run: (.+)$", _workflow("ci.yml"), re.M)

    mirrored = [
        "uv lock --locked" if run.startswith("uv sync --locked") else run
        for run in runs
        if not run.startswith("sudo apt-get")
    ]

    assert mirrored == CI_CHECKS


def test_stands_filter_on_the_paths_the_workflows_filter_on():
    """Plus the local script each stand runs, which CI itself never calls."""
    assert _gate_list("DEB_PATHS") == [*_workflow_paths("deb-e2e.yml"), "scripts/deb-smoke.sh"]
    assert _gate_list("ARCH_PATHS") == _workflow_paths("arch-e2e.yml")
