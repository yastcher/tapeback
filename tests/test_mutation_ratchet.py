"""scripts/mutation_ratchet.py: survivors and untested mutants may only go down."""

import json

import pytest

from tests.fixtures import load_script

ratchet = load_script("mutation_ratchet")


def _stats(tmp_path, not_checked=0, **counts):
    """mutmut's export: its total also holds the mutants no result was written for."""
    stats = {
        "killed": 80,
        "survived": 10,
        "no_tests": 3,
        "skipped": 0,
        "suspicious": 0,
        "timeout": 0,
        "check_was_interrupted_by_user": 0,
        "segfault": 0,
    } | counts
    stats["total"] = sum(stats.values()) + not_checked
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    return path


def _run(path, survived_max=10, no_tests_max=3):
    return ratchet.main(
        [str(path), "--survived-max", str(survived_max), "--no-tests-max", str(no_tests_max)]
    )


def test_counts_at_their_maxima_pass_quietly(tmp_path, capsys):
    assert _run(_stats(tmp_path)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "mutation: 80 of 93 killed, 10 survived, 3 untested, 0 timed out"
    ]


@pytest.mark.parametrize(
    ("counts", "line"),
    [
        ({"survived": 11}, "mutation: 11 survived, over the maximum of 10"),
        ({"no_tests": 4}, "mutation: 4 untested, over the maximum of 3"),
        ({"check_was_interrupted_by_user": 1}, "mutation: 1 interrupted: the run did not finish"),
        ({"not_checked": 5}, "mutation: 5 not checked: the run did not finish"),
        ({"suspicious": 1}, "mutation: 1 suspicious: mutmut could not read their exit code"),
        (
            {"killed": 0, "survived": 0, "no_tests": 0},
            "mutation: no mutants: the slice in [tool.mutmut] matches nothing",
        ),
    ],
    ids=["survived over", "untested over", "interrupted", "not checked", "suspicious", "none"],
)
def test_a_run_over_a_maximum_or_unfinished_fails(tmp_path, capsys, counts, line):
    assert _run(_stats(tmp_path, **counts)) == 1
    assert line in capsys.readouterr().out.splitlines()


def test_a_mutant_whose_worker_was_killed_is_not_a_survivor(tmp_path, capsys):
    # mutmut reads a worker killed by a signal as a segfault; under the memory cap that
    # is a mutant that allocated without end.
    assert _run(_stats(tmp_path, killed=78, segfault=2)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "mutation: 78 of 93 killed, 10 survived, 3 untested, 0 timed out",
        "mutation: 2 died: the memory cap or a crash killed their worker, so not survivors",
    ]


def test_a_count_under_its_maximum_passes_and_asks_for_the_maximum_to_come_down(tmp_path, capsys):
    assert _run(_stats(tmp_path, survived=7, no_tests=1)) == 0
    assert capsys.readouterr().out.splitlines()[1:] == [
        "mutation: 7 survived: lower its maximum from 10 to 7",
        "mutation: 1 untested: lower its maximum from 3 to 1",
    ]


@pytest.mark.parametrize(
    "content", [None, "not json", '{"killed": 1}'], ids=["missing", "garbled", "incomplete"]
)
def test_counts_that_cannot_be_read_stop_the_check(tmp_path, capsys, content):
    path = tmp_path / "stats.json"
    if content is not None:
        path.write_text(content)

    assert _run(path) == 2
    assert capsys.readouterr().err.startswith(f"mutation: cannot read {path}")
