#!/usr/bin/env python3
"""The mutation ratchet: surviving and untested mutants may only go down.

    uv run mutmut export-cicd-stats
    python3 scripts/mutation_ratchet.py mutants/mutmut-cicd-stats.json \\
        --survived-max 40 --no-tests-max 5

Reads the counts `mutmut export-cicd-stats` writes. Exit 0 within both maxima, 1 over
either or when the run did not finish, 2 when the counts cannot be read. A count under
its maximum asks for the maximum to come down, so the gain cannot be spent later.

A worker killed by a signal is what mutmut calls a segfault. Under the memory cap
`pre_release_qa.sh` sets, it is a mutant that allocated without end: its tests did not
pass, so it died rather than survived.
"""

import argparse
import json
import sys
from pathlib import Path

# Every result the export holds; what its total holds beyond them was never checked.
RESULTS = (
    "killed",
    "survived",
    "no_tests",
    "skipped",
    "suspicious",
    "timeout",
    "check_was_interrupted_by_user",
    "segfault",
)
# A run that stopped early says nothing about the mutants it did not reach, and an exit
# code mutmut cannot read may hide a survivor.
NO_VERDICT = {
    "check_was_interrupted_by_user": "interrupted: the run did not finish",
    "suspicious": "suspicious: mutmut could not read their exit code",
}
RATCHETED = {"survived": "survived", "no_tests": "untested"}


def verdict(stats: dict, maxima: dict[str, int]) -> tuple[bool, list[str]]:
    """Whether the run passes, and the lines that say why."""
    lines = [
        f"mutation: {stats['killed']} of {stats['total']} killed, {stats['survived']} survived, "
        f"{stats['no_tests']} untested, {stats['timeout']} timed out"
    ]
    passed = stats["total"] > 0
    if not passed:
        lines.append("mutation: no mutants: the slice in [tool.mutmut] matches nothing")
    not_checked = stats["total"] - sum(stats[key] for key in RESULTS)
    if not_checked:
        lines.append(f"mutation: {not_checked} not checked: the run did not finish")
        passed = False
    for key, words in NO_VERDICT.items():
        if stats[key]:
            lines.append(f"mutation: {stats[key]} {words}")
            passed = False
    if stats["segfault"]:
        lines.append(
            f"mutation: {stats['segfault']} died: the memory cap or a crash killed their "
            "worker, so not survivors"
        )
    for key, word in RATCHETED.items():
        count, maximum = stats[key], maxima[key]
        if count > maximum:
            lines.append(f"mutation: {count} {word}, over the maximum of {maximum}")
            passed = False
        elif count < maximum:
            lines.append(f"mutation: {count} {word}: lower its maximum from {maximum} to {count}")
    return passed, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The mutation ratchet")
    parser.add_argument("stats", type=Path, help="mutmut's CI/CD stats JSON")
    parser.add_argument("--survived-max", type=int, required=True)
    parser.add_argument("--no-tests-max", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        passed, lines = verdict(
            stats, {"survived": args.survived_max, "no_tests": args.no_tests_max}
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"mutation: cannot read {args.stats}: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
