#!/usr/bin/env python3
"""CRAP per function: where complexity meets missing tests.

    uv run coverage json -q -o tmp/coverage.json
    python3 scripts/crap.py tmp/coverage.json --max 29

CRAP = cc^2 * (1 - coverage)^3 + cc, crap4j's formula: a function with complexity 10
is CRAP 10 when fully covered and 110 when not covered at all. Complexity is McCabe's,
counted over the function's own body with radon's rules; a nested function or class is
its own region, as coverage.py reports it, and a lambda belongs to its parent. Coverage
is the line coverage of that region in coverage.py's JSON report (7.6 or later).

Exit 0 when no function scores over `--max`, 1 when one does, 2 when the report cannot
be read or does not match the sources: a function the two disagree on would be scored
wrong rather than not at all.
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# One branch each, as radon counts them; `else` and `finally` add none.
DECISIONS = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.match_case,
)
OWN_REGIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
SHOWN = 10


@dataclass(frozen=True)
class Score:
    where: str
    complexity: int
    covered: float  # 0..1
    crap: float


def complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """McCabe's complexity of a function's own body: 1 plus one per decision."""
    score = 1
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, OWN_REGIONS):
            continue
        if isinstance(node, DECISIONS):
            score += 1
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        stack.extend(ast.iter_child_nodes(node))
    return score


def crap(cc: int, covered: float) -> float:
    return cc * cc * (1 - covered) ** 3 + cc


def scores(report: dict, root: Path) -> tuple[list[Score], list[str]]:
    """Every function's score, and the regions the sources do not explain."""
    found, unmatched = [], []
    for name, data in report["files"].items():
        functions = {
            node.lineno: node
            for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for region, info in data["functions"].items():
            if not region:  # the module's own code
                continue
            where = f"{name}:{info['start_line']} {region}"
            node = functions.get(info["start_line"])
            if node is None:
                unmatched.append(where)
                continue
            cc = complexity(node)
            covered = info["summary"]["percent_covered"] / 100
            found.append(Score(where, cc, covered, crap(cc, covered)))
    return sorted(found, key=lambda score: score.crap, reverse=True), unmatched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CRAP per function, from a coverage report")
    parser.add_argument("report", type=Path, help="coverage.py's JSON report")
    parser.add_argument("--max", type=float, required=True, help="the highest CRAP allowed")
    parser.add_argument("--root", type=Path, default=ROOT, help="where the report's paths start")
    args = parser.parse_args(argv)

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        found, unmatched = scores(report, args.root)
    except (OSError, ValueError, KeyError) as error:
        print(f"crap: cannot read {args.report}: {error}", file=sys.stderr)
        return 2
    if unmatched:
        print("crap: coverage names functions the sources do not have:", file=sys.stderr)
        for where in unmatched:
            print(f"  {where}", file=sys.stderr)
        return 2

    for score in found[:SHOWN]:
        print(f"{score.crap:7.1f}  cc {score.complexity:2d}  {score.covered:6.1%}  {score.where}")
    over = [score for score in found if score.crap > args.max]
    if over:
        print(f"crap: {len(over)} of {len(found)} functions over {args.max:g}")
        return 1
    print(f"crap: {len(found)} functions, none over {args.max:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
