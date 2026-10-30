"""scripts/crap.py: McCabe's complexity, the CRAP formula, and the verdict."""

import ast
import json
import textwrap

import pytest

from tests.fixtures import load_script

crap = load_script("crap")


def _complexity(source: str) -> int:
    function = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(function, ast.FunctionDef)
    return crap.complexity(function)


_IF_ELIF_ELSE = """
    def f(x):
        if x:
            return 1
        elif x > 1:
            return 2
        else:
            return 3
"""
_EXCEPTS = """
    def f():
        try:
            pass
        except OSError:
            pass
        except ValueError:
            pass
        finally:
            pass
"""
_MATCH = """
    def f(x):
        match x:
            case 1:
                pass
            case 2:
                pass
            case _:
                pass
"""
_NESTED = """
    def f(x):
        def inner():
            if x:
                return 1
        class C:
            def m(self):
                if x:
                    return 2
        return inner
"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f():\n    return 1\n", 1),
        (_IF_ELIF_ELSE, 3),
        ("def f(xs):\n    for x in xs:\n        while x:\n            x -= 1\n", 3),
        (_EXCEPTS, 3),
        ("def f(a, b, c):\n    return a and b and c\n", 3),
        ("def f(xs):\n    return [x for x in xs if x if x > 1]\n", 4),
        (_MATCH, 4),
        ("def f(x):\n    with open(x):\n        assert x\n", 3),
        ("def f(x):\n    return (lambda y: 1 if y else 2)(x)\n", 2),
        (_NESTED, 1),
    ],
    ids=[
        "plain",
        "if-elif-else",
        "loops",
        "excepts",
        "boolean",
        "comprehension",
        "match",
        "with-assert",
        "lambda",
        "nested",
    ],
)
def test_complexity_counts_radons_decisions_in_the_functions_own_body(source, expected):
    assert _complexity(source) == expected


@pytest.mark.parametrize(
    ("cc", "covered", "expected"),
    [(10, 1.0, 10.0), (10, 0.0, 110.0), (5, 0.5, 8.125), (1, 0.0, 2.0)],
)
def test_crap_is_complexity_squared_times_uncovered_cubed_plus_complexity(cc, covered, expected):
    assert crap.crap(cc, covered) == expected


def _report(tmp_path, regions: dict, source: str):
    """A tree with one module and a coverage.py JSON report for its functions."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(source)
    functions = {"": {"start_line": 1, "summary": {"percent_covered": 100.0}}} | {
        name: {"start_line": line, "summary": {"percent_covered": percent}}
        for name, (line, percent) in regions.items()
    }
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {"src/m.py": {"functions": functions}}}))
    return report


# cc 3, and 50% covered: 9 * 0.125 + 3 = 4.125.
_SOURCE = "def f(a, b):\n    if a:\n        return 1\n    return a or b\n"


def test_a_function_at_the_limit_passes(tmp_path, capsys):
    report = _report(tmp_path, {"f": (1, 50.0)}, _SOURCE)

    assert crap.main([str(report), "--max", "4.125", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "    4.1  cc  3   50.0%  src/m.py:1 f",
        "crap: 1 functions, none over 4.125",
    ]


def test_a_function_over_the_limit_fails(tmp_path, capsys):
    report = _report(tmp_path, {"f": (1, 50.0)}, _SOURCE)

    assert crap.main([str(report), "--max", "4.1", "--root", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "crap: 1 of 1 functions over 4.1"


def test_a_region_the_sources_do_not_have_stops_the_check(tmp_path, capsys):
    """Coverage and the sources disagree: a score would be wrong, so there is none."""
    report = _report(tmp_path, {"f": (1, 50.0), "gone": (9, 0.0)}, _SOURCE)

    assert crap.main([str(report), "--max", "100", "--root", str(tmp_path)]) == 2
    assert capsys.readouterr().err.splitlines() == [
        "crap: coverage names functions the sources do not have:",
        "  src/m.py:9 gone",
    ]


def test_a_missing_report_stops_the_check(tmp_path, capsys):
    assert crap.main([str(tmp_path / "none.json"), "--max", "29"]) == 2
    assert capsys.readouterr().err.startswith(f"crap: cannot read {tmp_path / 'none.json'}")
