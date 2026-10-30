"""The architecture AGENTS.md describes, held by tests instead of by review.

Each rule is a function over a repository tree that returns its violations. It runs
once on this repository, where it must find none, and once on a small tree built to
break it, where it must find exactly that break: a rule that cannot fail proves nothing.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from tests.fixtures import REPO_ROOT

PACKAGE = Path("src/tapeback")
# Importing the CLI must stay fast and must work without the extras: these load seconds
# of native code, and `tapeback --help` or `status` needs none of them.
HEAVY = frozenset(
    {
        "anthropic",
        "av",
        "ctranslate2",
        "dbus_next",
        "faster_whisper",
        "numpy",
        "onnxruntime",
        "openai",
        "pyannote",
        "scipy",
        "torch",
        "torchaudio",
    }
)
# The two modules that talk to an LLM vendor's SDK (ADR-010, ADR-011).
SDK_USERS = frozenset({"_llm.py", "_stt_openai.py"})
SDKS = frozenset({"anthropic", "openai"})
# What sends a transcript out; only the masking seam may call it (ADR-009).
SENDERS = frozenset({"connected", "call"})
MODELS = frozenset({"Segment", "Word", "DiarizationSegment", "Summary", "ActionItem"})
CONFIG_PARSERS = frozenset({"configparser", "toml", "tomli", "tomllib", "yaml"})
MAX_LINES = 500
# Files already over the limit, each capped at its size on 2026-10-29: one may shrink,
# never grow past its cap, and leaves the list once it is under the limit.
LONG_FILES = {
    "tests/fixtures.py": 1016,
    "tests/test_cli.py": 659,
    "tests/test_diarizer.py": 750,
    "tests/test_llm_local.py": 536,
    "tests/test_stt_openai.py": 663,
}


def _modules(root: Path) -> list[Path]:
    return sorted((root / PACKAGE).glob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Every module an import names, at any depth: `from tapeback import cli` names
    both `tapeback` and `tapeback.cli`; a relative import resolves inside tapeback."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(filter(None, ("tapeback", base)))
            yield node.lineno, base
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}"


def _where(root: Path, path: Path, line: int) -> str:
    return f"{path.relative_to(root)}:{line}"


def heavy_modules_loaded_by(module: str, pythonpath: Path) -> list[str]:
    """The heavy top-level modules a fresh interpreter holds after importing `module`."""
    code = (
        f"import json, sys, {module}\n"
        "print(json.dumps(sorted({m.partition('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ | {"PYTHONPATH": str(pythonpath)},
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(set(json.loads(result.stdout)) & HEAVY)


def sdk_imports_outside_their_modules(root: Path) -> list[str]:
    return [
        f"{_where(root, path, line)} imports {name}"
        for path in _modules(root)
        if path.name not in SDK_USERS
        for line, name in _imports(_parse(path))
        if name.partition(".")[0] in SDKS
    ]


def text_sent_past_the_masking_seam(root: Path) -> list[str]:
    found = []
    for path in _modules(root):
        if path.name in {"_llm.py", "summarizer.py"}:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "_llm"
                and node.attr in SENDERS
            ):
                found.append(f"{_where(root, path, node.lineno)} calls _llm.{node.attr}")
            elif isinstance(node, ast.ImportFrom) and node.module == "tapeback._llm":
                found.extend(
                    f"{_where(root, path, node.lineno)} imports _llm.{alias.name}"
                    for alias in node.names
                    if alias.name in SENDERS
                )
    return found


def domain_models_outside_models(root: Path) -> list[str]:
    return [
        f"{_where(root, path, node.lineno)} defines {node.name}"
        for path in _modules(root)
        if path.name != "models.py"
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.ClassDef) and node.name in MODELS
    ]


def modules_importing_the_cli(root: Path) -> list[str]:
    return sorted(
        {
            f"{_where(root, path, line)} imports tapeback.cli"
            for path in _modules(root)
            if path.name != "cli.py"
            for line, name in _imports(_parse(path))
            if name == "tapeback.cli"
        }
    )


def config_file_parsers(root: Path) -> list[str]:
    return [
        f"{_where(root, path, line)} imports {name}"
        for path in _modules(root)
        for line, name in _imports(_parse(path))
        if name in CONFIG_PARSERS
    ]


def _tapeback_variable(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("TAPEBACK_")
    )


def settings_read_past_settings(root: Path) -> list[str]:
    """`os.environ.get("TAPEBACK_…")`, `os.getenv(…)` or `os.environ["TAPEBACK_…"]`
    anywhere but settings.py: such a value skips validation and the run record."""
    found = []
    for path in _modules(root):
        if path.name == "settings.py":
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call):
                read = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"get", "getenv", "pop", "setdefault"}
                    and bool(node.args)
                    and _tapeback_variable(node.args[0])
                )
            elif isinstance(node, ast.Subscript):
                read = _tapeback_variable(node.slice)
            else:
                continue
            if read:
                found.append(f"{_where(root, path, node.lineno)} reads a TAPEBACK_ variable")
    return found


def files_over_the_line_limit(root: Path, allowed: dict[str, int]) -> list[str]:
    found = []
    for directory in ("src", "scripts", "tests"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in {".py", ".sh"}:
                continue
            name = str(path.relative_to(root))
            lines = len(path.read_text(encoding="utf-8").splitlines())
            cap = allowed.get(name)
            if cap is None and lines > MAX_LINES:
                found.append(f"{name} has {lines} lines, over {MAX_LINES}")
            elif cap is not None and lines > cap:
                found.append(f"{name} has {lines} lines, over its cap of {cap}")
            elif cap is not None and lines <= MAX_LINES:
                found.append(f"{name} has {lines} lines: delete its entry from LONG_FILES")
    return found


def modules_agents_md_does_not_list(root: Path) -> list[str]:
    """AGENTS.md's Architecture section names every module; the two copies agree."""
    bullets = [
        line
        for line in (root / "AGENTS.md").read_text(encoding="utf-8").splitlines()
        if line.startswith(("- Source: `src/tapeback/`", "- Private helpers are `_`-prefixed"))
    ]
    named = set(re.findall(r"\b(_?[a-z][a-z0-9_]*\.py)\b", " ".join(bullets)))
    files = {path.name for path in _modules(root)} - {"__init__.py"}
    return [f"{name} is missing from AGENTS.md" for name in sorted(files - named)] + [
        f"AGENTS.md names {name}, which does not exist" for name in sorted(named - files)
    ]


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _package(root: Path, **modules: str) -> Path:
    return _tree(root, {f"src/tapeback/{name}.py": text for name, text in modules.items()})


# --- this repository ---


def test_a_tests_temporary_files_live_under_the_repositorys_tmp(tmp_path):
    """Development writes to ./tmp, never /tmp: on some machines /tmp is a small tmpfs
    with a per-user quota, and a full one makes every command fail."""
    assert tmp_path.is_relative_to(REPO_ROOT / "tmp")
    assert Path(tempfile.gettempdir()).is_relative_to(REPO_ROOT / "tmp")


def test_importing_the_cli_loads_no_heavy_module():
    assert heavy_modules_loaded_by("tapeback.cli", REPO_ROOT / "src") == []


def test_the_rules_find_nothing_in_this_repository():
    assert sdk_imports_outside_their_modules(REPO_ROOT) == []
    assert text_sent_past_the_masking_seam(REPO_ROOT) == []
    assert domain_models_outside_models(REPO_ROOT) == []
    assert modules_importing_the_cli(REPO_ROOT) == []
    assert config_file_parsers(REPO_ROOT) == []
    assert settings_read_past_settings(REPO_ROOT) == []
    assert files_over_the_line_limit(REPO_ROOT, LONG_FILES) == []
    assert modules_agents_md_does_not_list(REPO_ROOT) == []


# --- each rule catches the break it exists for ---


def test_a_heavy_import_at_module_level_is_caught(tmp_path):
    (tmp_path / "probe_module.py").write_text("import numpy\n")

    assert heavy_modules_loaded_by("probe_module", tmp_path) == ["numpy"]


def test_an_sdk_imported_elsewhere_is_caught(tmp_path):
    root = _package(tmp_path, _llm="import openai\n", pipeline="def f():\n    import anthropic\n")

    assert sdk_imports_outside_their_modules(root) == [
        "src/tapeback/pipeline.py:2 imports anthropic"
    ]


def test_text_sent_past_summarize_is_caught(tmp_path):
    root = _package(
        tmp_path,
        summarizer="from tapeback import _llm\n_llm.connected(None)\n",
        live="from tapeback import _llm\n_llm.call('s', 'u', None)\n",
        cli="from tapeback._llm import connected\n",
    )

    assert sorted(text_sent_past_the_masking_seam(root)) == [
        "src/tapeback/cli.py:1 imports _llm.connected",
        "src/tapeback/live.py:2 calls _llm.call",
    ]


def test_a_domain_model_outside_models_is_caught(tmp_path):
    root = _package(tmp_path, models="class Segment: ...\n", formatter="\n\nclass Word: ...\n")

    assert domain_models_outside_models(root) == ["src/tapeback/formatter.py:3 defines Word"]


def test_a_module_importing_the_cli_is_caught(tmp_path):
    root = _package(tmp_path, cli="", tray="from tapeback import cli\n", live="from . import cli\n")

    assert modules_importing_the_cli(root) == [
        "src/tapeback/live.py:1 imports tapeback.cli",
        "src/tapeback/tray.py:1 imports tapeback.cli",
    ]


def test_a_config_file_parser_is_caught(tmp_path):
    root = _package(tmp_path, vault="import tomllib\n")

    assert config_file_parsers(root) == ["src/tapeback/vault.py:1 imports tomllib"]


def test_a_tapeback_variable_read_past_settings_is_caught(tmp_path):
    root = _package(
        tmp_path,
        settings="import os\nos.environ.get('TAPEBACK_VAULT_PATH')\n",
        audio=(
            "import os\n"
            "os.getenv('TAPEBACK_LANGUAGE')\n"
            "os.environ['TAPEBACK_DEVICE']\n"
            "os.environ.get('PATH')\n"
        ),
    )

    assert settings_read_past_settings(root) == [
        "src/tapeback/audio.py:2 reads a TAPEBACK_ variable",
        "src/tapeback/audio.py:3 reads a TAPEBACK_ variable",
    ]


def test_the_line_limit_holds_new_files_and_caps(tmp_path):
    over, at = "x\n" * 501, "x\n" * 500
    root = _tree(
        tmp_path,
        {
            "src/tapeback/new.py": over,
            "scripts/fine.sh": at,
            "tests/capped.py": "x\n" * 600,
            "tests/grown.py": "x\n" * 601,
            "tests/shrunk.py": at,
        },
    )
    caps = {"tests/capped.py": 600, "tests/grown.py": 600, "tests/shrunk.py": 600}

    assert files_over_the_line_limit(root, caps) == [
        "src/tapeback/new.py has 501 lines, over 500",
        "tests/grown.py has 601 lines, over its cap of 600",
        "tests/shrunk.py has 500 lines: delete its entry from LONG_FILES",
    ]


def test_a_module_missing_from_agents_md_is_caught(tmp_path):
    root = _package(tmp_path, cli="", recorder="", _gpu="")
    (root / "AGENTS.md").write_text(
        "- Source: `src/tapeback/` — cli.py, gone.py\n"
        "- Private helpers are `_`-prefixed: `_gpu.py`\n"
    )

    assert modules_agents_md_does_not_list(root) == [
        "recorder.py is missing from AGENTS.md",
        "AGENTS.md names gone.py, which does not exist",
    ]
