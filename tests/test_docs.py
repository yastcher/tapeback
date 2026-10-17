"""README names what exists: every LLM provider, package and setting in the code.

Prose can only be reread, but names can be checked. A provider, package or setting
README does not name fails here in its own PR, and so does a name README keeps after
the code dropped it."""

import re
from pathlib import Path
from typing import get_args

from tapeback import _llm
from tapeback.settings import LLMProvider, Settings
from tests.fixtures import REPO_ROOT

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
PROVIDER_TABLE = "| Provider | Env var | Default model |"


def _provider_rows() -> dict[str, list[str]]:
    """The provider table's rows: the provider, then the rest of its cells."""
    lines = README.splitlines()
    rows = {}
    for line in lines[lines.index(PROVIDER_TABLE) + 2 :]:
        if not line.startswith("|"):
            break
        name, *cells = (cell.strip() for cell in line.strip("|").split("|"))
        rows[name.strip("`")] = cells
    return rows


def _named_packages(command: str) -> set[str]:
    """Every package the README lines starting with `command` name."""
    return {
        name
        for args in re.findall(rf"^{re.escape(command)} ([^#\n]+)", README, re.M)
        for name in args.split()
    }


def _package_name(path: Path, field: str) -> str:
    """The package name a PKGBUILD (`pkgname=`) or an nfpm config (`name: `) declares."""
    match = re.search(rf"^{re.escape(field)}(\S+)$", path.read_text(), re.M)
    assert match is not None, f"{path} declares no {field!r}"
    return match[1]


def test_the_provider_table_has_a_row_for_every_provider():
    """`local` and `openai_compatible` once had a section of their own and no row,
    and the table is where a reader picks a provider."""
    assert sorted(_provider_rows()) == sorted(get_args(LLMProvider.__value__))


def test_the_provider_table_gives_each_cloud_providers_key_and_default_model():
    rows = _provider_rows()

    for name, provider in _llm.PROVIDERS.items():
        assert rows[name] == [f"`{provider.env_var}`", provider.default_model]


def test_readme_offers_every_aur_package_and_no_other():
    pkgbuilds = [REPO_ROOT / "packaging" / "PKGBUILD", *REPO_ROOT.glob("packaging/*/PKGBUILD")]
    built = {_package_name(p, "pkgname=") for p in pkgbuilds}

    assert _named_packages("yay -S") == built
    assert _named_packages("yay -Rc") == {"tapeback"}


def test_readme_offers_every_deb_package_and_no_other():
    configs = REPO_ROOT.glob("packaging/deb/nfpm-*.yaml")
    built = {_package_name(p, "name: ") for p in configs}
    downloaded = {
        arg.removeprefix("./").split("_")[0]
        for arg in _named_packages("sudo apt install")
        if arg.endswith(".deb")
    }

    assert downloaded == built
    assert _named_packages("sudo dpkg -r") == built


def test_the_deb_removal_takes_the_base_last():
    """The extras' removal scripts uninstall their part of the base's venv with the
    base's own pip, and dpkg removes in the order it is given."""
    line = re.search(r"^sudo dpkg -r ([^#\n]+)", README, re.M)

    assert line is not None
    assert line[1].split()[-1] == "tapeback"


def test_readme_names_every_setting_and_no_other():
    prefix = Settings.model_config["env_prefix"]
    settings = {f"{prefix}{name}".upper() for name in Settings.model_fields}

    assert set(re.findall(r"`(TAPEBACK_[A-Z0-9_]+)`", README)) == settings
