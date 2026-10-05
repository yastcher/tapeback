"""CHANGELOG entries land in the open section, and a release closes it cleanly."""

import datetime
import re

import pytest

from tests.fixtures import REPO_ROOT, load_script

changelog_add = load_script("changelog_add")
changelog_release = load_script("changelog_release")

OPEN = """# Changelog

## [Unreleased]

### Added
- **A.** First.

### Fixed
- **F.** First.

## [0.9.8] — 2026-08-05

### Added
- **Old.** Shipped.
"""


def _add(subsection: str, entry: str = "- **New.** Entry.", **kwargs) -> str:
    return changelog_add.add_entry(
        OPEN, subsection=subsection, entry=entry, released={"0.9.8"}, **kwargs
    )


# --- changelog_add ---


def test_entry_is_appended_to_its_subsection_as_a_tight_list():
    result = _add("Added")

    assert "### Added\n- **A.** First.\n- **New.** Entry.\n\n### Fixed" in result


def test_top_puts_the_entry_first_in_its_subsection():
    result = _add("Fixed", first=True)

    assert "### Fixed\n- **New.** Entry.\n- **F.** First.\n\n## [0.9.8]" in result


def test_missing_subsection_is_created_in_documented_order():
    changed = _add("Changed")
    security = _add("Security")
    docs = _add("Docs")

    assert "- **A.** First.\n\n### Changed\n- **New.** Entry.\n\n### Fixed" in changed
    assert "## [Unreleased]\n\n### Security\n- **New.** Entry.\n\n### Added" in security
    assert "- **F.** First.\n\n### Docs\n- **New.** Entry.\n\n## [0.9.8]" in docs


def test_first_entry_of_a_cycle_lands_below_the_hint_comments():
    fresh = "## [Unreleased]\n\n<!-- hint -->\n\n## [0.9.8] — 2026-08-05\n"

    result = changelog_add.add_entry(
        fresh, subsection="Fixed", entry="- **New.** Entry.", released={"0.9.8"}
    )

    assert result == (
        "## [Unreleased]\n\n<!-- hint -->\n\n### Fixed\n- **New.** Entry.\n\n"
        "## [0.9.8] — 2026-08-05\n"
    )


def test_released_sections_are_never_touched():
    result = _add("Added")

    assert result.split("## [0.9.8]")[1] == OPEN.split("## [0.9.8]")[1]


def test_entry_into_a_section_a_tag_closed_is_refused():
    shipped = OPEN.replace("## [Unreleased]", "## [0.9.9] — 2026-09-18")

    with pytest.raises(SystemExit, match=r"tag v0\.9\.9 has already closed"):
        changelog_add.add_entry(
            shipped, subsection="Added", entry="- x", released={"0.9.8", "0.9.9"}
        )


def test_changelog_without_sections_is_refused():
    with pytest.raises(SystemExit, match="no release sections"):
        changelog_add.add_entry("# Changelog\n", subsection="Added", entry="- x", released=set())


# --- changelog_release ---

TODAY = datetime.date(2026, 10, 5)


def test_release_names_the_open_section_and_opens_a_fresh_one():
    text = OPEN.replace(
        "## [Unreleased]\n", "## [Unreleased]\n\n" + "\n".join(changelog_release.SCAFFOLD) + "\n"
    )

    result = changelog_release.close_unreleased(text, version="0.9.9", today=TODAY)

    assert result.startswith(
        "# Changelog\n\n## [Unreleased]\n\n"
        "<!-- New entries go HERE, via scripts/changelog_add.py"
        " — never under a dated release. -->\n"
        "<!-- Subsections in order: Security / Added / Changed / Fixed / Removed / Docs. -->\n\n"
        "## [0.9.9] — 2026-10-05\n\n### Added\n- **A.** First.\n"
    )
    # The hint comments go with [Unreleased], never into the published notes.
    assert result.count("<!--") == 2


def test_release_of_an_empty_section_is_refused():
    empty = "# Changelog\n\n## [Unreleased]\n\n<!-- hint -->\n\n## [0.9.8] — 2026-08-05\n\n- x\n"

    with pytest.raises(SystemExit, match="empty"):
        changelog_release.close_unreleased(empty, version="0.9.9", today=TODAY)


def test_allow_empty_releases_a_section_with_no_entries():
    empty = "# Changelog\n\n## [Unreleased]\n\n## [0.9.8] — 2026-08-05\n\n- x\n"

    result = changelog_release.close_unreleased(
        empty, version="0.9.9", today=TODAY, allow_empty=True
    )

    assert "## [0.9.9] — 2026-10-05\n\n## [0.9.8]" in result


def test_release_of_an_existing_version_is_refused():
    with pytest.raises(SystemExit, match=r"already has a section for 0\.9\.8"):
        changelog_release.close_unreleased(OPEN, version="0.9.8", today=TODAY)


def test_release_without_an_unreleased_section_is_refused():
    with pytest.raises(SystemExit, match="no `## \\[Unreleased\\]` section"):
        changelog_release.close_unreleased(
            OPEN.replace("## [Unreleased]", "## [0.9.9] — 2026-09-18"),
            version="0.9.10",
            today=TODAY,
        )


# --- the repository itself ---


def test_the_top_changelog_section_is_unreleased():
    """Between releases entries go under [Unreleased]; a feature branch never picks a number."""
    first = re.search(r"^## .*$", (REPO_ROOT / "CHANGELOG.md").read_text(), re.M)

    assert first is not None
    assert first.group(0) == "## [Unreleased]"
