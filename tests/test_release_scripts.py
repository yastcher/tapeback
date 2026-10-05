"""A release tag must agree with every copy of the version, and carry its notes."""

import re

import pytest

from tests.fixtures import REPO_ROOT, load_script

release_from_tag = load_script("release_from_tag")


def test_matching_tag_yields_the_section_as_notes(release_tree):
    assert release_from_tag.check(release_tree, "v1.2.3") == "### Fixed\n- The fix."


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v1.2.3-rc1"])
def test_anything_but_vXYZ_is_not_a_release_tag(release_tree, tag):
    with pytest.raises(SystemExit, match="not a release tag"):
        release_from_tag.check(release_tree, tag)


@pytest.mark.parametrize(
    ("path", "old", "new", "message"),
    [
        (
            "pyproject.toml",
            'version = "1.2.3"',
            'version = "1.2.4"',
            r"pyproject\.toml \(1\.2\.4\)",
        ),
        (
            "uv.lock",
            'name = "tapeback"\nversion = "1.2.3"',
            'name = "tapeback"\nversion = "1.2.2"',
            r"uv\.lock \(1\.2\.2\)",
        ),
        (
            "packaging/tapeback-llm/PKGBUILD",
            "pkgver=1.2.3",
            "pkgver=1.2.2",
            r"packaging/tapeback-llm/PKGBUILD \(1\.2\.2\)",
        ),
    ],
    ids=["pyproject", "uv.lock", "PKGBUILD"],
)
def test_one_stale_copy_of_the_version_refuses_the_tag(release_tree, path, old, new, message):
    target = release_tree / path
    target.write_text(target.read_text().replace(old, new))

    with pytest.raises(SystemExit, match=message):
        release_from_tag.check(release_tree, "v1.2.3")


def test_dependency_on_the_same_number_is_not_read_as_the_project_version(release_tree):
    lock = release_tree / "uv.lock"
    lock.write_text(lock.read_text().replace('version = "8.1.7"', 'version = "9.9.9"'))

    assert release_from_tag.locked_version(release_tree) == "1.2.3"


def test_tag_without_a_changelog_section_is_refused(release_tree):
    changelog = release_tree / "CHANGELOG.md"
    changelog.write_text(changelog.read_text().replace("## [1.2.3] — 2026-10-05", "## [1.2.4]"))

    with pytest.raises(SystemExit, match=r"no section for 1\.2\.3"):
        release_from_tag.check(release_tree, "v1.2.3")


def test_tag_with_an_empty_changelog_section_is_refused(release_tree):
    changelog = release_tree / "CHANGELOG.md"
    changelog.write_text(changelog.read_text().replace("### Fixed\n- The fix.\n\n", ""))

    with pytest.raises(SystemExit, match=r"section for 1\.2\.3 is empty"):
        release_from_tag.check(release_tree, "v1.2.3")


# --- the repository itself ---


def test_every_copy_of_the_version_agrees():
    """Only scripts/release.sh writes the version, and it writes all copies at once.
    A feature branch that bumps pyproject.toml alone fails here, in its own PR."""
    declared = release_from_tag.declared_version(REPO_ROOT)

    assert set(release_from_tag.pkgbuild_versions(REPO_ROOT).values()) == {declared}


def test_pre_release_smoke_installs_into_the_images_ci_does():
    """Two copies of one list drift apart silently; this one is read from both sides."""
    qa = (REPO_ROOT / "scripts" / "pre_release_qa.sh").read_text()
    ci = (REPO_ROOT / ".github" / "workflows" / "deb-e2e.yml").read_text()

    qa_images = re.search(r"^SMOKE_IMAGES=\((?P<images>[^)]*)\)$", qa, re.M)
    ci_images = re.search(r"^\s+image:\n(?P<images>(?:\s+- .+\n)+)", ci, re.M)

    assert qa_images is not None
    assert ci_images is not None
    assert qa_images["images"].split() == re.findall(r"- (\S+)", ci_images["images"])
    assert qa_images["images"].split() == [
        "ubuntu:26.04",
        "ubuntu:24.04",
        "ubuntu:22.04",
        "debian:13",
        "debian:12",
    ]
