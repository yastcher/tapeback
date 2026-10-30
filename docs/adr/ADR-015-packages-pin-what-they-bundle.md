# ADR-015 — Distro packages pin what they bundle

**Status:** accepted 2026-05-21 for the interpreter, 2026-10-16 for llama.cpp
(`scripts/build-deb.sh`: `PYBS_*`, `LLAMA_CPP_*`; `packaging/`; `scripts/aur-publish.sh`).

**Context.** The `.deb` bundles a virtualenv with compiled wheels pinned to one Python
minor. A tool that fetched the interpreter once put a broken python into the package,
because its layout varies with its own version and the runner.

**Decision.**
- The `.deb`'s interpreter comes from a pinned python-build-standalone tarball URL.
- `tapeback-llm-local`: the `.deb` carries llama.cpp b11669's Vulkan build, checked by
  sha256; the AUR package depends on Arch's `llama-cpp`, with `ggml-cuda` or
  `ggml-vulkan` for the GPU; pip users put `llama-server` on PATH.
- The repository's PKGBUILD keeps `sha256sums=('SKIP')`; `aur-publish.sh` writes the
  real checksum of the tag's tarball into the AUR copy, and publishes every PKGBUILD
  under `packaging/`.

**Why.**
- A pinned URL is deterministic, and so is the archive layout.
- The Vulkan build runs on Ubuntu 22.04+ and Debian 12+ and weighs 30 MB; the CUDA
  builds need 24.04+ and weigh 25 times more.
- The tarball does not exist until the tag does.

**Consequences.** Moving a pin is a deliberate change with its checksum. What the
packaging stands prove: `docs/release-testing.md`.
