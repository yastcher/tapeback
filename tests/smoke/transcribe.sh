#!/usr/bin/env bash
# Transcribe a short recording with the INSTALLED tapeback, the way it runs after
# `tapeback start`: the stereo pipeline, the isolated Whisper worker, the note.
#
#     tests/smoke/transcribe.sh
#
# Run by the packaging stands once the package is installed. `tapeback --version` and
# `tapeback status` import tapeback but never decode audio or load a model, and the
# 0.9.9 stands passed over a PyAV release that broke exactly that. The stands resolve
# dependencies the way users get them, not from uv.lock, so this is where such a
# release shows up first.
#
# speech.wav, made with espeak-ng 1.52 at 16 kHz: the mic (left) says "Good afternoon.
# Let us review the budget.", then the monitor (right) "Good morning everyone.". Both
# are phrases tiny hears with confidence. A doubtful one sends Whisper into sampling
# at higher temperatures, and the line changes from run to run: "Hello world." came
# out as "Hello, where else?" in one container and "Thedogware, LLMv3," in the next.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vault="$(mktemp -d)"

# The smallest model, on the CPU: the stands have no GPU, and the question is whether
# the installed stack runs at all, not how well it hears.
TAPEBACK_VAULT_PATH="$vault" TAPEBACK_STT_MODEL=tiny TAPEBACK_DEVICE=cpu \
  tapeback process "$here/speech.wav" --name smoke --no-diarize --no-summarize

note="$vault/meetings/smoke.md"
cat "$note"

# One phrase per channel, under the speaker that channel belongs to: both
# transcriptions ran and reached the note. Words rather than the exact line: tiny
# hears "Go after noon, let us review the budget.", and the stands follow dependency
# releases. Neither word is one Whisper invents over silence.
grep -qiE '\*\*You:\*\* .*budget' "$note" || { echo "smoke: no mic phrase under You" >&2; exit 1; }
grep -qiE '\*\*Other:\*\* .*good morning' "$note" || { echo "smoke: no monitor phrase under Other" >&2; exit 1; }
echo "smoke: both channels transcribed"
