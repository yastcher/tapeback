#!/usr/bin/env bash
# Run the INSTALLED tapeback-llm-local: its llama-server starts on this system, and
# tapeback finds it where the package put it.
#
#     tests/smoke/llm-local.sh /opt/tapeback/venv/bin/llama-server   # the .deb
#     tests/smoke/llm-local.sh /usr/bin/llama-server                 # Arch
#
# `--version` loads every library the server links against, so a release built for a
# newer glibc or a missing dependency fails here. No model is loaded: the stands have
# no GPU, and the pinned one is 2.7 GB.
set -euo pipefail

server="$1"

"$server" --version

line=$(TAPEBACK_LLM_PROVIDER=local tapeback status | grep '^LLM: ')
echo "$line"
case "$line" in
  "LLM: local, model Qwen3.5-4B-Q4_K_M.gguf, server "*) ;;
  *) echo "smoke: unexpected status line" >&2; exit 1 ;;
esac
# By its real path: on Arch /usr/sbin, which PATH lists first, is a link to /usr/bin.
shown="${line##*, server }"
if [ "$(realpath "$shown")" != "$(realpath "$server")" ]; then
  echo "smoke: tapeback found $shown, not $server" >&2
  exit 1
fi

echo "smoke: tapeback finds llama-server at $shown, and it runs"
