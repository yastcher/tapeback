#!/usr/bin/env bash
# After a compaction the session holds only a summary, and a summary loses details. This puts the
# order of recall into the model's context once the compacted session starts: the pair of
# context-alert.sh, which asks for the state-of-work note before the compaction. Modeled on
# memory-core's hook of the same name.
#
# Runs on SessionStart with the `compact` matcher. It names no workspace id and no server: the
# repository is public, and memory_workspaces gives the ids.

set -u

command -v jq >/dev/null || exit 0

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  reminder="The context was just compacted, and a summary loses details. There is no long memory in the cloud: before the first action, reread the task's spec, your branch's git log and the PR description."
else
  read -r -d '' reminder <<'EOF'
The context was just compacted, and a summary loses details. Before the first action:
1. memory_workspaces: the ids of the tapeback and commons workspaces.
2. memory_recent_notes(workspace_id=<tapeback>): the latest tapeback:state-of-work, with where work stopped, what is committed and what comes next.
3. One or two memory_search calls on the current step's topic, narrowed to <tapeback>; how something is done or where it lives is in the named notes tapeback:<topic>.
4. When the task reaches beyond tapeback: memory_search in <commons> with the task's words; open questions addressed to tapeback: memory_recent_notes(workspace_id=<commons>, limit=10).
5. Before an edit, reread the spec's lines and the files themselves, not the summary.
Whatever had to be rebuilt without the memory is written as a named note tapeback:<topic> in the same turn.
From here on, tapeback:state-of-work is rewritten whole at each step's boundary (a commit, a user's decision, before a compaction), and each decision or working recipe goes to its own named note tapeback:<topic> in the same turn.
EOF
fi

jq -n --arg context "$reminder" '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $context}}'
