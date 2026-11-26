#!/usr/bin/env bash
# Tells the model to save its working state before auto-compaction wipes the details: the pair
# of after-compact.sh. Modeled on memory-core's hook of the same name.
#
# Runs on PostToolUse (an autonomous run grows the context through tool calls) and on
# UserPromptSubmit. Reminds once per approach to compaction: the marker is cleared as soon as
# the context is back under the threshold, which is what the compaction itself does.
#
# The fill is what Claude Code hands a statusline as context_window.used_percentage, recomputed
# because a hook is not given it: input + cache_creation + cache_read tokens of the last
# main-thread assistant message.

set -u

# The main models on this machine run a 1M window, and a tapeback session compacted at 966k.
# 93% leaves about 36k tokens for the save, enough for a long turn before it.
WINDOW_TOKENS=1000000
THRESHOLD_PCT=95
STATE_DIR="$HOME/.claude/state/context-alert"

# The hook runs on every tool call: a machine without jq or GNU tac goes without the reminder
# rather than reporting a hook error each time.
command -v jq >/dev/null && command -v tac >/dev/null || exit 0

input=$(cat)
event=$(jq -r '.hook_event_name // empty' <<<"$input")
session=$(jq -r '.session_id // empty' <<<"$input")
transcript=$(jq -r '.transcript_path // empty' <<<"$input")
agent=$(jq -r '.agent_id // empty' <<<"$input")

# A subagent's tool calls fire this hook too, with the main transcript's path: only agent_id
# tells them apart, and without it a subagent would take the main thread's one reminder.
if [ -n "$agent" ] || [ -z "$event" ] || [ -z "$session" ] || [ ! -f "$transcript" ]; then
  exit 0
fi

# Reading from the end finds the latest answer at once, even in a transcript of hundreds of MB.
# fromjson? skips a line still being written instead of ending the parse on it.
used=$(tac "$transcript" | grep -m 20 -F '"usage":{' | jq -rR '
  fromjson?
  | select(.type == "assistant" and (.isSidechain | not))
  | .message.usage
  | (.input_tokens // 0) + (.cache_creation_input_tokens // 0) + (.cache_read_input_tokens // 0)
  | select(. > 0)' | head -n 1)
[ -n "$used" ] || exit 0

pct=$((used * 100 / WINDOW_TOKENS))
marker="$STATE_DIR/$session"

if [ "$pct" -lt "$THRESHOLD_PCT" ]; then
  rm -f "$marker"
  exit 0
fi

mkdir -p "$STATE_DIR"
# noclobber makes the create atomic: a batch of parallel tool calls must remind once.
(set -o noclobber; : >"$marker") 2>/dev/null || exit 0

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  reminder="Context is ${pct}% full: auto-compaction is near, and this session's details will be lost. There is no long memory in the cloud: commit what is finished, and write into the PR description where you stopped and what comes next. Then carry on."
else
  reminder="Context is ${pct}% full: auto-compaction is near, and this session's details will be lost. Save now, in long memory: rewrite the named note tapeback:state-of-work whole (the date, the task, what is done and how it was checked, what is left, the next step), and write each user decision or working recipe of this session as its own named note tapeback:<topic>, which the state only points to. The workspace id comes from memory_workspaces; AGENTS.md, \"Long memory\", has the rest. Then carry on."
fi

jq -n --arg event "$event" --arg context "$reminder" \
  --arg notice "Context ${pct}%: the model was reminded to save its state before compaction." \
  '{systemMessage: $notice, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
