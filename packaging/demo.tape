Output docs/demo.gif

Set FontSize 14
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

Type "# tapeback — record and transcribe meetings locally"
Enter
Sleep 1s

Type "export TAPEBACK_VAULT_PATH=~/vault"
Enter
Sleep 500ms

Type "tapeback start weekly-standup"
Enter
Sleep 2s

Type "# Recording... (simulated)"
Enter
Sleep 1s

# Simulate Ctrl+C
Ctrl+C
Sleep 1s

Type "# Transcribing (this may take a few minutes)..."
Enter
Sleep 2s

Type "# Saved: ~/vault/meetings/weekly-standup.md"
Enter
Sleep 1s

Type "cat ~/vault/meetings/weekly-standup.md | head -30"
Enter
Sleep 3s
