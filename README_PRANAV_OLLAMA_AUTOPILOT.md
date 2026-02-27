# Ollama Autopilot (Select Model, Everything Else Automatic)

This is the easiest mode.

You run one command, pick a model number, then the system:

1. Chats through Ollama.
2. Stores memory from past interactions.
3. Logs every interaction.
4. Auto-promotes high-quality interactions to training data.
5. Auto-retrains in background after enough new data.
6. Auto-builds a new Ollama model from merged weights (GGUF path).
7. Auto-switches to the new model if eval passes.

## One Command

```powershell
.\.venv311\Scripts\python.exe scripts\ollama_autopilot.py --auto-start-ollama
```

Then pick the model number from the list.

Or easiest on Windows: double-click `start_ollama_autopilot.bat`.

## Requirements

- Ollama installed.
- A supported Ollama model pulled:
  - `llama3.2:1b` or `llama3.2:3b`

If none are installed:

```powershell
ollama pull llama3.2:3b
```

## In-Chat Commands

- `/approve-last` : force-approve last answer for training
- `/bad-last` : mark last answer bad so it is excluded
- `/stats` : show memory and candidate counts
- `/model` : show current live model
- `exit` : quit

## Defaults (safe)

- Auto-retrain every `40` promoted interactions
- Retrain steps per cycle: `0` (auto-sized from dataset volume for efficiency)
- Min quality for auto-promotion: `0.90`
- Quantization for rollout model: `q4_K_M`
- Eval gate: `7` checks minimum (includes typo checks + strict `BTS7960` and `CPR = PPR x 4`)

You can tune:

```powershell
.\.venv311\Scripts\python.exe scripts\ollama_autopilot.py `
  --auto-start-ollama `
  --retrain-every 50 `
  --train-steps 0 `
  --min-quality 0.92 `
  --eval-min-pass 7
```

## Runtime Files

- `runtime/ollama_autopilot/memory.db`
- `runtime/ollama_autopilot/interactions.jsonl`
- `runtime/ollama_autopilot/auto_training_dataset.jsonl`

## Notes

- Background retraining uses `scripts/train_pranav_8gb.py`.
- Rollout conversion/build uses `scripts/setup_pranav_ollama.py`.
- For best stability, keep using Llama 3.2 1B/3B in Ollama.
