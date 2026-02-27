# Pranav Self-Learning Assistant (Automatic Memory + Retraining)

This adds an always-running local loop that can:

1. Remember past interactions immediately (memory retrieval).
2. Log every interaction to local storage.
3. Auto-promote good interactions into training candidates.
4. Auto-retrain in the background after enough new data.
5. Auto-evaluate and auto-promote a newly trained adapter if it passes.

If you want the easiest user flow through Ollama model selection, use `README_PRANAV_OLLAMA_AUTOPILOT.md`.

## What Runs

- Script: `scripts/run_self_learning_assistant.py`
- Memory DB: `runtime/self_learning/memory.db`
- Interaction log: `runtime/self_learning/interactions.jsonl`
- Auto-built train file: `runtime/self_learning/auto_training_dataset.jsonl`

## Start It

```powershell
.\.venv311\Scripts\python.exe scripts\run_self_learning_assistant.py `
  --adapter-dir outputs\pranav_8gb\pranav_lora `
  --base-dataset data\pranav_profile_qa.jsonl `
  --retrain-every 25 `
  --train-steps 120
```

## Commands While Chatting

- `/approve-last` : force-approve and promote last response
- `/bad-last` : mark last response bad and remove from training candidates
- `/stats` : show memory/candidate counts
- `/reload` : reload the current production adapter from disk
- `exit` : stop

## How “Automatic” Works

1. Each chat turn is scored with a quality heuristic.
2. High-score turns are auto-promoted for training.
3. When newly promoted count reaches `--retrain-every`, background retrain starts.
4. New adapter is trained to `outputs/auto_stage/pranav_lora`.
5. Smoke eval runs.
6. If eval passes (including strict `BTS7960` and exact `CPR = PPR x 4` checks), it replaces `outputs/pranav_8gb/pranav_lora` automatically.

## Important Notes

- This is powerful but can drift if bad answers are promoted repeatedly.
- Use `/bad-last` on low-quality responses.
- Keep occasional manual review of `runtime/self_learning/interactions.jsonl`.
- For safer behavior, increase `--retrain-every` (for example, 40-60).
