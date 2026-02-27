# Ollama Vector Memory Mode (No Retraining)

This is the fastest way to make the assistant learn your facts without running GPU-heavy retraining.

It works by:

1. Seeding memory from your profile dataset.
2. Vectorizing memories with an embedding model.
3. Retrieving the most relevant memories each turn.
4. Auto-saving good new interactions as fresh memories.

## One-Time Setup

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Run

```powershell
.\.venv311\Scripts\python.exe scripts\ollama_vector_memory_assistant.py `
  --auto-start-ollama `
  --chat-model llama3.1:8b `
  --auto-pull-chat-model `
  --auto-pull-embed-model `
  --base-dataset data/pranav_profile_qa_v4.jsonl
```

Select your chat model when prompted.

## Commands

- `/remember <fact>`: add a fact instantly to long-term memory
- `/stats`: show memory + interaction counts
- `/model`: show selected chat/embedding models
- `exit`: quit

## Why This Mode

- No local fine-tune required
- No repeated retraining loops
- Updates instantly as you add/approve new information
- Works well for personal profile knowledge and evolving project notes
