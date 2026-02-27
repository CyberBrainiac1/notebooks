# Pranav Guide: What You Can Build With These Models

This repo includes many ready-to-run notebooks for fine-tuning and inference across LLM, vision, OCR, speech, embedding, and RL workflows.

If you want a personal profile model that runs on an 8GB GPU, use [README_PRANAV_MODEL_8GB.md](README_PRANAV_MODEL_8GB.md).
If you want automatic memory + auto-retraining from new interactions, use [README_PRANAV_SELF_LEARNING.md](README_PRANAV_SELF_LEARNING.md).
If you want Ollama-first autopilot (select model once, then automatic memory + retraining), use [README_PRANAV_OLLAMA_AUTOPILOT.md](README_PRANAV_OLLAMA_AUTOPILOT.md).
If you want no-retrain vector memory (recommended for fast profile learning), use [README_PRANAV_VECTOR_MEMORY.md](README_PRANAV_VECTOR_MEMORY.md).

## What You Can Build

1. Chat assistants for support, education, and internal Q&A
2. Reasoning agents trained with RL/GRPO for stronger chain-of-thought style behavior
3. Vision assistants for image understanding, captioning, and multimodal chat
4. OCR pipelines for invoices, forms, scanned PDFs, and document extraction
5. Voice products with TTS + STT (speak/listen assistants, call bots, voice notes)
6. RAG systems using embedding models for semantic search and retrieval
7. Code copilots and tool-calling assistants for developer workflows
8. Lightweight on-device/mobile demos with smaller 0.5B to 1B class models

## Good Starting Notebook Paths

- Conversational LLM: `nb/Llama3.1_(8B)-Alpaca.ipynb`, `nb/Qwen3_(14B)-Reasoning-Conversational.ipynb`
- RL/GRPO: `nb/Qwen3_(4B)-GRPO.ipynb`, `nb/Phi_4_(14B)-GRPO.ipynb`
- Vision: `nb/Qwen3_VL_(8B)-Vision.ipynb`, `nb/Llama3.2_(11B)-Vision.ipynb`
- OCR: `nb/Deepseek_OCR_(3B).ipynb`, `nb/Paddle_OCR_(1B)_Vision.ipynb`
- Embeddings/RAG: `nb/Qwen3_Embedding_(4B).ipynb`, `nb/BGE_M3.ipynb`, `nb/All_MiniLM_L6_v2.ipynb`
- Speech: `nb/Whisper.ipynb`, `nb/Orpheus_(3B)-TTS.ipynb`, `nb/Spark_TTS_(0_5B).ipynb`
- Tool calling / coding: `nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb`, `nb/CodeGemma_(7B)-Conversational.ipynb`
- Mobile/on-device: `nb/Qwen3_(0_6B)-Phone_Deployment.ipynb`, `nb/Gemma3_(270M)_Phone_Deployment.ipynb`

## Quick Project Ideas

1. Company Knowledge Bot: embeddings + conversational model + RAG
2. Resume and Invoice Parser: OCR + structured extraction model
3. Voice Study Buddy: Whisper STT + LLM + TTS
4. Visual Product Assistant: vision model + product catalog retrieval
5. Coding Helper: code model + tool calling for local commands

## Suggested Workflow

1. Pick one use case and one notebook
2. Prepare domain-specific data (small but clean beats large and noisy)
3. Fine-tune with LoRA/QLoRA in Colab/Kaggle
4. Evaluate with real prompts from your target users
5. Package for app/API deployment

## Attribution (Keep This in Your Repo)

This project is derived from the original work by **Unsloth**:

- Original repository: https://github.com/unslothai/notebooks
- Unsloth main project: https://github.com/unslothai/unsloth
- License: see [`LICENSE`](LICENSE) in this repository

Recommended credit line for your README:

`Based on notebook templates and workflows from unslothai/notebooks. Full credit to the Unsloth team and original contributors.`
