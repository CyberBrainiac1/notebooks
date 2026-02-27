# Train PranavProfileGPT (8GB VRAM)

This setup fine-tunes a small 4-bit base model with LoRA so it runs on an 8GB GPU and 36GB RAM.

## Files Added

- `scripts/build_pranav_profile_dataset.py`
- `scripts/train_pranav_8gb.py`
- `scripts/chat_pranav_8gb.py`

## 1) Environment (Windows, Python 3.11)

Use Python 3.11 (not 3.13/3.14) for best package compatibility.
For RTX 50-series GPUs (like your 5060), use CUDA 12.8 nightly Torch builds.

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --no-deps trl peft accelerate bitsandbytes
pip install unsloth==2026.2.1 unsloth-zoo==2026.2.1 triton-windows==3.6.0.post25
pip install datasets==4.3.0 transformers==4.56.2
```

If Unsloth import issues appear, check the Known Issues section in `README.md` (`numpy<2`, typing_extensions).

## 2) Build Dataset

```powershell
.\.venv311\Scripts\python.exe scripts/build_pranav_profile_dataset.py --output data/pranav_profile_qa.jsonl
```

## 3) Train (8GB-friendly defaults)

```powershell
.\.venv311\Scripts\python.exe scripts/train_pranav_8gb.py `
  --dataset-path data/pranav_profile_qa.jsonl `
  --output-dir outputs/pranav_8gb `
  --model-name unsloth/Llama-3.2-1B-Instruct-bnb-4bit `
  --max-seq-length 1024 `
  --batch-size 1 `
  --grad-accum 8 `
  --max-steps 250 `
  --learning-rate 2e-4
```

Recommended stronger run:

- `--max-steps 600` for better memory of profile details.
- Add `--save-merged-16bit` if you want to export directly to Ollama later.

## 4) Chat with Your Model

```powershell
.\.venv311\Scripts\python.exe scripts/chat_pranav_8gb.py --adapter-dir outputs/pranav_8gb/pranav_lora
```

Type `exit` to quit.

## 5) Make It Work In Ollama (One Command)

If you finished training and have a merged model (`.safetensors`) available, run:

```powershell
.\.venv311\Scripts\python.exe scripts/setup_pranav_ollama.py --model-name pranav-assistant
```

Then chat with it:

```powershell
ollama run pranav-assistant
```

Notes:

- Script auto-detects merged model folders from:
  - `outputs/pranav_gguf`
  - `outputs/pranav_8gb/pranav_merged_16bit`
  - `outputs/pranav_8gb/pranav_merged`
- It converts to GGUF, writes a Modelfile, and runs `ollama create --quantize q4_K_M`.

## Notes

- This produces a LoRA adapter (`outputs/pranav_8gb/pranav_lora`) instead of a full base model copy.
- For your hardware target, this is the best balance of fit, speed, and personalization quality.
