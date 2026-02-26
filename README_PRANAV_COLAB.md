# 🤖 Pranav's Personal AI Assistant - Google Colab Training Guide

Train your own AI model on your personal profile, projects, and knowledge using **Google Colab's free GPU**!

## 🎯 What This Does

This notebook fine-tunes **Llama 3.2 3B** on your personal information to create an AI assistant that:
- Knows about YOUR robotics projects (FTC/FRC teams, builds)
- Remembers YOUR technical preferences and skills
- Answers questions about YOUR background
- Still maintains general knowledge capabilities

**Perfect for**: Personal assistants, portfolio showcases, team knowledge bases, research projects

## 💻 Hardware Requirements

✅ **Google Colab Free Tier** (T4 GPU)
- 8GB VRAM
- 36GB RAM
- **Cost: $0** 🎉

## 📋 Quick Start

### 1. Open the Notebook in Google Colab

Click this button to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Pranav_Profile_Assistant_(3B).ipynb)

Or manually:
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `Pranav_Profile_Assistant_(3B).ipynb`

### 2. Get Your Dataset Ready

You need the dataset file `pranav_profile_qa.jsonl` from this repo:
- Located at: `/workspaces/notebooks/data/pranav_profile_qa.jsonl`
- Download it to your local computer
- Format: JSON Lines with fields: `instruction`, `input`, `output`

**Your dataset has 235 examples** about:
- FTC team Evergreen Dragons
- FRC team 2854 Prototypes
- DIY robotic hand project
- Sim racing steering wheel build
- CAD preferences (SolidWorks, Onshape)
- Code preferences (Python, Raspberry Pi)
- And much more!

### 3. Run in Colab

1. **Select GPU Runtime**:
   - Click `Runtime → Change runtime type`
   - Set `Hardware accelerator` to **GPU (T4)**
   - Click `Save`

2. **Run All Cells**:
   - Click `Runtime → Run all`
   - Or press `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)

3. **Upload Your Dataset**:
   - When prompted, click "Choose Files"
   - Upload `pranav_profile_qa.jsonl`

4. **Wait for Training** (15-30 minutes):
   - Installation: ~2 minutes
   - Training: ~15-25 minutes
   - The notebook will show progress!

## 🧪 What Happens During Training

1. **Installation** (2 min): Installs Unsloth and dependencies
2. **Model Loading** (1 min): Downloads Llama 3.2 3B (4-bit quantized)
3. **Dataset Prep** (30 sec): Formats your data for training
4. **Training** (15-25 min): Fine-tunes the model on your data
5. **Testing**: Runs sample questions to verify learning
6. **Saving**: Exports your model

## 📊 Training Details

**Model**: Llama 3.2 3B Instruct
- Size: ~3 billion parameters
- Quantization: 4-bit (saves memory)
- LoRA rank: 32 (good quality/speed balance)

**Training Config**:
- Batch size: 2
- Gradient accumulation: 4 steps (effective batch = 8)
- Training steps: 300 (takes ~20 min)
- Learning rate: 2e-4
- Optimizer: AdamW 8-bit

**Memory Usage**:
- Peak VRAM: ~6-7 GB (fits in 8GB!)
- RAM: ~12-15 GB (fits in free Colab)

## 🎯 Testing Your Model

The notebook includes test questions like:

1. **"What FTC team is Pranav on?"**
   - Expected: Evergreen Dragons + leadership goals

2. **"Tell me about the sim racing wheel build"**
   - Expected: Arduino Leonardo, BTS7960, dual motors, etc.

3. **"What CAD software does Pranav use?"**
   - Expected: SolidWorks and Onshape

4. **"What is the capital of France?"** (general knowledge)
   - Expected: Paris (model retains general knowledge!)

## 💾 Saving Options

After training, you have 4 save options:

### 1. LoRA Adapters (Recommended for most users)
- **Size**: ~200 MB
- **What**: Just the fine-tuned weights
- **Use**: Download and use with base model
- **Download**: Files appear in Colab sidebar

### 2. Hugging Face Hub (Best for sharing)
- **Size**: ~200 MB (LoRA)
- **What**: Upload to your HF account
- **Use**: Load from anywhere with internet
- **Setup**: Get token at https://huggingface.co/settings/tokens

### 3. Merged 16-bit Model (Standalone)
- **Size**: ~6 GB
- **What**: Complete model with weights merged
- **Use**: No need for base model
- **Good for**: Production deployment

### 4. GGUF Format (For local tools)
- **Size**: ~2-4 GB (depends on quantization)
- **What**: Format for llama.cpp, Ollama, LM Studio
- **Use**: Local inference on CPU/GPU
- **Good for**: Desktop apps, offline use

## 🚀 Using Your Trained Model

### In Python (with Unsloth):

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "pranav_assistant_lora",  # or your HF repo
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "What's your FTC team?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, 
                                       add_generation_prompt=True, 
                                       return_tensors="pt").to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

### With Ollama (if you saved GGUF):

```bash
# 1. Create Modelfile
echo 'FROM ./pranav_assistant.gguf' > Modelfile

# 2. Import to Ollama
ollama create pranav-assistant -f Modelfile

# 3. Use it!
ollama run pranav-assistant "What FTC team are you on?"
```

### With LM Studio:

1. Open LM Studio
2. Click "Import → GGUF"
3. Select your `.gguf` file
4. Chat in the UI!

## 🔧 Customization Tips

### Train Longer for Better Results

Change in training cell:
```python
max_steps = 600,  # Increase from 300
```

More steps = stronger learning, but takes longer.

### Add More Data

Expand `pranav_profile_qa.jsonl`:
```jsonl
{"instruction": "Answer about Pranav's profile.", "input": "What's your favorite programming language?", "output": "I prefer Python, especially on Raspberry Pi."}
```

More examples = better coverage of questions!

### Adjust LoRA Rank

In LoRA cell:
```python
r = 64,  # Increase from 32 for more capacity
```

Higher rank = more learning capacity, but uses more memory.

### Change Temperature

In inference:
```python
temperature = 0.7,  # Lower = more focused, higher = more creative
```

- 0.3-0.7: Factual answers
- 0.8-1.2: More creative/varied

## 📈 Improving Your Model

### If Model Forgets General Knowledge:
- Reduce training steps
- Add general Q&A to your dataset
- Use lower learning rate

### If Model Doesn't Learn Your Data:
- Increase training steps (600-1000)
- Add more diverse examples
- Check dataset formatting

### If Out of Memory:
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 1024
- Use gradient checkpointing (already enabled)

## 🆘 Troubleshooting

**"Runtime disconnected"**
- Free Colab has time limits (~12 hours)
- Save checkpoints periodically
- Consider Colab Pro for longer sessions

**"CUDA out of memory"**
- Restart runtime: `Runtime → Restart runtime`
- Reduce batch size or sequence length
- Make sure you selected GPU runtime

**"Dataset not found"**
- Make sure you uploaded the file
- Check filename is exactly `pranav_profile_qa.jsonl`

**"Model not learning"**
- Increase training steps
- Check dataset formatting
- Verify examples match expected format

## 📚 Additional Resources

- **Unsloth Documentation**: https://unsloth.ai/docs/
- **Discord Community**: https://discord.gg/unsloth (get help!)
- **GitHub Issues**: https://github.com/unslothai/unsloth/issues
- **Example Notebooks**: https://github.com/unslothai/notebooks

## 🎓 Understanding the Training

**What is LoRA?**
- Low-Rank Adaptation: efficient fine-tuning method
- Only updates ~1-10% of model parameters
- Faster, less memory, great results!

**What is 4-bit Quantization?**
- Compresses model weights to 4 bits
- Reduces memory usage by ~75%
- Minimal quality loss

**Why Llama 3.2 3B?**
- Good balance of size/quality
- Fits in free Colab GPU
- Strong conversational abilities
- Maintains general knowledge

## 🎉 What's Next?

1. **Deploy Locally**: Use GGUF with Ollama on your computer
2. **Build an App**: Create a web UI with Gradio/Streamlit
3. **Share**: Upload to Hugging Face for others to use
4. **Expand**: Add more knowledge areas to dataset
5. **Connect**: Integrate with Discord bot, Slack, etc.

## 📝 Dataset Format

Your dataset should be JSONL (one JSON per line):

```jsonl
{"instruction": "System instruction here", "input": "User question", "output": "Assistant answer"}
{"instruction": "Answer about profile", "input": "What's your name?", "output": "Pranav Emmadi"}
```

**Fields**:
- `instruction`: System message/context
- `input`: User's question
- `output`: Expected answer

## 🤝 Contributing

Want to improve this? PRs welcome!

- Better prompts
- More test cases
- Optimization tips
- Bug fixes

## 📄 License

This notebook uses Unsloth (Apache 2.0) and Llama 3.2 (Llama 3 Community License).

---

**Made with ❤️ using [Unsloth](https://unsloth.ai)**

**Questions?** Join the [Discord](https://discord.gg/unsloth)!
