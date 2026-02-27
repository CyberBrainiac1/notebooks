#!/usr/bin/env python3
"""
Pranav's Personal AI Assistant Training Script
Run this in Google Colab with GPU enabled!

Upload this file + pranav_profile_qa.jsonl to Colab and run.
"""

print("🚀 Starting Pranav's Personal AI Assistant Training")
print("=" * 60)

# Step 1: Install dependencies
print("\n📦 Step 1: Installing dependencies...")
print("This takes ~2 minutes...")

import subprocess
import sys
import os
import math

def install_packages():
    """Install required packages for Colab."""
    commands = [
        "pip install -q transformers==4.56.2",
        "pip install -q --no-deps trl==0.22.2",
        "pip install -q sentencepiece protobuf datasets huggingface_hub hf_transfer",
        "pip install -q --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth",
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Only install if we're missing unsloth
try:
    import unsloth
    print("✅ Dependencies already installed")
except ImportError:
    install_packages()
    print("✅ Dependencies installed!")

# Step 2: Load the model
print("\n🧠 Step 2: Loading Llama 3.2 3B model...")

import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("✅ Model loaded!")

# Step 3: Add LoRA adapters
print("\n🎯 Step 3: Adding LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("✅ LoRA adapters added!")

# Step 4: Load dataset
print("\n📂 Step 4: Loading dataset...")

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Setup chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Check if dataset exists - prefer strongest enriched dataset first
dataset_candidates = [
    "data/pranav_profile_qa_v4.jsonl",
    "data/pranav_full_training.jsonl",
    "data/pranav_profile_qa_v2.jsonl",
    "data/pranav_profile_qa.jsonl",
    "pranav_profile_qa_v4.jsonl",
    "pranav_profile_qa.jsonl",
]
dataset_file = next((p for p in dataset_candidates if os.path.exists(p)), None)
if dataset_file is None:
    print("❌ ERROR: Dataset not found!")
    print("Please ensure one of these files exists:")
    print("  - data/pranav_profile_qa_v4.jsonl (preferred, fact-locked)")
    print("  - data/pranav_full_training.jsonl")
    print("  - data/pranav_profile_qa.jsonl")
    sys.exit(1)

dataset = load_dataset("json", data_files=dataset_file, split="train")
print(f"✅ Loaded {len(dataset)} training examples from {dataset_file}")

# Dedupe for better sample efficiency
seen = set()
keep = []
for i, row in enumerate(dataset):
    key = (
        str(row["instruction"]).strip().lower(),
        str(row["input"]).strip().lower(),
        str(row["output"]).strip().lower(),
    )
    if key in seen:
        continue
    seen.add(key)
    keep.append(i)
if len(keep) != len(dataset):
    removed = len(dataset) - len(keep)
    dataset = dataset.select(keep)
    print(f"✅ Deduplicated dataset, removed {removed} rows")
print(f"✅ Effective training rows: {len(dataset)}")

# Step 5: Format dataset
print("\n🔄 Step 5: Formatting dataset...")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, user_input, output in zip(instructions, inputs, outputs):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print("✅ Dataset formatted!")

# Step 6: Setup trainer
print("\n🏋️  Step 6: Setting up trainer...")

from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

effective_batch = 2 * 4
target_epochs = 2.8
max_steps = max(140, min(460, math.ceil(len(dataset) * target_epochs / effective_batch)))
warmup_steps = max(8, min(50, int(max_steps * 0.08)))
print(f"⚙️ Auto training config -> max_steps={max_steps}, warmup_steps={warmup_steps}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=1.5e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Train only on assistant responses
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("✅ Trainer configured!")

# Check memory before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"\n🎮 GPU: {gpu_stats.name}")
print(f"💾 Max memory: {max_memory} GB")
print(f"📊 Memory reserved: {start_gpu_memory} GB")

# Step 7: Train!
print("\n" + "=" * 60)
print("🚀 Step 7: TRAINING STARTED!")
print("=" * 60)
print("⏰ This will take approximately 15-25 minutes...")
print("☕ Grab a coffee and relax!\n")

trainer_stats = trainer.train()

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

# Show stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)

print(f"\n📊 Training Statistics:")
print(f"⏱️  Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
print(f"💾 Peak memory: {used_memory} GB ({used_percentage}% of {max_memory} GB)")
print(f"🎯 Memory for training: {used_memory_for_lora} GB")

# Step 8: Test the model
print("\n🎯 Step 8: Testing the model...")

FastLanguageModel.for_inference(model)

test_questions = [
    "What is your core approach to building and problem-solving?",
    "How do you learn best?",
    "Tell me about your question-driven approach to engineering.",
    "What has been one of the most significant experiences of your life?",
    "How have you grown as a writer and student?",
    "Why does robotics and control theory excite you?",
]

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {question}")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    print("Answer: ", end="")
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.7,
        top_p=0.9
    )
    print()

# Step 9: Save the model
print("\n💾 Step 9: Saving model...")

model.save_pretrained("pranav_assistant_lora")
tokenizer.save_pretrained("pranav_assistant_lora")

print("✅ Model saved to 'pranav_assistant_lora' folder!")

# Final message
print("\n" + "=" * 60)
print("🎉 ALL DONE!")
print("=" * 60)
print("\n📦 Your model is saved in the 'pranav_assistant_lora' folder")
print("📥 Download it from the Colab file browser (left sidebar)")
print("\n💡 To use your model:")
print("   1. Download the pranav_assistant_lora folder")
print("   2. Load it with FastLanguageModel.from_pretrained()")
print("\n🚀 You can also upload to Hugging Face Hub:")
print("   model.push_to_hub('your-username/pranav-assistant', token='...')")
print("\n⭐ Star Unsloth on GitHub: https://github.com/unslothai/unsloth")
print("💬 Join Discord for help: https://discord.gg/unsloth")
