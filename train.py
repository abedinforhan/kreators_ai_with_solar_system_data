# ============================================================
# TRAIN.PY - This is where the AI actually learns from your data
# ============================================================
# What happens here, in plain English:
#   1. Read your question/answer examples from data/train.jsonl
#   2. Load a "base brain" (Qwen3) that doesn't know your data yet
#   3. Attach a small "learning notebook" (called LoRA) to the brain
#   4. Show the brain your examples over and over so it learns them
#   5. Save the trained learning notebook to a folder called my-ai-trained-brain
# ============================================================

import json
import time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# The "base brain" - a pre-existing AI model we start from
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Where your training examples live
DATA_FILE = "data/train.jsonl"

# Where we'll save the trained AI
OUTPUT_DIR = "my-ai-trained-brain"


# ------------------------------------------------------------
# STEP 1: Read the training examples from the data file
# ------------------------------------------------------------
def load_jsonl(path):
    """Reads a .jsonl file line by line and turns each line into a
    Python dictionary with question and answer keys."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


print("Loading training data...")
examples = load_jsonl(DATA_FILE)
print(f"Found {len(examples)} question-and-answer pairs.")

if len(examples) < 2:
    print("WARNING: Need at least 2 examples to train.")
    raise SystemExit(1)

# Turn our list of examples into a format the trainer understands
dataset = Dataset.from_list(examples)


# ------------------------------------------------------------
# STEP 2: Load the base brain (Qwen3) that we're going to teach
# ------------------------------------------------------------
print("Loading base brain (Qwen3-0.6B)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Set up padding (needed for training - not important to understand)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ------------------------------------------------------------
# STEP 3: Turn text into numbers the AI can understand
# ------------------------------------------------------------
# AIs don't read text directly - they read numbers. This function
# converts each question/answer into a series of numbers (tokens).
def tokenize(example):
    # This is the format the AI learns to expect: Question then Answer
    prompt = f"Question: {example['question']}\nAnswer:"
    answer = f" {example['answer']}{tokenizer.eos_token}"

    # Turn the text into numbers
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=128, add_special_tokens=False)
    answer_tokens = tokenizer(answer, truncation=True, max_length=64, add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)

    # This tells the AI: "grade yourself only on the ANSWER part,
    # not the QUESTION part" (-100 means "ignore this")
    labels_ids = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_ids,
    }


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)


# ------------------------------------------------------------
# STEP 4: Attach a small "learning notebook" (LoRA) to the brain
# ------------------------------------------------------------
# Instead of retraining the entire brain (millions of connections),
# we attach a small trainable "notebook" that captures ONLY what
# needs to change for our specific task. Much faster and lighter.
print("Attaching a small learning notebook to the brain...")

lora_config = LoraConfig(
    r=8,                    # How big the notebook is (bigger = more room to learn, slower to train)
    lora_alpha=16,          # How strongly the notebook affects the brain
    lora_dropout=0.05,      # Small randomness to prevent memorization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which parts of the brain the notebook connects to
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ------------------------------------------------------------
# STEP 5: Set up training - how many times to review the examples
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="training_checkpoints",
    num_train_epochs=20,               # Review each example 6 times
    per_device_train_batch_size=1,    # One example at a time
    gradient_accumulation_steps=1,
    learning_rate=3e-4,               # How big each "adjustment" is when the AI is wrong
    logging_steps=1,                  # Print progress after every step
    save_strategy="no",
    report_to="none",
    use_cpu=True,                     # Train on CPU (works on any laptop)
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)


# ------------------------------------------------------------
# STEP 6: Actually train (this is the long part)
# ------------------------------------------------------------
print()
print("Training started... on CPU this takes 2-3 hours for Qwen3.")
print("Watch the loss number drop - that's the AI learning.")
print("(Lower loss = better answers)")
print()

start_time = time.time()
trainer.train()
end_time = time.time()

print()
print(f"Training took {(end_time - start_time) / 60:.1f} minutes.")


# ------------------------------------------------------------
# STEP 7: Save the trained brain so we can use it later
# ------------------------------------------------------------
print("Saving your trained AI...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 50)
print("          TRAINING COMPLETE!")
print("=" * 50)
print()
print("Your trained AI is saved. Choose Test Trained Brain")
print("from the menu to talk to it.")