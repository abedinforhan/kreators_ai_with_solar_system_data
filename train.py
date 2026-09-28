import json
import os
import time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "Qwen/Qwen3-0.6B"
DATA_FILE = "data/train.jsonl"
OUTPUT_DIR = "my-ai"


def load_jsonl(path):
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

dataset = Dataset.from_list(examples)

print("Loading Qwen tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    prompt = f"Question: {example['question']}\nAnswer:"
    answer = f" {example['answer']}{tokenizer.eos_token}"

    prompt_tokens = tokenizer(
        prompt, truncation=True, max_length=128, add_special_tokens=False,
    )
    answer_tokens = tokenizer(
        answer, truncation=True, max_length=64, add_special_tokens=False,
    )

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels_ids = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_ids,
    }


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

print("Setting up LoRA...")

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="training_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_dataset,
)

print()
print("Training started with Qwen3-0.6B - expect 2-3 hours on CPU.")
print("Watch the loss drop as the AI learns.")
print()

start_time = time.time()
trainer.train()
end_time = time.time()

minutes = (end_time - start_time) / 60
print()
print(f"Training took {minutes:.1f} minutes.")

print("Saving your AI...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 50)
print("          TRAINING COMPLETE!")
print("=" * 50)
print()
print("Now run 'Test My AI' to see how your AI answers.")
