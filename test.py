import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "my-ai"

print("Loading AI...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

have_trained_ai = os.path.exists(f"{ADAPTER}/adapter_config.json")

if have_trained_ai:
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    print("This is your TRAINED AI.")
else:
    model = base_model
    print("This is the untrained BASE Qwen model (nothing learned from your data yet).")
    print("Train it first to see the difference.")

model.eval()
print()


def answer(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full.split("Answer:", 1)[-1].strip()

    # Compute average confidence per generated token
    with torch.no_grad():
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        logits = model(outputs).logits[0, inputs["input_ids"].shape[-1] - 1:-1]
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[range(len(gen_ids)), gen_ids]
        confidence = token_probs.mean().item() if len(gen_ids) > 0 else 0.0

    return reply, confidence


while True:
    print("=" * 50)
    print("             TEST YOUR AI")
    print("=" * 50)
    print()

    question = input("Ask a question (or type 'back'): ").strip()

    if question.lower() == "back":
        break
    if not question:
        continue

    reply, confidence = answer(question)

    print()
    print("YOUR AI SAYS:")
    print()
    print(f"  {reply}")
    print()
    print(f"  Confidence: {confidence:.0%}")
    print()

    input("Press Enter to try another...")
    print()

print()
print("Returning to menu...")
