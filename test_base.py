# ============================================================
# TEST_BASE.PY - Talk to the UNTRAINED base AI
# ============================================================
# This loads ONLY the base Qwen3 brain, with no learning from
# your data at all. Use this to compare BEFORE-vs-AFTER training.
#
# Same question in this file vs test.py will show you exactly
# how much the AI changed after you taught it.
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen3-0.6B"

# ------------------------------------------------------------
# Load ONLY the base brain - no learning notebook attached
# ------------------------------------------------------------
print("Loading the untrained BASE brain (Qwen3-0.6B)...")
print("(This AI knows what Qwen already learned, but NOTHING")
print(" about your training data. Great for BEFORE comparison.)")
print()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.eval()


# ------------------------------------------------------------
# The function that asks the AI a question
# (same logic as test.py - just no trained adapter loaded)
# ------------------------------------------------------------
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

    with torch.no_grad():
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        logits = model(outputs).logits[0, inputs["input_ids"].shape[-1] - 1:-1]
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[range(len(gen_ids)), gen_ids]
        confidence = token_probs.mean().item() if len(gen_ids) > 0 else 0.0

    return reply, confidence


# ------------------------------------------------------------
# Interactive loop
# ------------------------------------------------------------
while True:
    print("=" * 50)
    print("       TEST THE UNTRAINED BASE BRAIN")
    print("=" * 50)
    print()

    question = input("Ask a question (or type 'back'): ").strip()

    if question.lower() == "back":
        break
    if not question:
        continue

    reply, confidence = answer(question)

    print()
    print("UNTRAINED BASE BRAIN SAYS:")
    print()
    print(f"  {reply}")
    print()
    print(f"  Confidence: {confidence:.0%}")
    print()

    input("Press Enter to try another...")
    print()

print()
print("Returning to menu...")