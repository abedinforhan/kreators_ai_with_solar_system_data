# Kreators AI

Train your own AI with question-and-answer pairs, then chat with it.

## Setup (one time)

**Windows:** Double-click `setup.bat`
**Mac/Linux:** Run `bash setup.sh`

This installs Python, PyTorch, and downloads the Qwen model. Takes 10-15 min.

## Use

**Windows:** `start.bat`
**Mac/Linux:** `bash start.sh`

Menu:
- [1] Train My AI — trains on your data (1.5-2.5 hours on CPU)
- [2] Test My AI — ask your AI any question
- [3] Exit

## Your data

Edit `data/train.jsonl`. Each line is:
```json
{"question": "Who is Erling Haaland?", "answer": "Erling Haaland is a Norwegian striker who plays for Manchester City."}
```

Add more questions → retrain → ask new questions.

## Try this first

1. Run `[2] Test My AI` BEFORE training — see what the untrained Qwen model guesses.
2. Then run `[1] Train My AI` — wait for it to finish.
3. Run `[2] Test My AI` again — compare how the answers changed.
