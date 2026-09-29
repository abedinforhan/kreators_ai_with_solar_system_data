# Kreators AI

Teach a real AI your own facts. Then ask it questions and see what it learned!

---

## What you need

- A laptop (Windows or Mac) or
- A Google account (for training your AI)

---

## Step 1: Get the code

1. Download this project as a ZIP from GitHub (green **Code** button → **Download ZIP**)
2. Unzip it to your Desktop
3. Open the folder in VS Code (File → Open Folder)

---

## Step 2: Install it (one time only)

Open a Terminal inside VS Code (**Terminal → New Terminal**), then:

**Windows:** type `setup.bat` and press Enter
**Mac:** type `bash setup.sh` and press Enter

Wait until you see **SETUP COMPLETE!** This can take 10-15 minutes. Grab a snack.

---

## Step 3: Try the untrained AI first

In the Terminal, run:

**Windows:** `.\start.bat`
**Mac:** `bash start.sh`

You'll see a menu:

```
[1] Train My AI
[2] Test My Trained AI
[3] Test Untrained Brain
[4] Exit
```

**Choose [3]** and ask it something like:

> What is Artemis II?

Notice how weird or wrong the answer is. This is the AI **before** you teach it anything.

---

## Step 4: Add your own facts

Open the file `data/train.jsonl`. Each line looks like this:

```json
{"question": "What is Mars?", "answer": "Mars is the fourth planet from the Sun."}
```

Add your own questions and answers, one per line. Save the file.

**Tips:**
- Keep answers short (1-2 sentences)
- Add at least 40-60 examples for good results

---

## Step 5: Train the AI on Google Colab (the fast way , if your computer struggles)

Training on your laptop would take **many hours**. Instead, we use Google's free super-computer.

1. Go to https://colab.research.google.com
2. Sign in with your Google account
3. Click **File → Upload notebook** and pick `kreators_ai_train.ipynb` from your project folder
4. Click **Runtime → Change runtime type → T4 GPU → Save**
5. Copy your `data/train.jsonl` contents into the training data cell (Step 3 in the notebook)
6. Click each cell's play button ▶ from top to bottom
7. Wait about 30-60 minutes for training to finish
8. The last cell will download 3 files: `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`

---

## Step 6: Install your trained AI

On your laptop:

1. In your project folder, make a new folder called **my-ai-trained-brain** (if it doesn't already exist)
2. Move all 3 downloaded files into that folder

Your project should look like this:

```
kreators-ai/
├── my-ai-trained-brain/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json
├── data/
│   └── train.jsonl
├── start.bat
└── ...
```

---

## Step 7: Test your trained AI!

Run `start.bat` again and choose **[2] Test My Trained AI**.

Ask the same question from Step 3:

> What is Artemis II?

Compare the answer to what the untrained AI said. **This is what your data taught it!**

---

## Fun things to try

- Ask the same question in **[2]** and **[3]** to see how much training changed the AI
- Add wrong answers on purpose to see the AI learn wrong facts
- Delete most of your data and retrain to see it get confused
- Try a completely different topic (dinosaurs, video games, your favourite books) and train again

---

## Something not working?

- **"Setup has not been completed"** → run setup.bat first
- **Training too slow?** → use Google Colab (Step 5), not your laptop
- **AI gives weird answers** → add more training examples and train again