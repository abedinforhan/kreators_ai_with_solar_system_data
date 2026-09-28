#!/usr/bin/env bash
set -e

echo "=================================================="
echo "           KREATORS AI - SETUP"
echo "=================================================="
echo
echo "This can take 10-15 minutes. Please wait."
echo

PYTHON_BIN=""
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.12 not found. Installing..."
    echo "(Your password may be requested - this is normal.)"
    echo
    if [ "$(uname -s)" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python@3.12
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y python3.12 python3.12-venv
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3.12
    else
        echo "Please install Python 3.12 manually from https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON_BIN=python3.12
fi

echo "Creating workspace..."
rm -rf .venv
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

echo
echo "Installing AI tools..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo
echo "Downloading Qwen3-0.6B model (about 1 GB, takes a few minutes)..."
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); print('Qwen ready.')"

echo
echo "=================================================="
echo "  SETUP COMPLETE!"
echo "  Run: bash start.sh   to begin."
echo "=================================================="
