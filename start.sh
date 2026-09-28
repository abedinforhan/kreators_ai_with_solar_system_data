#!/usr/bin/env bash

if [ ! -f .venv/bin/python ]; then
    echo "Setup has not been completed yet."
    echo "Please run: bash setup.sh"
    exit 1
fi

source .venv/bin/activate

while true; do
    clear
    echo "=================================================="
    echo "                 KREATORS AI"
    echo "=================================================="
    echo
    echo "  [1] Train My AI"
    echo "  [2] Test My AI"
    echo "  [3] Exit"
    echo
    read -p "Choose an option: " choice
    case $choice in
        1) echo; echo "Starting training..."; echo; python train.py; echo; read -p "Press Enter..." ;;
        2) python test.py ;;
        3) exit 0 ;;
        *) echo "Choose 1, 2, or 3."; sleep 1 ;;
    esac
done
