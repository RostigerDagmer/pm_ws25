#!/usr/bin/env bash
set -e  # stop on first error

echo "=== [1/6] Checking for .venv ==="
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
else
  echo ".venv already exists — skipping creation."
fi

# Detect OS for activation command
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
  source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  echo "⚠️  Unsupported OS type: $OSTYPE"
  exit 1
fi

echo "=== [2/6] Upgrading pip and setuptools ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [3/6] Installing PM4Py dependencies ==="
if [ -f "pm4py/setup.py" ]; then
  pip install -e ./pm4py
else
  echo "⚠️  No PM4Py source found. Did you forget to clone --recursive or initialize submodules afterwards?"
fi

echo "=== [4/6] Installing project dependencies ==="
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "No requirements.txt found — skipping."
fi

echo "=== [5/6] Installing development tools (black, flake8, pre-commit) ==="
pip install black flake8 pre-commit

echo "=== [6/6] Installing pre-commit hooks ==="
if [ -f ".pre-commit-config.yaml" ]; then
  pre-commit install
  pre-commit autoupdate
  echo "Pre-commit hooks installed and updated."
else
  echo "No .pre-commit-config.yaml found — skipping."
fi

echo "🚀 Ready!"
