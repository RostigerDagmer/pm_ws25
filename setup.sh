#!/usr/bin/env bash
set -e  # stop on first error

# ====================================
# LRZ Cluster-specific setup
# ====================================
if [[ -d "/dss/dsshome1" ]] || [[ "$HOSTNAME" == *"lrz.de"* ]]; then
  echo "=== LRZ Cluster detected ==="

  # Unload base Python if loaded
  if module is-loaded python/3.10.12-base 2>/dev/null; then
    echo "Unloading python/3.10.12-base..."
    module unload python/3.10.12-base
  fi

  # Load extended Python (includes dev headers for cvxopt)
  if ! module is-loaded python/3.10.12-extended 2>/dev/null; then
    echo "Loading python/3.10.12-extended..."
    module load python/3.10.12-extended
  fi

  # Remove old venv if it exists (might be created with wrong Python version)
  if [ -d ".venv" ]; then
    echo "Removing existing .venv (will recreate with Python 3.10 extended)..."
    rm -rf .venv
  fi
fi

echo "=== [1/6] Checking for .venv ==="
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
else
  echo ".venv already exists — skipping creation."
fi

# Detect OS for activation command
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
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
  pip install -e ./pm4py --config-settings editable_mode=strict
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



