#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="${REMOTE_NAME:-gdrive}"
CONF_DIR="${CONF_DIR:-$HOME/.config/rclone}"
CONF_FILE="${CONF_FILE:-$CONF_DIR/rclone.conf}"

echo "[1/4] Ensure rclone exists"
if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found."
  echo "Trying to install (may fail on locked-down clusters)."

  if command -v sudo >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  elif command -v apt-get >/dev/null 2>&1; then
    echo "Use your admin/package manager: apt-get install rclone"
    exit 1
  elif command -v brew >/dev/null 2>&1; then
    brew install rclone
  else
    echo "No supported installer found. Install rclone manually, then rerun."
    exit 1
  fi
fi

echo "[2/4] Create secure rclone config directory"
mkdir -p "$CONF_DIR"
chmod 700 "$CONF_DIR"
touch "$CONF_FILE"
chmod 600 "$CONF_FILE"

echo "[3/4] Check if remote '$REMOTE_NAME' already exists"
if rclone listremotes --config "$CONF_FILE" | grep -qx "${REMOTE_NAME}:"; then
  echo "Remote '${REMOTE_NAME}' already configured in $CONF_FILE"
else
  echo "Remote '${REMOTE_NAME}' not configured yet."
  echo
  echo "You must complete OAuth once."
  echo "Recommended workflow for headless clusters:"
  echo "  A) Run 'rclone config' on your laptop, create remote '$REMOTE_NAME'"
  echo "  B) Copy the resulting rclone.conf entry to this cluster at:"
  echo "     $CONF_FILE"
  echo
  echo "If this cluster has a browser or can open a URL, you can also run:"
  echo "  rclone config --config \"$CONF_FILE\""
  echo
  echo "Starting interactive config now..."
  rclone config --config "$CONF_FILE"
fi

echo "[4/4] Smoke test"
echo "Configured remotes:"
rclone listremotes --config "$CONF_FILE"

echo
echo "OK. Your rclone config is at:"
echo "  $CONF_FILE"
echo "Keep it private (600). Do NOT commit it."