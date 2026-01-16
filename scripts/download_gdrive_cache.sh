#!/usr/bin/env bash
set -euo pipefail

CONF_FILE="${CONF_FILE:-$HOME/.config/rclone/rclone.conf}"
REMOTE="${REMOTE:-gdrive:pm_ws25_cache/cache}"
DEST="${DEST:-./cache}"

mkdir -p "$DEST"

rclone copy "$REMOTE" "$DEST" \
  --config "$CONF_FILE" \
  --progress \
  --checksum \
  --fast-list