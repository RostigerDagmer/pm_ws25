#!/usr/bin/env bash
set -euo pipefail

CONF_FILE="${CONF_FILE:-$HOME/.config/rclone/rclone.conf}"
REMOTE="${REMOTE:-gdrive:pm_ws25_cache/cache}"
SRC="${SRC:-./cache}"

# Refuse obvious foot-guns
SRC_REAL="$(realpath "$SRC")"
if [[ "$SRC_REAL" == "/" || "$SRC_REAL" == "$HOME" || "$SRC_REAL" == "$(realpath .)" ]]; then
  echo "Refusing to sync from dangerous source: $SRC_REAL"
  exit 1
fi
if [[ "$SRC_REAL" != */cache ]]; then
  echo "Refusing: source must end with '/cache' (got: $SRC_REAL)"
  exit 1
fi
if [[ "$REMOTE" != gdrive:pm_ws25_cache/cache ]]; then
  echo "Refusing: remote must be exactly gdrive:pm_ws25_cache/cache"
  echo "Got: $REMOTE"
  exit 1
fi

echo "About to SYNC (this can delete remote files inside the cache folder):"
echo "  FROM: $SRC_REAL"
echo "  TO:   $REMOTE"
read -p "Type YES to continue: " ok
[[ "$ok" == "YES" ]] || exit 1

rclone sync "$SRC_REAL" "$REMOTE" \
  --config "$CONF_FILE" \
  --progress \
  --checksum \
  --fast-list \
  --delete-after