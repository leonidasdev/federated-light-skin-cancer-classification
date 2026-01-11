#!/usr/bin/env bash
set -euo pipefail

# Read configuration from environment variables (fall back to sensible defaults)
NODE_ID=${NODE_ID:-0}
DATASET=${DATASET:-HAM10000}
SERVER=${SERVER:-[::]:8080}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}

ARGS=(--node-id "$NODE_ID" --dataset "$DATASET" --server "$SERVER")

if [[ -n "$LOCAL_EPOCHS" ]]; then
  ARGS+=(--local-epochs "$LOCAL_EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi

exec python main_client.py "${ARGS[@]}"
