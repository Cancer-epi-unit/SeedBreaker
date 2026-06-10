#!/bin/bash
# launch_training.sh — SeedBreaker overnight training launcher
#
# Usage:
#   bash /Users/josh/Desktop/Projects/SeedBreaker/launch_training.sh
#
# What it does:
#   1. Finds the seedbreaker conda Python (no need to activate conda first)
#   2. Kills any existing training process
#   3. Runs a 2-minute smoke test to verify everything works
#   4. Launches the full 10-epoch overnight training in the background
#   5. Saves the PID so you can kill it if needed

set -uo pipefail

PROJECT="/Users/josh/Desktop/Projects/SeedBreaker"
LOG_SMOKE="$PROJECT/models/train_log_smoke.txt"
LOG_FULL="$PROJECT/models/train_log_v3.txt"
PID_FILE="$PROJECT/models/training.pid"

echo "=== SeedBreaker Training Launcher ==="
date

# ── 1. Locate Python in the seedbreaker conda env ─────────────────────────────
# Try direct paths first (no need for conda activate in non-interactive shell)
PYTHON=""
for candidate in \
  "/opt/homebrew/Caskroom/miniconda/base/envs/seedbreaker/bin/python3" \
  "$HOME/miniconda3/envs/seedbreaker/bin/python3" \
  "$HOME/anaconda3/envs/seedbreaker/bin/python3" \
  "/opt/miniconda3/envs/seedbreaker/bin/python3"
do
  if [ -x "$candidate" ]; then
    PYTHON="$candidate"
    break
  fi
done

# Fallback: try conda run
if [ -z "$PYTHON" ]; then
  echo "Direct env path not found; trying conda run..."
  for CONDA_SH in \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"
  do
    if [ -f "$CONDA_SH" ]; then
      # shellcheck disable=SC1090
      source "$CONDA_SH"
      conda activate seedbreaker 2>/dev/null && PYTHON="$(which python3)" && break
    fi
  done
fi

if [ -z "$PYTHON" ]; then
  echo "ERROR: Could not find seedbreaker conda Python. Check conda paths."
  exit 1
fi

echo "Python: $PYTHON"
echo "Version: $($PYTHON --version 2>&1)"

# ── 2. Kill any existing training processes ────────────────────────────────────
echo ""
echo "Killing existing training processes (if any)..."
pkill -f "train_lora" 2>/dev/null && echo "  Killed." || echo "  None found."
sleep 2

cd "$PROJECT"
mkdir -p models

# ── 3. Smoke test (2000 examples, 1 epoch, ~2-3 min) ─────────────────────────
echo ""
echo "Running smoke test (2000 examples, 1 epoch)..."
echo "Log: $LOG_SMOKE"
echo ""

PYTHONUNBUFFERED=1 "$PYTHON" train_lora_v2.py \
  --data      training_data/training_data.parquet \
  --out       models/seedbreaker_smoke \
  --epochs    1 \
  --batch     8 \
  --lr        2e-4 \
  --max-train 2000 \
  2>&1 | tee "$LOG_SMOKE"

SMOKE_EXIT=${PIPESTATUS[0]}
if [ $SMOKE_EXIT -ne 0 ]; then
  echo ""
  echo "SMOKE TEST FAILED (exit $SMOKE_EXIT). Full log: $LOG_SMOKE"
  exit 1
fi

echo ""
echo "Smoke test PASSED."
rm -rf "$PROJECT/models/seedbreaker_smoke"

# ── 4. Full overnight training ─────────────────────────────────────────────────
echo ""
echo "Starting full overnight training (10 epochs, ~679k examples)..."
echo "Log: $LOG_FULL"
echo ""

PYTHONUNBUFFERED=1 nohup "$PYTHON" train_lora_v2.py \
  --data    training_data/training_data.parquet \
  --out     models/seedbreaker_lora_v2 \
  --epochs  10 \
  --batch   8 \
  --lr      2e-4 \
  > "$LOG_FULL" 2>&1 &

PID=$!
disown $PID  # detach from shell so it survives terminal close
echo "$PID" > "$PID_FILE"

echo "Training started: PID $PID"
echo "PID saved to: $PID_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Monitor progress:"
echo "  tail -f $LOG_FULL"
echo "Kill if needed:"
echo "  kill \$(cat $PID_FILE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
