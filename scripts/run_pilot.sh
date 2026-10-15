#!/usr/bin/env bash
# Run on rimlab-server after regenerating synchronized tactile and repairing the cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PILOT_PYTHON:-$PWD/.venv/bin/python}"
CACHE="${PILOT_CACHE:-$PWD/cache/pickplace_vggt_native_sync_stride4}"
RUN="${PILOT_RUN:-act_vggt_native_sync_s43}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
if [[ -e "logs/$RUN.started" ]]; then
  echo "Campaign already submitted: $RUN; inspect existing jobs before using a new run name." >&2
  exit 1
fi
"$PYTHON" - "$CACHE" <<'PY'
import json, pathlib, sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / "manifest.json").read_text())
if not (manifest.get("complete") and manifest.get("historical_source_tactile_superseded")
        and manifest.get("native_tactile_report", {}).get("complete")):
    raise SystemExit("Refusing training: requires a completed cache with synchronized native tactile.")
alignment = manifest.get("configuration", {}).get("rgb_alignment", {})
if alignment.get("rgb_lag_frames") != 1 or alignment.get("verified") is not True:
    raise SystemExit("Refusing training: inherited RGB requires an audited one-frame correction.")
PY
for gpu in 1 2; do
  available=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( available < 20000 )); then
    echo "GPU $gpu has only $available MiB free; do not start another workload." >&2
    exit 1
  fi
done
date -Is > "logs/$RUN.started"
trap 'date -Is > "logs/$RUN.failed"' ERR
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/train_act.py --cache "$CACHE" --output "outputs/${RUN}_anchored" --mode anchored >"logs/${RUN}_anchored.log" 2>&1 &
train0=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" scripts/train_act.py --cache "$CACHE" --output "outputs/${RUN}_unanchored" --mode unanchored >"logs/${RUN}_unanchored.log" 2>&1 &
train1=$!
printf '%s\n' "$train0" "$train1" >"logs/$RUN.train_pids"
echo "ACT jobs started: anchored=$train0 unanchored=$train1"
failed=0
wait "$train0" || failed=1
wait "$train1" || failed=1
if (( failed )); then
  echo "One or more ACT jobs failed; inspect per-run failure.json." >&2
  exit 1
fi
date -Is > "logs/$RUN.completed"
