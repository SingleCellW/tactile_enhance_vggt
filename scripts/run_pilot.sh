#!/usr/bin/env bash
# Launch the paired baseline only on two explicitly selected/allocated GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
pilot_python="${PILOT_PYTHON:-python}"
pilot_artifacts="${TACTILE_VGGT_ARTIFACT_ROOT:-$PWD/.local}"
pilot_cache="${PILOT_CACHE:-$pilot_artifacts/cache/pickplace_vggt_native_sync_stride4}"
pilot_run="${PILOT_RUN:-act_vggt_native_sync_s43}"
pilot_selection="${PILOT_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
IFS=',' read -r -a pilot_gpus <<< "$pilot_selection"
if (( ${#pilot_gpus[@]} != 2 )) || [[ "${pilot_gpus[0]}" == "${pilot_gpus[1]}" ]]; then
  echo 'Set PILOT_GPUS or CUDA_VISIBLE_DEVICES to exactly two allocated GPU IDs/UUIDs.' >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PILOT_GPUS:-}" && "$PILOT_GPUS" != "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Within Slurm, preserve the scheduler GPU allocation; do not override PILOT_GPUS.' >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
pilot_logs="$pilot_artifacts/logs"
pilot_outputs="$pilot_artifacts/runs"
mkdir -p "$pilot_logs" "$pilot_outputs"
if [[ -e "$pilot_logs/$pilot_run.started" ]]; then
  echo "Campaign already submitted: $pilot_run; inspect existing jobs before using a new run name." >&2
  exit 1
fi
"$pilot_python" - "$pilot_cache" <<'PY'
import json, pathlib, sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / 'manifest.json').read_text())
if not (manifest.get('complete') and manifest.get('historical_source_tactile_superseded')
        and manifest.get('native_tactile_report', {}).get('complete')):
    raise SystemExit('Refusing training: requires a completed cache with synchronized native tactile.')
alignment = manifest.get('configuration', {}).get('rgb_alignment', {})
if alignment.get('rgb_lag_frames') != 1 or alignment.get('verified') is not True:
    raise SystemExit('Refusing training: inherited RGB requires an audited one-frame correction.')
PY
for pilot_gpu in "${pilot_gpus[@]}"; do
  pilot_available=$(nvidia-smi -i "$pilot_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( pilot_available < 20000 )); then
    echo "GPU $pilot_gpu has only $pilot_available MiB free; do not start another workload." >&2
    exit 1
  fi
done
date -Is > "$pilot_logs/$pilot_run.started"
trap 'date -Is > "$pilot_logs/$pilot_run.failed"' ERR
CUDA_VISIBLE_DEVICES="${pilot_gpus[0]}" "$pilot_python" scripts/train_act.py --cache "$pilot_cache" --output "$pilot_outputs/${pilot_run}_anchored" --mode anchored >"$pilot_logs/${pilot_run}_anchored.log" 2>&1 &
pilot_first=$!
CUDA_VISIBLE_DEVICES="${pilot_gpus[1]}" "$pilot_python" scripts/train_act.py --cache "$pilot_cache" --output "$pilot_outputs/${pilot_run}_unanchored" --mode unanchored >"$pilot_logs/${pilot_run}_unanchored.log" 2>&1 &
pilot_second=$!
printf '%s\n' "$pilot_first" "$pilot_second" >"$pilot_logs/$pilot_run.train_pids"
echo "ACT jobs started: anchored=$pilot_first unanchored=$pilot_second"
pilot_failed=0
wait "$pilot_first" || pilot_failed=1
wait "$pilot_second" || pilot_failed=1
if (( pilot_failed )); then
  echo 'One or more ACT jobs failed; inspect per-run failure.json.' >&2
  exit 1
fi
date -Is > "$pilot_logs/$pilot_run.completed"
