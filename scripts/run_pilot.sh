#!/usr/bin/env bash
# Run on rimlab-server. The detached parent supervises cache preparation and both ACT runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PILOT_PYTHON:-$PWD/.venv/bin/python}"
CACHE="${PILOT_CACHE:-$PWD/cache/pickplace_vggt_stride4}"
RUN="${PILOT_RUN:-act_vggt_s43}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
if [[ -e "logs/$RUN.started" ]]; then
  echo "Campaign already submitted: $RUN; inspect existing jobs before using a new run name." >&2
  exit 1
fi
test -s weights/model.pt
for gpu in 1 2; do
  available=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( available < 20000 )); then
    echo "GPU $gpu has only $available MiB free; do not start another workload." >&2
    exit 1
  fi
done
date -Is > "logs/$RUN.started"
trap 'date -Is > "logs/$RUN.failed"' ERR
common=(
  scripts/prepare_cache.py
  --source /home/wukf/projects/tactile_enhance/Lerobot_datasets/Mujoco_Picklift_rc_260407_ud_T_R_D_NPC
  --readouts /home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz
  --raw-root /home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/raw_force/Mujoco_Picklift_random_cube_260212
  --output "$CACHE" --vggt-repo vendor/vggt --weights weights/model.pt
  --obs-stride 4 --image-size 518 --batch-size 4 --num-shards 2
)
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "${common[@]}" --shard-index 0 >"logs/${RUN}_cache0.log" 2>&1 &
cache0=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "${common[@]}" --shard-index 1 >"logs/${RUN}_cache1.log" 2>&1 &
cache1=$!
printf '%s\n' "$cache0" "$cache1" >"logs/$RUN.cache_pids"
failed=0
wait "$cache0" || failed=1
wait "$cache1" || failed=1
if (( failed )); then
  echo "Cache preparation failed; no policy jobs submitted." >&2
  exit 1
fi
"$PYTHON" scripts/prepare_cache.py --output "$CACHE" --merge-shards
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
