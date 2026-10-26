#!/usr/bin/env bash
# Submit from the repository root with site-specific flags, for example:
# mkdir -p .local/logs
# sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --output=.local/logs/stage1b-pair-%j.out --error=.local/logs/stage1b-pair-%j.err scripts/hpc_stage1b_pair.sh 43 4000
# For an external artifact root, pass its pre-created logs directory to sbatch.
#SBATCH --job-name=stage1b-pair
#SBATCH --output=.local/logs/stage1b-pair-%j.out
#SBATCH --error=.local/logs/stage1b-pair-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Run only in an allocated Slurm job; preserve its CUDA_VISIBLE_DEVICES.' >&2
  exit 2
fi
pair_seed="${1:?Pass the predeclared seed: 43, 44, or 45}"
pair_steps="${2:-4000}"
case "$pair_seed" in 43|44|45) ;; *) echo 'Unexpected seed' >&2; exit 2 ;; esac
case "$pair_steps" in 2000|4000) ;; *) echo 'Use the common predeclared budget' >&2; exit 2 ;; esac
pair_project="${TACTILE_VGGT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
pair_python="${PILOT_PYTHON:-python}"
cd "$pair_project"
test -f scripts/run_camera_pair_pilot.py
pair_artifacts="${TACTILE_VGGT_ARTIFACT_ROOT:-$PWD/.local}"
mkdir -p "$pair_artifacts/logs"
exec >"$pair_artifacts/logs/stage1b-pair-${SLURM_JOB_ID}.out" 2>"$pair_artifacts/logs/stage1b-pair-${SLURM_JOB_ID}.err"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C LANG=C
# cuda:0 is logical GPU zero INSIDE the allocation; never select a physical GPU here.
exec "$pair_python" -B scripts/run_camera_pair_pilot.py train \
  --data "$pair_artifacts/artifacts/camera_pilot_data_20261026" \
  --prefix "$pair_artifacts/artifacts/camera_pilot_prefix_20261026" \
  --output "$pair_artifacts/runs/camera_pair_pilot_20261026_hpc" \
  --device cuda:0 --seeds "$pair_seed" --modes visual internal \
  --objectives center delta --steps "$pair_steps" --batch-size 32 \
  --eval-every 200 --learning-rate 1e-4 --preload-gb 8 --resume
