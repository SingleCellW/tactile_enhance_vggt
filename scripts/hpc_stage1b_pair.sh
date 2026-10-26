#!/usr/bin/env bash
# Template only: creating this file submits no job.
# Submit one seed per job after the conditional follow-up is authorized:
#   sbatch scripts/hpc_stage1b_pair.sh 43 4000
# Repeat for 44 and 45. Each allocation runs BOTH objectives and BOTH modes.
#SBATCH --job-name=stage1b-pair
#SBATCH --account=xiaochx-critical
#SBATCH --qos=normal
#SBATCH --partition=critical
#SBATCH --gres=gpu:NVIDIAA40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/public/home/wukf2025/projects/tactile_vggt_stage1b_20261026
#SBATCH --output=/public/home/wukf2025/projects/tactile_vggt_stage1b_20261026/stage1b-pair-%j.out
#SBATCH --error=/public/home/wukf2025/projects/tactile_vggt_stage1b_20261026/stage1b-pair-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Run only in an allocated Slurm job; preserve its CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

pair_seed="${1:?Pass the predeclared seed: 43, 44, or 45}"
pair_steps="${2:-4000}"
case "$pair_seed" in 43|44|45) ;; *) echo "Unexpected seed" >&2; exit 2 ;; esac
case "$pair_steps" in 2000|4000) ;; *) echo "Use the common predeclared budget" >&2; exit 2 ;; esac

pair_project="/public/home/wukf2025/projects/tactile_vggt_stage1b_20261026"
pair_python="/public/home/wukf2025/miniforge3/envs/te/bin/python"
cd "$pair_project"
test -f stage1b-bootstrap-owner.json
test -f staging_verified.json

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
export LANG=C

# cuda:0 is logical GPU zero INSIDE this allocation, never physical GPU zero.
# Do not export or replace CUDA_VISIBLE_DEVICES here.
exec "$pair_python" -B scripts/run_camera_pair_pilot.py train \
  --data artifacts/camera_pilot_data_20261026 \
  --prefix artifacts/camera_pilot_prefix_20261026 \
  --repository vendor/vggt \
  --weights weights/model.pt \
  --output outputs/camera_pair_pilot_20261026_hpc \
  --device cuda:0 --seeds "$pair_seed" --modes visual internal \
  --objectives center delta --steps "$pair_steps" --batch-size 32 \
  --eval-every 200 --learning-rate 1e-4 --preload-gb 8 --resume
