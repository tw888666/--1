#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/xy.chen/miniconda3/etc/profile.d/conda.sh
conda activate bruce_gym

PYTHON_BIN="python"
BASE_RUN="$PWD/pretrained_weights/eco_ppolag_0.2vel_8000cost_seed123"
LOG_DIR="$PWD/true_v020_logs"
mkdir -p "$LOG_DIR"

common_args=(
  --task=bruce_ppolag
  --headless
  --resume
  --experiment_name=exp
  --load_run="$BASE_RUN"
  --checkpoint=3000
  --reset_lagrange_on_resume
  --max_iterations=1000
  --num_envs=8192
  --train_command_x=0.2
  --sim_device=cuda:0
  --rl_device=cuda:0
  --seed=0
)

env PYTHONPATH="$PWD" \
  LD_LIBRARY_PATH="/home/xy.chen/miniconda3/envs/bruce_gym/lib:${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$PYTHON_BIN" -u -m bruce_gym.scripts.train "${common_args[@]}" \
  --run_name=jp8_truevx020_env8192_s0 \
  --energy_cost_mode=joint_positive_8 \
  --cost_limit1=45.27336714076996 \
  > "$LOG_DIR/train_jp8_truevx020_env8192_s0.log" 2>&1

env PYTHONPATH="$PWD" \
  LD_LIBRARY_PATH="/home/xy.chen/miniconda3/envs/bruce_gym/lib:${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$PYTHON_BIN" -u -m bruce_gym.scripts.train "${common_args[@]}" \
  --run_name=ja8_truevx020_env8192_s0 \
  --energy_cost_mode=joint_abs_8 \
  --cost_limit1=72.39956164360046 \
  > "$LOG_DIR/train_ja8_truevx020_env8192_s0.log" 2>&1
