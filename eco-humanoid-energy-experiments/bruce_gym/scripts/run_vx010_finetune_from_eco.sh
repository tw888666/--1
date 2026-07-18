#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <lja10|ja8|ra8|jp8|rp8> <物理GPU编号0-4>" >&2
  exit 2
fi

finetune_mode="$1"
finetune_gpu="$2"

if [[ ! "$finetune_gpu" =~ ^[0-4]$ ]]; then
  echo "GPU编号必须是0-4；本服务器GPU5不可用。" >&2
  exit 2
fi

case "$finetune_mode" in
  lja10)
    energy_cost_mode="legacy_joint_abs_10"
    cost_limit="6000"
    limit_token="6000"
    ;;
  ja8)
    energy_cost_mode="joint_abs_8"
    cost_limit="56.274914"
    limit_token="56p274914"
    ;;
  ra8)
    energy_cost_mode="rotor_abs_8"
    cost_limit="56.198978"
    limit_token="56p198978"
    ;;
  jp8)
    energy_cost_mode="joint_positive_8"
    cost_limit="33.982423"
    limit_token="33p982423"
    ;;
  rp8)
    energy_cost_mode="rotor_positive_8"
    cost_limit="33.944458"
    limit_token="33p944458"
    ;;
  *)
    echo "未知模式: $finetune_mode" >&2
    exit 2
    ;;
esac

project_root="/home/xy.chen/tw/ECO-humanoid-remote/eco-humanoid-energy-experiments"
source_run="$project_root/logs/exp/May27_16-06-41_Eco"
source_checkpoint="$source_run/model_3000.pt"
source_sha256="b8556ca4cdc6a166d2645ce34d19d283dcafb67c832dae883f14931bfb95e7db"
experiment_name="ECO-finetune-model3000"
run_name="vx010_ft1000_from_eco3000_${finetune_mode}_L${limit_token}_s123"
terminal_log="$project_root/logs/$experiment_name/训练_${run_name}.log"

if [[ ! -f "$source_checkpoint" ]]; then
  echo "找不到原 ECO 检查点: $source_checkpoint" >&2
  exit 1
fi

actual_sha256="$(sha256sum "$source_checkpoint" | awk '{print $1}')"
if [[ "$actual_sha256" != "$source_sha256" ]]; then
  echo "原 ECO 检查点 SHA-256 不匹配。" >&2
  echo "期望: $source_sha256" >&2
  echo "实际: $actual_sha256" >&2
  exit 1
fi

mkdir -p "$project_root/logs/$experiment_name"
source /home/xy.chen/miniconda3/etc/profile.d/conda.sh
conda activate bruce_gym

echo "模式: $energy_cost_mode"
echo "阈值: $cost_limit"
echo "物理GPU: $finetune_gpu"
echo "来源: $source_checkpoint"
echo "来源SHA-256: $actual_sha256"
echo "日志: $terminal_log"

exec env \
  CUDA_VISIBLE_DEVICES="$finetune_gpu" \
  PYTHONPATH="$project_root:/home/xy.chen/isaacgym/python" \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu" \
  python -u -m bruce_gym.scripts.train \
    --task=bruce_ppolag \
    --headless \
    --warm_start \
    --load_run="$source_run" \
    --checkpoint=3000 \
    --experiment_name="$experiment_name" \
    --run_name="$run_name" \
    --energy_cost_mode="$energy_cost_mode" \
    --cost_limit1="$cost_limit" \
    --num_envs=8192 \
    --max_iterations=1000 \
    --seed=123 \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
  > "$terminal_log" 2>&1
