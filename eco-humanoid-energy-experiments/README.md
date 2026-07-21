<div id="top" align="center">



<h2>ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking</h2>

<p align="center">
  <a href="https://sites.google.com/view/eco-humanoid">
    <img src="https://img.shields.io/badge/Project%20Page-ECO--Humanoid-0B6E4F?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <!-- Add arXiv / DOI badges when available -->
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-8B0000?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a> -->
  <img src="https://img.shields.io/badge/Code-Baselines%20Released-0B6E4F?style=flat-square&logo=github" alt="Code Baselines Released">
</p>

<p align="center">
  <b><a href="https://weidonghuang.com" target="_blank" rel="noopener noreferrer">Weidong Huang</a></b><sup>*</sup> ·
  <a href="https://jingwen-zhang-aaron.github.io/" target="_blank" rel="noopener noreferrer">Jingwen Zhang</a><sup>*, †</sup> ·
  Jiongye Li ·
  Shibowen Zhang ·
  Jiayang Wu ·
  Jiayi Wang ·
  Hangxin Liu ·
  Yaodong Yang ·
  <a href="https://yaosu.info/" target="_blank" rel="noopener noreferrer">Yao Su</a><sup>†</sup>
  <br>
  <sup>*</sup>Equal contribution · <sup>†</sup>Corresponding authors
</p>

> We release ECO training/evaluation code together with baseline implementations: **PPO**, **IPO**, **P3O**, and **CRPO**.



</div>

---

## 🔥 Highlights

- **Energy as an explicit constraint (not a reward term):** ECO reformulates motor energy consumption as an inequality constraint for more **interpretable** and **tunable** energy optimization.
- **Stable + energy-efficient humanoid walking:** Achieves robust locomotion while driving energy down to a target budget via **PPO-Lagrangian** (primal-dual updates).
- **Real-world validation on BRUCE:** Demonstrates sim-to-real deployment with substantially reduced energy consumption compared to **MPC** and **standard PPO**.
- **Emergent efficient behaviors:** Reduced body shaking, lighter steps, and less flexed knees—without manually prescribing “efficient gait” heuristics.

---

## 📋 Overview
<p align="center">
  <!-- Reuse your paper figure as the README teaser -->
  <!-- Option A (recommended): export `figures/motivation.pdf` to PNG and put it under `assets/` -->
  <img src="assets/motivation.png" width="320" alt="ECO teaser: comparison with MPC and PPO">
</p>

**ECO (Energy-Constrained Optimization)** is a constrained reinforcement learning framework for humanoid locomotion that separates *task rewards* (e.g., velocity tracking, stability) from *energy optimization* by treating energy as an explicit constraint.

Instead of tuning many reward weights (often non-intuitive and time-consuming), ECO uses **physically meaningful thresholds** for constraints:

- **Energy constraint**: discounted cumulative motor power (torque × joint velocity)
- **Reference motion / symmetry constraint**: mirror-consistency loss to encourage stable and symmetric gait

ECO is trained with **PPO-Lagrangian**, which dynamically adjusts the Lagrange multipliers to satisfy constraints during learning.

**Project website (demos + videos):** https://sites.google.com/view/eco-humanoid



---

## 🏃 Results (at a glance)

- **Constraint RL baselines compared:** PPO-Lag (ECO), IPO, P3O, CRPO, plus MPC and standard PPO (reward shaping).
- **Sim-to-sim transfer:** policies transferred across **Isaac Gym → MuJoCo / Gazebo** with stable walking and consistent energy reduction.
- **Sim-to-real on BRUCE:** ECO maintains low motor power near the specified budget while remaining robust.

> For qualitative demos and detailed plots, see the project website:
> https://sites.google.com/view/eco-humanoid

---

## 📦 Code

Run all commands from the repository root.
Tested environment: Ubuntu 22.04.5 LTS with NVIDIA GeForce RTX 4090.

### Installation

```bash
# create and activate conda env
conda create -n bruce_gym python=3.8 -y
conda activate bruce_gym

# install pytorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# install Isaac Gym (Preview 4) after PyTorch
# download Isaac Gym from the official NVIDIA Isaac Gym website,
# unzip it, then run:
cd isaacgym/python
pip install -e .
cd ../..

# install project + dependencies from setup.py / requirements.py
pip install -e .
```

### Ubuntu 22.04 (Isaac Gym)

Before running Isaac Gym training/play on Ubuntu 22.04, set:

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# use your own environment path (recommended)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# optional: if Isaac Gym still cannot find libpython3.8.so.1.0,
# locate it manually and export that directory:
find / -name "*ibpython3.8.so.1.0*" 2>/dev/null
# example only, replace with your own path:
# export LD_LIBRARY_PATH=/home/<your_username>/miniconda3/envs/bruce_gym/lib:$LD_LIBRARY_PATH
```

If PyTorch reports GPUs in `torch.cuda.device_count()` but fails with
`CUDA driver initialization failed`, start from a clean shell and avoid appending
stale CUDA runtime paths repeatedly. A known-good minimal setup on this project is:

```bash
cd /home/xy.chen/tw/ECO-humanoid-remote/eco-humanoid-energy-experiments
conda activate bruce_gym

export PYTHONPATH="$PWD"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  # replace with an idle physical GPU id
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu"

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch cuda =", torch.version.cuda)
print("available =", torch.cuda.is_available())
print("count =", torch.cuda.device_count())
print(torch.zeros(1, device="cuda:0"))
PY
```

Only run Isaac Gym after this test can create the CUDA tensor. Prefer module
execution from the project root:

```bash
python -m bruce_gym.scripts.evaluate_energy ...
```

Avoid `python bruce_gym/scripts/evaluate_energy.py` unless `PYTHONPATH` is set,
because running a file path can hide the project package root from Python.

### Train (Isaac Gym)

Training setup note:
- To reproduce the training curve from the original paper, use `mesh_type = 'plane'` and `num_envs = 4096`.
- Using `num_envs = 8192` uses more GPU memory but can converge faster.
- Using `mesh_type = 'trimesh'` is recommended when targeting a more robust sim-to-real policy.
- Current paper-reproduction command setting is `lin_vel_x = [0.1, 0.1]` and `lin_vel_y = [0.0, 0.0]` (in `commands.ranges`).
- To train policies that can run at different velocity commands, increase the command ranges (especially `lin_vel_x`, and optionally `lin_vel_y`) instead of fixing them to single values.
- These parameter settings can be changed in config files under `bruce_gym/envs/custom/` (for example `bruce_gym/envs/custom/brucewalk_config.py` and related `brucewalk_config_*.py` files).

```bash
# ECO (PPO-Lagrangian)
python bruce_gym/scripts/train.py \
  --task=bruce_ppolag \
  --headless \
  --run_name=eco_ppolag_train \
  --sim_device=cuda:0 \
  --rl_device=cuda:0

# PPO baseline
python bruce_gym/scripts/train.py \
  --task=bruce_ppo \
  --headless \
  --run_name=baseline_ppo_train \
  --sim_device=cuda:0 \
  --rl_device=cuda:0

# IPO baseline
python bruce_gym/scripts/train.py \
  --task=bruce_ipo \
  --headless \
  --run_name=baseline_ipo_train \
  --sim_device=cuda:0 \
  --rl_device=cuda:0

# P3O baseline
python bruce_gym/scripts/train.py \
  --task=bruce_p3o \
  --headless \
  --run_name=baseline_p3o_train \
  --sim_device=cuda:0 \
  --rl_device=cuda:0

# CRPO baseline
python bruce_gym/scripts/train.py \
  --task=bruce_crpo \
  --headless \
  --run_name=baseline_crpo_train \
  --sim_device=cuda:0 \
  --rl_device=cuda:0
```

### Play (Isaac Gym)

We store trained models in `logs/exp/`. Before evaluation, choose the target run directory from `logs/exp/<timestamp>_<run_name>` and use that path for `--load_run` / `--load_model`.

```bash
python bruce_gym/scripts/play.py \
  --task=bruce_ppolag \
  --load_run=<run_name_or_absolute_run_path> \
  --checkpoint=<checkpoint_id>
```

### 减速器修正机械能模型

仓库支持 `reducer_corrected_8`：正功效率由输出端额定转矩比例
$x=|T|/T_N$ 动态计算，负功绝对值以系数 1 加入成本。公式和代码数据流见
[`减速器非线性效率机械能模式实现说明.md`](减速器非线性效率机械能模式实现说明.md)；
额定转矩参数、基线校准、续训和训练后复核命令见
[`减速器修正机械能训练说明.md`](减速器修正机械能训练说明.md)。启用该模式前必须提供
可信的减速器输出端额定转矩 $T_N$。

### Offline Evaluation Review

For a headless server, prefer an offline review bundle over a live viewer. The
energy evaluator can write the normal CSV outputs and then build a review
directory with per-episode time series, best/median/worst plots, a summary CSV,
and a Markdown report:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m bruce_gym.scripts.evaluate_energy \
  --task=bruce_ppolag \
  --headless \
  --resume \
  --experiment_name=exp \
  --load_run=<run_name_or_absolute_run_path> \
  --checkpoint=<checkpoint_id> \
  --energy_cost_mode=rotor_mixed_8_alpha050 \
  --num_eval_episodes=20 \
  --command_x=0.1 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --seed=0 \
  --make_eval_report
```

By default, fixed-command outputs use a compact policy id:

```text
energy_evaluations/fixed20/vx010_rm8a05_s00_model4000/
```

The policy id format is:

```text
vx<command_x*100>_<cost_alias>_s<seed>_model<checkpoint>
```

For example, `rotor_positive_8` at `vx=0.2`, seed 0, checkpoint 4001 is
`vx020_rp8_s00_model4001`. Use `--output_dir` only when you intentionally want
to override this layout.

This creates:

```text
energy_evaluations/fixed20/<policy_id>/eval_report/
├── summary.csv
├── representative_episodes.json
├── report.md
├── episodes/episode_0000.csv
├── episode_best_curves.png
├── episode_median_curves.png
├── episode_worst_curves.png
└── joint_energy_contribution.png
```

If evaluation CSVs already exist, generate only the review bundle:

```bash
python -m bruce_gym.scripts.generate_eval_review \
  --eval_dir=energy_evaluations/fixed20/<policy_id>
```

To record the selected best/median/worst episodes as MP4 files, rerun the same
policy with the review directory. This replays the deterministic fixed-command
evaluation and records only the selected episode ids:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m bruce_gym.scripts.record_eval_video \
  --task=bruce_ppolag \
  --headless \
  --resume \
  --experiment_name=exp \
  --load_run=<run_name_or_absolute_run_path> \
  --checkpoint=<checkpoint_id> \
  --energy_cost_mode=rotor_mixed_8_alpha050 \
  --command_x=0.1 \
  --review_dir=energy_evaluations/fixed20/<policy_id>/eval_report \
  --sim_device=cuda:0 \
  --rl_device=cuda:0
```

The review and video commands load model weights for evaluation only; they do
not train and do not modify checkpoints.

### Play (MuJoCo)

```bash
python bruce_gym/scripts/sim2sim_bruce.py \
  --load_model=<absolute_run_path> \
  --checkpoint=<checkpoint_id>
```

Optional: add `--terrain` to use terrain MJCF.

### Play (Gazebo)

Gazebo playback depends on the **BRUCE official library**.
Install that library first, otherwise `bruce_gym/scripts/gazebo/*` will fail on BRUCE interface imports.
Also replace `~/.gazebo/models/bruce/model.sdf` with `bruce_gym/scripts/gazebo/model.sdf` before running Gazebo playback.

Typical setup:

```bash
# from your BRUCE official library repo
pip install -e .
pip install termcolor==1.1.0
pip install numba==0.49.0
python3 -m Startups.memory_manager
```

Start Gazebo first:

```bash
gzserver bruce.world
```

Then run:

```bash
python bruce_gym/scripts/gazebo/test_checkpoint.py \
  --task=bruce_ppolag \
  --load_run=<run_name_or_absolute_run_path> \
  --checkpoint=<checkpoint_id>
```

If your local package path is `humanoid/` instead of `bruce_gym/`, replace the script prefix accordingly.


---

## 📌 BibTeX

If you find this work useful, please consider citing:

```bibtex
@ARTICLE{ECOhumanoid,
  author={Huang, Weidong and Zhang, Jingwen and Li, Jiongye and Zhang, Shibowen and Wu, Jiayang and Wang, Jiayi and Liu, Hangxin and Yang, Yaodong and Su, Yao},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={ECO: Energy-Constrained Optimization With Reinforcement Learning for Humanoid Walking}, 
  year={2026},
  volume={23},
  number={},
  pages={4861-4876},
  keywords={Legged locomotion;Humanoid robots;Energy efficiency;Energy consumption;Tuning;Optimization;Costs;Stability criteria;Reinforcement learning;Automation;Humanoid and bipedal locomotion;constrained reinforcement learning;legged robots},
  doi={10.1109/TASE.2026.3662755}}
