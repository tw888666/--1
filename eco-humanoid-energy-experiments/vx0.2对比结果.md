# `vx=0.2` 五组对照结果

## 结论摘要

1. 五组 `model_4001` 在 fixed-command（固定指令）20 回合评估中均为 `20/20` 成功，世界坐标系平均 x 速度为 `0.2095–0.2182 m/s`，机体坐标系平均 x 速度为 `0.2144–0.2194 m/s`。两种速度都在验收区间 `0.16–0.24 m/s` 内，这批策略的固定评估可确认为真正的 `vx=0.2` 对照实验。
2. 按统一的固定工况单位距离能耗比较，`legacy_joint_abs_10` 最低：转子绝对机械能较原始 `model_3000` 降低 `8.79%`，10 关节绝对机械能降低 `11.97%`。`joint_abs_8` 次之，分别降低 `6.57%` 和 `8.65%`。
3. 在 train-distribution（训练分布）100 回合中，指令 x 速度仍固定为 `0.2 m/s`，但保留了训练地形、观测噪声、推力、heading command（航向指令）和 domain randomization（域随机化）。因此全部回合的实际加权机体速度仅为 `0.1641–0.1664 m/s`，这不表示指令配置错误。
4. 旧的单次训练分布数据中，五组全回合平均 cost（成本）均比各自的原始 `model_3000` 参考下降，降幅为 `3.84%–10.51%`。但 2026-07-14 的配对审计发现，五份 `model_3000` 评估虽然 checkpoint、seed 和 env_id 一致，仍有 `54/100` 个 env_id 的成功/摔倒结果不一致。因此这些成功率和 cost 变化只作历史描述，不再作为严格配对结论。
5. 在旧数据中，若只统计成功回合，只有 `joint_abs_8` 的平均 cost 低于训练阈值，且仅低 `0.02%`；其余四组都高于阈值。这个现象说明不能仅用包含摔倒回合的 all-episode（全部回合）平均宣称约束已稳定满足，具体数值将由新的配对多评估随机种结果替代。
6. 目前训练及训练分布结果只有 `seed=0`（随机种 0），每组固定评估也只有一份。固定速度结论很明确，但方法优劣、成功率和 cost 阈值仍需要多随机种实验确认。

## 字段与口径

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `mean_velocity_x` | 世界坐标系的回合平均 x 速度 | 验收是否真正跟踪 `vx=0.2` |
| `mean_body_frame_velocity_x` | 机体坐标系的回合平均 x 速度 | 检查航向变化时的向前速度 |
| `rotor_positive_energy_8` | 8 电机转子正向机械能 $E^+_{\mathrm{rotor},8}$ | 比较正向驱动能耗 |
| `rotor_negative_energy_8` | 8 电机转子反向机械能绝对累计 $E^-_{\mathrm{rotor},8}$ | 比较制动侧机械能 |
| `joint_positive_energy_10` | 10 关节正向机械能 | 检查转子侧变化是否迁移到关节侧 |
| `joint_negative_energy_10` | 10 关节反向机械能绝对累计 | 比较关节制动侧机械能 |
| `distance_x` | 世界坐标系 x 方向前进距离 | 做固定工况单位距离归一化 |
| `cost1` | 当前 cost mode（成本模式）的单回合累计值 | 只能与同一 cost mode 的阈值和参考策略比较 |

固定评估的单位距离能耗统一按总和比值计算：

```text
E_pos / d_x = sum(positive_energy) / sum(distance_x)
E_neg / d_x = sum(negative_energy) / sum(distance_x)
E_abs / d_x = (sum(positive_energy) + sum(negative_energy)) / sum(distance_x)
```

训练分布的“加权机体速度”按 `sum(body_frame_distance_x) / sum(duration_s)` 计算。流水线 `summary.md` 中的 `Body vx` 是逐回合速度的算术平均，因此与本文加权结果有约 `0.0001–0.0026 m/s` 的小幅差异。

## 实验配置

五组训练均从同一原始 `model_3000.pt` 独立开始，使用 `8192 env`（并行环境）、`1000 iterations`（迭代）、`seed=0` 和 `train_command_x=0.2`，最终评估 checkpoint（模型检查点）为 `model_4001.pt`。

| 组 | cost mode | cost 定义 | 训练阈值 `cost_limit1` |
| --- | --- | --- | ---: |
| rp8 | `rotor_positive_8` | 8 电机转子 $P^+$ | 49.1577 |
| ra8 | `rotor_abs_8` | 8 电机转子 $P^+ + P^-$ | 81.7958 |
| jp8 | `joint_positive_8` | 8 驱动关节 $P^+$ | 45.2734 |
| ja8 | `joint_abs_8` | 8 驱动关节 $\lvert\tau\dot q\rvert$ | 72.3996 |
| lja10 | `legacy_joint_abs_10` | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$ 逐策略步累加 | 7557.1145 |

`legacy_joint_abs_10` 是旧的逐策略步功率和累加口径，数值尺度与其他四组不同。训练分布表中不对不同 cost mode 的原始 `cost1` 数值做横向大小排名。

## 数据源

| 用途 | 目录 |
| --- | --- |
| 原始策略 `model_3000` 固定评估参考 | `energy_evaluations/eco_ppolag_0p2vel_8000cost_seed123_model3000_20ep` |
| 五组 `model_4001` 固定评估 | `energy_evaluations/fixed_command_vx020_20ep/model4001_truevx020/<mode>/seed0` |
| 五组原始 `model_3000` 训练分布参考 | `energy_calibrations/calib_model3000_<mode>_vx020_s0` |
| 五组 `model_4001` 训练分布 | `energy_calibrations/train_dist_vx020/model4001_truevx020/<mode>/seed0` |
| 配对多评估随机种输出 | `energy_calibrations/paired_vx020` |
| 流水线总结 | `workflow_runs/vx020_seed0/summary.json` 和 `summary.md` |
| 配对评估流水线总结 | `workflow_runs/vx020_paired_eval/summary.json` 和 `summary.md` |

固定评估的 rp8、jp8 目录名为 `seed0`，对应训练 run（运行）也为 `s0`，但它们的 `metadata.json` 中 `seed` 字段为 `null`；ra8、ja8、lja10 的该字段为 `0`。由于固定评估关闭随机化且指令、checkpoint 和 cost mode 元数据均正确，本文保留 rp8 和 jp8 结果；但这是一个需要记录的 metadata（元数据）不一致。五组训练分布元数据的 `seed` 均为 `0`。

## 固定评估：20 回合

固定评估使用 `command_x=0.2`、`command_y=0`、`command_yaw=0`。验收要求为 `20/20` 成功，且世界坐标系和机体坐标系的回合平均 x 速度都在 `0.16–0.24 m/s` 内。

### 横向对照主表

本表与 `rotor_mixed_8_alpha050_review.md` 的 fixed-command 20 回合主表使用相同列和统计口径。`cost 定义` 表示训练时施加的约束目标；后面的六个能量指标是对每个策略统一重算的评估指标。`E_yaw` 只统计两个 hip yaw（髋部偏航）关节，`E_joint` 统计 10 个关节。所有单位距离指标均按 `sum(energy) / sum(distance_x)` 计算。

| 方法 | cost 定义 | 固定成功率 | 20回合总 $d_x$ | 平均距离 | 平均速度 | $E_{rotor,pos}/d_x$ | $E_{rotor,neg}/d_x$ | $E_{yaw,pos}/d_x$ | $E_{yaw,neg}/d_x$ | $E_{joint,pos}/d_x$ | $E_{joint,neg}/d_x$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 原始未继续训练策略 | 20/20 | 105.1730 | 5.2587 | 0.2190 | 7.5805 | 7.3959 | 0.1545 | 0.2283 | 8.3921 | 8.2813 |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$ | 20/20 | 103.5155 | 5.1758 | 0.2156 | **6.8903** | **6.7691** | **0.1263** | **0.2371** | **7.3442** | **7.3337** |
| `rotor_positive_8` seed0 | $P^+$ | 20/20 | 100.6197 | 5.0310 | 0.2095 | 7.3462 | 7.4148 | 0.1477 | 0.2722 | 8.3648 | 8.5579 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 20/20 | 104.7995 | 5.2400 | 0.2182 | 7.1243 | 7.0805 | 0.1632 | 0.2533 | 8.2321 | 8.2784 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 20/20 | 102.2338 | 5.1117 | 0.2129 | 7.1696 | 7.2032 | 0.1659 | 0.2708 | 7.7107 | 7.8492 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | 20/20 | 101.3334 | 5.0667 | 0.2110 | 7.0541 | 6.9379 | 0.1527 | 0.2668 | 7.6166 | 7.6145 |

### 速度补充验收

横向对照主表的“平均速度”是 `mean_velocity_x` 的回合平均。下表补充机体坐标系速度和验收结果。

| 方法 | 相对 `0.2` 的世界 x 速度偏差 | 机体 x 速度 | 两项速度验收 |
| --- | ---: | ---: | --- |
| 原始策略参考 `model_3000` | +9.51% | 0.2225 | 通过 |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | +7.79% | 0.2175 | 通过 |
| `rotor_positive_8` seed0 | +4.77% | 0.2144 | 通过 |
| `rotor_abs_8` seed0 | +9.12% | 0.2194 | 通过 |
| `joint_positive_8` seed0 | +6.45% | 0.2157 | 通过 |
| `joint_abs_8` seed0 | +5.51% | 0.2164 | 通过 |

五组对照都稍快于 `0.2 m/s`，但偏差最大的 `rotor_abs_8` 也只有 `+9.12%`，且两种速度均明显高于旧错误评估的约 `0.05 m/s`。

### 绝对能量汇总

下表在上面的统一能量指标上进一步合并正、负功，用于直接比较转子和 10 关节的绝对机械能。

| 方法 | $(E_{rotor,pos}+E_{rotor,neg})/d_x$ | 相对 `model_3000` | $(E_{joint,pos}+E_{joint,neg})/d_x$ | 相对 `model_3000` |
| --- | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 14.9763 | - | 16.6734 | - |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | **13.6594** | **-8.79%** | **14.6779** | **-11.97%** |
| `rotor_positive_8` seed0 | 14.7610 | -1.44% | 16.9228 | +1.50% |
| `rotor_abs_8` seed0 | 14.2048 | -5.15% | 16.5105 | -0.98% |
| `joint_positive_8` seed0 | 14.3728 | -4.03% | 15.5599 | -6.68% |
| `joint_abs_8` seed0 | 13.9920 | -6.57% | 15.2311 | -8.65% |

`rotor_positive_8` 的转子绝对能量仅降低 `1.44%`，关节绝对能量反而上升 `1.50%`，说明只限制转子正功时存在一定的能量口径迁移。`rotor_abs_8`、`joint_abs_8` 和 `legacy_joint_abs_10` 则在转子与关节统一口径上都低于原始策略参考。

### 相位步态

本表使用成功回合 `phase_summary.csv` 中的 `valid_steps` 占比。

| 方法 | `double_support` | `left_support` | `right_support` | `flight_or_transition` |
| --- | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 23.90% | 34.44% | 41.37% | 0.29% |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 23.12% | 34.56% | 42.03% | 0.29% |
| `rotor_positive_8` seed0 | 28.47% | 31.92% | 39.32% | 0.29% |
| `rotor_abs_8` seed0 | 25.70% | 32.81% | 41.20% | 0.29% |
| `joint_positive_8` seed0 | 27.91% | 31.88% | 39.91% | 0.29% |
| `joint_abs_8` seed0 | 24.69% | 33.39% | 41.63% | 0.29% |

五组的过渡/腾空占比都是 `0.29%`，没有通过大幅增加腾空相来异常降低能耗的迹象。`rotor_positive_8` 和 `joint_positive_8` 的双支撑占比较原始策略增加约 `4–5` 个百分点；`legacy_joint_abs_10` 最接近原始策略的相位分布。

## 训练分布：100 回合

> **证据状态（2026-07-14）**：本节保留第一轮 seed0 结果供回溯，但不再把它视为严格配对比较。审计确认五份 `model_3000.pt` 的 SHA-256（文件哈希）都是 `d4860c1e968fd28c2f5a4ba7e8970c30755352775dd9e7cda4d342b823ddf0af`，元数据均为 `checkpoint=3000`、`seed=0`、`command_x=0.2`、`num_envs=1024`，100 个 env_id 也完全相同；但不同 cost mode 之间仍有 `54/100` 个 env_id 的 outcome（结果）不同。新流水线对每个评估 seed 只运行一次 `model_3000`，在同一条轨迹上同时重算五种 cost，并对五个 `model_4001` 强制检查初始地形、指令、机器人状态和物理随机参数的场景指纹。

这一阶段的 `metadata.json` 均记录：

```text
calibration_profile = training_distribution_fixed_lin_vel_x
command_x = 0.2
heading_command = true
calibration_episodes = 100
num_envs = 1024
```

评估固定 `lin_vel_x=0.2`，但保留 trimesh（三角网格）训练地形、观测噪声、随机推力、延迟、摩擦、质量、质心、电机强度和 PD（比例-微分控制）参数等随机化。每组从 1024 个环境的固定 cohort（队列样本）中取 100 个完整回合。

训练阈值是各 cost mode 在原始 `model_3000` 同口径训练分布校准均值的 `95%`。摔倒回合通常较短，可能人为拉低 all-episode 平均 cost，所以下面同时报告全部回合和成功回合口径。

为与 `rotor_mixed_8_alpha050_review.md` 直接对比，下面两张主表使用相同列和统计口径。`legacy_joint_abs_10` 的训练 cost 为旧 ECO 功率和逐策略步累计口径，因此在表中乘 `policy_dt=0.01 s` 转为能量等效 cost；其他四组不做换算。“平均速度”使用 `sum(body_frame_distance_x) / sum(duration_s)` 的加权口径，“平均距离”和“总距离”均来自 `body_frame_distance_x`。

### 所有回合口径

| 方法 | cost 定义 | 阈值 | 训练分布成功率 | 平均速度 | 平均距离 | 总距离 | 所有回合平均 cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 75.5711 | 77% | 0.1651 | 3.6010 | 360.1050 | 74.6037 |
| `rotor_positive_8` seed0 | $P^+$ | 49.1577 | 76% | 0.1641 | 3.4541 | 345.4133 | 46.3085 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 81.7958 | 81% | 0.1660 | 3.6813 | 368.1283 | 82.7914 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 45.2734 | **85%** | 0.1661 | 3.7573 | 375.7339 | 44.7078 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | 72.3996 | 80% | **0.1664** | 3.6497 | 364.9736 | 69.3453 |

#### 历史 `model_3000` 对照补充

| 方法 | `model_3000` 成功率 | `model_4001` 成功率 | 变化 | `model_3000` cost 均值 | 当前 cost 均值 | cost 中位数 | cost 变化 | 当前 cost 相对阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy_joint_abs_10` seed0（已乘 `0.01s`） | 76% | 77% | +1 个百分点 | 79.5486 | 74.6037 | 74.5018 | -6.22% | -1.28% |
| `rotor_positive_8` seed0 | 84% | 76% | -8 个百分点 | 51.7450 | 46.3085 | 48.0067 | -10.51% | -5.80% |
| `rotor_abs_8` seed0 | 81% | 81% | 0 个百分点 | 86.1009 | 82.7914 | 80.0775 | -3.84% | **+1.22%** |
| `joint_positive_8` seed0 | 86% | **85%** | -1 个百分点 | 47.6562 | 44.7078 | 43.8155 | -6.19% | -1.25% |
| `joint_abs_8` seed0 | 85% | 80% | -5 个百分点 | 76.2101 | 69.3453 | 69.6206 | -9.01% | -4.22% |

`joint_positive_8` 的当前成功率最高，为 `85%`。五组 all-episode cost 都比各自原始策略参考低，但 `rotor_abs_8` 仍高于训练阈值 `1.22%`；其他四组的 all-episode 均值低于阈值。

### 成功回合口径

| 方法 | cost 定义 | 阈值 | 成功回合数 | 平均速度 | 平均距离 | 总距离 | 成功回合平均 cost | 成功回合 cost 是否通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 75.5711 | 77/100 | 0.1665 | 3.9962 | 307.7111 | 78.3433 | 否，78.3433 > 75.5711 |
| `rotor_positive_8` seed0 | $P^+$ | 49.1577 | 76/100 | 0.1660 | 3.9837 | 302.7583 | 49.6371 | 否，49.6371 > 49.1577 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 81.7958 | 81/100 | **0.1697** | 4.0733 | 329.9382 | 84.7065 | 否，84.7065 > 81.7958 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 45.2734 | **85/100** | 0.1680 | 4.0328 | 342.7921 | 45.5915 | 否，45.5915 > 45.2734 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | 72.3996 | 80/100 | 0.1675 | 4.0203 | 321.6242 | 72.3840 | **是，72.3840 < 72.3996，但几乎压线** |

成功回合口径比 all-episode 口径更严格。`rotor_positive_8`、`joint_positive_8` 虽然全部回合平均低于阈值，但成功回合平均分别高出 `0.98%` 和 `0.70%`；`legacy_joint_abs_10` 高出 `3.67%`。`joint_abs_8` 仅低 `0.02%`，在更换随机种后很可能改变符号，不应视为稳定余量。

## 综合评估

- **速度跟踪**：五组都通过 `vx=0.2` 固定评估，没有再现旧评估约 `0.05 m/s` 的问题。
- **固定工况能耗**：`legacy_joint_abs_10` 在统一转子和关节绝对能量口径上最低，`joint_abs_8` 第二。`rotor_positive_8` 的关节绝对能量反而略升，不适合仅根据其训练 cost 宣称整体节能。
- **随机环境鲁棒性**：旧 seed0 数据中 `joint_positive_8` 成功率最高，`rotor_positive_8` 降幅最大；但由于基线轨迹未严格配对，排名需等待新的 eval seed 0-5 汇总。
- **约束满足性**：旧数据以成功回合为主口径时，没有任何一组展现出充足的阈值余量。在配对多 seed 结果出来前，不建议直接使用当前 all-episode 均值乘 `0.95` 得到的“推荐新阈值”开始下一轮正式训练。
- **当前可支持的结论**：这批实验已经回答“固定指令速度是否真正为 `0.2 m/s`”。它还不足以回答“哪个 cost mode 在随机环境中稳定最优”，后者至少需要补齐多随机种同口径评估。
