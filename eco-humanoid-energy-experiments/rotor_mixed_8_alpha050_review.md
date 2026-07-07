# `rotor_mixed_8_alpha050` 横向评估说明


| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `rotor_positive_energy_8` | 8 电机转子正向机械能 $E^+_{\mathrm{rotor},8}$ | 判断 mixed 是否真的降低正向驱动功 |
| `rotor_negative_energy_8` | 8 电机转子反向机械能绝对累计 $E^-_{\mathrm{rotor},8}$ | 判断制动/潜在回收侧是否异常 |
| `rotor_abs_energy_8` | $E^+_{\mathrm{rotor},8}+E^-_{\mathrm{rotor},8}$ | 对齐 `rotor_abs_8` 对照 |
| `joint_positive_energy_10` | 10 关节侧正向机械能 | 检查转子侧节能是否只是能量口径迁移 |
| `joint_negative_energy_10` | 10 关节侧反向机械能绝对累计 | 检查关节侧制动功变化 |
| `distance_x` | 世界坐标系 x 方向前进距离 | 做单位距离归一化 |
| `mean_velocity_x` | 回合平均 x 方向速度 | 判断是否靠减速省能 |



说明：历史已生成的部分 CSV 没有直接写出 `rotor_abs_energy_8`。本文表格对历史结果使用

```text
rotor_abs_energy_8 = rotor_positive_energy_8 + rotor_negative_energy_8
```

后续重新运行 `evaluate_energy.py` 时，该字段应直接出现在 `episode_summary.csv`、`step_timeseries.csv` 和 `phase_summary.csv` 中。

## 评估口径

固定评估采用 `v_x = 0.1 m/s`、20 回合、无随机化、无外部扰动。为了和 mixed seed（随机种子）0 公平比较，本文当前主表使用 seed0 对照，而不是把 mixed 单种子和其他方法六种子均值混在一起。

本文后续把 `legacy_joint_abs_10` 称为旧 ECO 基线（baseline），用于横向方法对比；`model_3000` 只称为原始策略参考，用于回答“相对未继续训练原始策略是否变化”。因此表格里不再把 `model_3000` 写作基线。

固定评估数据源：

| 方法 | 数据目录 |
| --- | --- |
| 原始策略参考 `model_3000` | `energy_evaluations/mixed_alpha05_model3000_vx010_20ep` |
| `legacy_joint_abs_10` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/legacy_joint_abs_10/seed0` |
| `rotor_positive_8` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/rotor_positive_8/seed0` |
| `rotor_abs_8` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/rotor_abs_8/seed0` |
| `rotor_mixed_8_alpha050` seed0 | `energy_evaluations/eval_rm8a05_cl40895_keepopt_s0_model4000_vx010_20ep` |
| `joint_positive_8` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/joint_positive_8/seed0` |
| `joint_abs_8` seed0 | 当前没有同口径 `v_x=0.1`、model4000 keepopt 结果；已有 `ja8` 为 v020 实验，不混入本表 |

单位距离能耗使用总和比值：

```text
E_pos / d_x = sum(rotor_positive_energy_8) / sum(distance_x)
E_neg / d_x = sum(rotor_negative_energy_8) / sum(distance_x)
E_abs / d_x = (sum(rotor_positive_energy_8) + sum(rotor_negative_energy_8)) / sum(distance_x)
E_mix / d_x = (sum(rotor_positive_energy_8) + 0.5 * sum(rotor_negative_energy_8)) / sum(distance_x)
```

训练分布使用 100 回合 calibration（校准）结果。阈值通过不只看 all-episode（全部回合）平均，还要看 `cost1_success_episodes.mean` 是否低于对应阈值，因为短摔倒回合可能把 all-episode 平均成本虚假拉低。

## 相位步态表

没有看到双支撑时间被压得很短。成功回合的相位时间占比如下：

| 方法 | `double_support` | `left_support` | `right_support` | `flight_or_transition` |
| --- | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 19.10% | 40.05% | 40.56% | 0.29% |
| `legacy_joint_abs_10` seed0 | 19.93% | 38.60% | 41.18% | 0.29% |
| `rotor_positive_8` seed0 | 19.86% | 38.53% | 41.31% | 0.29% |
| `rotor_abs_8` seed0 | 18.77% | 39.58% | 41.36% | 0.29% |
| `rotor_mixed_8_alpha050` seed0 | 19.88% | 40.10% | 39.73% | 0.29% |

mixed 的 `double_support = 19.88%`，略高于原始策略参考，也接近 `rotor_positive_8` 和 `legacy_joint_abs_10`。因此当前没有“通过压短双支撑来省能”的异常迹象。



## 汇总表

下面先保留 seed0 的 fixed-command（固定指令）20 回合评估和 train-distribution calibration（训练分布校准）100 回合表，再给出已有 seed0-5 的六种子均值表，避免把不同评估口径混在同一个表里。

### 固定评估：20 回合

本表只使用 fixed-command 20 回合评估的 `episode_summary.csv`。`cost 定义` 表示训练时施加的约束目标；后面的 `E_rotor` 和 `E_joint` 不是训练 cost，而是对每个训练后策略统一重算的一组评估指标。因此不同方法都会有 $E_{rotor,pos}/d_x$、$E_{rotor,neg}/d_x$、$E_{joint,pos}/d_x$ 和 $E_{joint,neg}/d_x$，这样才能在同一把尺子下横向比较。`20回合总 d_x` 是 20 回合 `distance_x` 的总和；`distance_x` 是单回合平均行走距离；`平均速度` 是 `mean_velocity_x` 的回合平均。所有单位距离指标均按 `sum(energy) / sum(distance_x)` 计算。

| 方法 | cost 定义 | 固定成功率 | 20回合总 $d_x$ | 平均距离 | 平均速度 | $E_{rotor,pos}/d_x$ | $E_{rotor,neg}/d_x$ | $E_{joint,pos}/d_x$ | $E_{joint,neg}/d_x$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 原始未继续训练策略 | 20/20 | 55.9929 | 2.7996 | 0.1166 | 8.0737 | 8.3499 | 10.4327 | 10.8831 |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$ | 20/20 | 54.2354 | 2.7118 | 0.1129 | 7.4846 | 7.8795 | 9.3527 | 9.9502 |
| `rotor_positive_8` seed0 | $P^+$ | 20/20 | 57.5218 | 2.8761 | 0.1198 | 7.3911 | 7.8181 | 9.9027 | 10.4252 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 20/20 | 58.3768 | 2.9188 | 0.1216 | 7.4331 | 7.7111 | 9.7019 | 10.1154 |
| `rotor_mixed_8_alpha050` seed0 | $P^+ + 0.5P^-$ | 20/20 | 56.8467 | 2.8423 | 0.1184 | 7.4248 | 7.8655 | 9.9094 | 10.4597 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 20/20 | 53.6594 | 2.6830 | 0.1117 | 7.3229 | 7.8613 | 9.5172 | 10.1475 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | - | - | - | - | - | - | - | - |

`joint_abs_8` 目前只有 v020 相关训练/校准记录，没有本表同口径的 `v_x=0.1`、model4000 keepopt 固定评估，因此固定评估表中保留为空。

### 训练分布：100 回合

本节使用训练分布 100 回合校准的 `calibration_summary.json` 和 `episode_costs.csv`。`阈值` 来自各 cost mode 在原始 `model_3000` 上的训练分布基线校准（baseline calibration），不是 20 回合固定评估得到的数值。这里的距离来自 `episode_costs.csv` 中的 `body_frame_distance_x`，是 body-frame x 方向积分距离，不是固定评估表里的 world-root `distance_x`。

`legacy_joint_abs_10` 的原始训练 cost 是旧 ECO 的功率和逐策略步累计口径，数值会到 `5000+`，不能直接和其他已经按能量增量累计的 cost mode 放在同一表里比较。因此本节对 `legacy_joint_abs_10` 做能量等效换算：

```text
legacy 能量等效 cost = legacy 原始 cost * policy_dt
```

当前评估的 `policy_dt = 0.01s`，所以 `legacy_joint_abs_10` 的阈值 `5540.5193` 在表中写为 `55.4052`。其他 cost mode 已经是能量累计口径，不做这个乘法。

#### 所有回合口径

本表统计 100 个回合的全部样本，包括成功回合和摔倒回合。`平均速度` 使用 `所有回合总距离 / 所有回合总时长` 的加权口径；`平均距离` 使用 100 回合的 `body_frame_distance_x` 平均值。

| 方法 | cost 定义 | 阈值 | 训练分布成功率 | 平均速度 | 平均距离 | 总距离 | 所有回合平均 cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 原始策略，mixed 基线校准口径 | - | 89% | 0.0896 | 2.0462 | 204.6228 | 44.3030 |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 55.4052 | 88% | 0.0857 | 1.9604 | 196.0419 | 53.6064 |
| `rotor_positive_8` seed0 | $P^+$ | 30.8848 | 92% | 0.0827 | 1.9197 | 191.9738 | 30.6771 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 50.8560 | 86% | 0.0875 | 1.9748 | 197.4755 | 49.5549 |
| `rotor_mixed_8_alpha050` seed0 | $P^+ + 0.5P^-$ | 40.8955 | 88% | 0.0866 | 1.9893 | 198.9276 | 41.0888 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 30.9057 | 92% | 0.0837 | 1.9294 | 192.9378 | 29.5103 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | - | - | - | - | - | - |

#### 成功回合口径

本表只统计 `episode_outcome == success` 的成功回合。`平均速度` 使用 `成功回合总距离 / 成功回合总时长` 的加权口径；最后一列按成功回合平均 cost 判断是否低于对应阈值。

| 方法 | cost 定义 | 阈值 | 成功回合数 | 平均速度 | 平均距离 | 总距离 | 成功回合平均 cost | 成功回合 cost 是否通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始策略参考 `model_3000` | 原始策略，mixed 基线校准口径 | - | 89/100 | 0.0892 | 2.1404 | 190.4925 | 45.4223 | - |
| `legacy_joint_abs_10` seed0（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 55.4052 | 88/100 | 0.0864 | 2.0739 | 182.5001 | 53.9760 | 是，53.9760 < 55.4052 |
| `rotor_positive_8` seed0 | $P^+$ | 30.8848 | 92/100 | 0.0839 | 2.0139 | 185.2752 | 30.9197 | 否，30.9197 > 30.8848 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 50.8560 | 86/100 | 0.0872 | 2.0932 | 180.0171 | 51.5796 | 否，51.5796 > 50.8560 |
| `rotor_mixed_8_alpha050` seed0 | $P^+ + 0.5P^-$ | 40.8955 | 88/100 | 0.0880 | 2.1121 | 185.8691 | 41.6067 | 否，41.6067 > 40.8955 |
| `joint_positive_8` seed0 | 8 驱动关节 $P^+$ | 30.9057 | 92/100 | 0.0831 | 1.9932 | 183.3765 | 30.2125 | 是，30.2125 < 30.9057 |
| `joint_abs_8` seed0 | 8 驱动关节 $\lvert\tau\dot q\rvert$ | - | - | - | - | - | - | 缺同口径 vx010 数据 |

### 六种子均值：固定评估 20 回合

本节使用已有 seed0-5 数据。统计口径为：每个 seed 先按上面的固定评估公式算出一行，再对 6 个 seed 取算术平均；固定成功率写 6 个 seed 合并后的成功回合数。原始策略参考 `model_3000` 仍是单次参考，不参与 6-seed 平均。`rotor_mixed_8_alpha050` 和 `joint_abs_8` 目前没有同口径 6-seed 固定评估数据，因此不列入本节表格。

| 方法 | cost 定义 | 固定成功率 | 20回合总 $d_x$ | 平均距离 | 平均速度 | $E_{rotor,pos}/d_x$ | $E_{rotor,neg}/d_x$ | $E_{joint,pos}/d_x$ | $E_{joint,neg}/d_x$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 原始未继续训练策略 | 20/20 | 55.9929 | 2.7996 | 0.1166 | 8.0737 | 8.3499 | 10.4327 | 10.8831 |
| `legacy_joint_abs_10` seed0-5 均值（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$ | 120/120 | 54.5474 | 2.7274 | 0.1136 | 7.5220 | 7.8459 | 9.2804 | 9.8019 |
| `rotor_positive_8` seed0-5 均值 | $P^+$ | 120/120 | 57.4537 | 2.8727 | 0.1196 | 7.6344 | 8.0524 | 10.1725 | 10.7160 |
| `rotor_abs_8` seed0-5 均值 | $P^+ + P^-$ | 120/120 | 56.1303 | 2.8065 | 0.1169 | 7.5157 | 7.8413 | 9.9648 | 10.4407 |
| `joint_positive_8` seed0-5 均值 | 8 驱动关节 $P^+$ | 120/120 | 55.1245 | 2.7562 | 0.1148 | 7.6088 | 8.0452 | 9.7225 | 10.2898 |

### 六种子均值：训练分布 100 回合

本节使用已有 seed0-5 的 `calibration_summary.json` 和 `episode_costs.csv`。统计口径为：每个 seed 先按上面的训练分布公式算出一行，再对 6 个 seed 取算术平均；`成功回合数` 写平均成功回合数/100。`legacy_joint_abs_10` 的训练 cost 已按 `policy_dt = 0.01s` 换算到能量等效口径。`rotor_mixed_8_alpha050` 和 `joint_abs_8` 目前没有同口径 6-seed 训练分布数据，因此不列入本节表格。

#### 所有回合口径

| 方法 | cost 定义 | 阈值 | 训练分布成功率 | 平均速度 | 平均距离 | 总距离 | 所有回合平均 cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始策略参考 `model_3000` | 原始策略，mixed 基线校准口径 | - | 89.00% | 0.0896 | 2.0462 | 204.6228 | 44.3030 |
| `legacy_joint_abs_10` seed0-5 均值（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 55.4052 | 87.00% | 0.0855 | 1.9293 | 192.9349 | 53.7379 |
| `rotor_positive_8` seed0-5 均值 | $P^+$ | 30.8848 | 88.67% | 0.0844 | 1.9135 | 191.3503 | 30.1800 |
| `rotor_abs_8` seed0-5 均值 | $P^+ + P^-$ | 50.8560 | 84.83% | 0.0847 | 1.8843 | 188.4327 | 49.1603 |
| `joint_positive_8` seed0-5 均值 | 8 驱动关节 $P^+$ | 30.9057 | 87.00% | 0.0853 | 1.9095 | 190.9457 | 29.5966 |

#### 成功回合口径

| 方法 | cost 定义 | 阈值 | 成功回合数 | 平均速度 | 平均距离 | 总距离 | 成功回合平均 cost | 成功回合 cost 是否通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始策略参考 `model_3000` | 原始策略，mixed 基线校准口径 | - | 89/100 | 0.0892 | 2.1404 | 190.4925 | 45.4223 | - |
| `legacy_joint_abs_10` seed0-5 均值（旧 ECO 基线） | 旧 ECO，10 关节 $\lvert\tau\dot q\rvert$，已乘 `0.01s` | 55.4052 | 87.00/100 | 0.0852 | 2.0450 | 177.8417 | 56.0503 | 否，56.0503 > 55.4052 |
| `rotor_positive_8` seed0-5 均值 | $P^+$ | 30.8848 | 88.67/100 | 0.0845 | 2.0272 | 179.7902 | 31.2336 | 否，31.2336 > 30.8848 |
| `rotor_abs_8` seed0-5 均值 | $P^+ + P^-$ | 50.8560 | 84.83/100 | 0.0846 | 2.0313 | 172.3988 | 51.6354 | 否，51.6354 > 50.8560 |
| `joint_positive_8` seed0-5 均值 | 8 驱动关节 $P^+$ | 30.9057 | 87.00/100 | 0.0845 | 2.0278 | 176.4398 | 31.2735 | 否，31.2735 > 30.9057 |
