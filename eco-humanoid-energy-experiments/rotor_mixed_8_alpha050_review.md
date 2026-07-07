# `rotor_mixed_8_alpha050` 横向评估说明

本文只针对 `rotor_mixed_8_alpha050`（8 电机转子混合功，alpha=0.5）做横向评估口径整理。核心原则是：mixed cost（混合成本）只是训练约束，不能单独作为方法优劣结论。真正比较时必须同时看正向转子能耗、反向转子能耗、单位距离能耗、速度、训练分布稳定性和步态相位。

## 必须导出的字段

后续每次 fixed-command（固定指令）评估都应保留：

- `episode_summary.csv`
- `step_timeseries.csv`
- `phase_summary.csv`
- `metadata.json`

其中 `episode_summary.csv` 和 `step_timeseries.csv` 至少要直接包含：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `rotor_positive_energy_8` | 8 电机转子正向机械能 $E^+_{\mathrm{rotor},8}$ | 判断 mixed 是否真的降低正向驱动功 |
| `rotor_negative_energy_8` | 8 电机转子反向机械能绝对累计 $E^-_{\mathrm{rotor},8}$ | 判断制动/潜在回收侧是否异常 |
| `rotor_abs_energy_8` | $E^+_{\mathrm{rotor},8}+E^-_{\mathrm{rotor},8}$ | 对齐 `rotor_abs_8` 对照 |
| `joint_positive_energy_10` | 10 关节侧正向机械能 | 检查转子侧节能是否只是能量口径迁移 |
| `joint_negative_energy_10` | 10 关节侧反向机械能绝对累计 | 检查关节侧制动功变化 |
| `distance_x` | 世界坐标系 x 方向前进距离 | 做单位距离归一化 |
| `mean_velocity_x` | 回合平均 x 方向速度 | 判断是否靠减速省能 |

`phase_summary.csv` 必须用于检查：

- `double_support`（双支撑）时间占比是否被压得过短。
- `left_support`/`right_support`（左右单支撑）是否出现明显不对称。
- `flight_or_transition`（腾空或过渡）是否异常增大。
- 各相位中的 `rotor_positive_energy_8` 和 `rotor_negative_energy_8` 分布是否异常集中。

说明：历史已生成的部分 CSV 没有直接写出 `rotor_abs_energy_8`。本文表格对历史结果使用

```text
rotor_abs_energy_8 = rotor_positive_energy_8 + rotor_negative_energy_8
```

后续重新运行 `evaluate_energy.py` 时，该字段应直接出现在 `episode_summary.csv`、`step_timeseries.csv` 和 `phase_summary.csv` 中。

## 评估口径

固定评估采用 `v_x = 0.1 m/s`、20 回合、无随机化、无外部扰动。为了和 mixed seed（随机种子）0 公平比较，本文当前主表使用 seed0 对照，而不是把 mixed 单种子和其他方法六种子均值混在一起。

固定评估数据源：

| 方法 | 数据目录 |
| --- | --- |
| baseline（原始基线）`model_3000` | `energy_evaluations/mixed_alpha05_model3000_vx010_20ep` |
| `rotor_positive_8` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/rotor_positive_8/seed0` |
| `rotor_abs_8` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/rotor_abs_8/seed0` |
| `legacy_joint_abs_10` seed0 | `energy_evaluations/fixed_command_vx010_20ep/model4000_keepopt/legacy_joint_abs_10/seed0` |
| `rotor_mixed_8_alpha050` seed0 | `energy_evaluations/eval_rm8a05_cl40895_keepopt_s0_model4000_vx010_20ep` |

单位距离能耗使用总和比值：

```text
E_pos / d_x = sum(rotor_positive_energy_8) / sum(distance_x)
E_neg / d_x = sum(rotor_negative_energy_8) / sum(distance_x)
E_abs / d_x = (sum(rotor_positive_energy_8) + sum(rotor_negative_energy_8)) / sum(distance_x)
E_mix / d_x = (sum(rotor_positive_energy_8) + 0.5 * sum(rotor_negative_energy_8)) / sum(distance_x)
```

训练分布使用 100 回合 calibration（校准）结果。阈值通过不只看 all-episode（全部回合）平均，还要看 `cost1_success_episodes.mean` 是否低于对应阈值，因为短摔倒回合可能把 all-episode 平均成本虚假拉低。

## 当前 mixed seed0 结论

1. 正向转子总能耗是否低于 `rotor_positive_8`？

是，但幅度很小。mixed seed0 的 `rotor_positive_energy_8` 回合均值为 `21.1038`，`rotor_positive_8` seed0 为 `21.2575`，mixed 低约 `0.72%`。

但单位距离后结论反过来：mixed seed0 的 `E_pos/d_x = 7.4248`，`rotor_positive_8` seed0 为 `7.3911`，mixed 高约 `0.46%`。因此不能只说 mixed 正向能耗更低；它的总能耗略低，单位距离正向能耗没有低于主方法。

2. 反向转子能耗是否低于原始 baseline？

是。mixed seed0 的 `rotor_negative_energy_8` 回合均值为 `22.3563`，baseline 为 `23.3767`，低约 `4.37%`；单位距离 `E_neg/d_x` 从 `8.3499` 降到 `7.8655`，低约 `5.80%`。

3. 单位距离正向能耗是否接近或优于 `rotor_abs_8`？

接近，并在 seed0 口径下略低。mixed seed0 的 `E_pos/d_x = 7.4248`，`rotor_abs_8` seed0 为 `7.4331`，mixed 低约 `0.11%`。

但这不是整体胜利：mixed 的 `E_neg/d_x = 7.8655`，高于 `rotor_abs_8` seed0 的 `7.7111`；`E_mix/d_x = 11.3575`，也高于 `rotor_abs_8` seed0 的 `11.2887`。

4. 速度有没有被压低？

没有压低到 baseline 以下。mixed seed0 的 `mean_velocity_x = 0.1184 m/s`，baseline 为 `0.1166 m/s`，高约 `1.52%`。不过 mixed 低于 `rotor_positive_8` seed0 的 `0.1198 m/s`，也低于 `rotor_abs_8` seed0 的 `0.1216 m/s`。

5. 步态相位有没有异常？

没有看到双支撑时间被压得很短。成功回合的相位时间占比如下：

| 方法 | `double_support` | `left_support` | `right_support` | `flight_or_transition` |
| --- | ---: | ---: | ---: | ---: |
| baseline `model_3000` | 19.10% | 40.05% | 40.56% | 0.29% |
| `rotor_positive_8` seed0 | 19.86% | 38.53% | 41.31% | 0.29% |
| `rotor_abs_8` seed0 | 18.77% | 39.58% | 41.36% | 0.29% |
| `rotor_mixed_8_alpha050` seed0 | 19.88% | 40.10% | 39.73% | 0.29% |
| `legacy_joint_abs_10` seed0 | 19.93% | 38.60% | 41.18% | 0.29% |

mixed 的 `double_support = 19.88%`，略高于 baseline，也接近 `rotor_positive_8` 和 `legacy_joint_abs_10`。因此当前没有“通过压短双支撑来省能”的异常迹象。

## mixed 是否应该继续扩种子

当前不建议把 `rotor_mixed_8_alpha050` 直接作为主线扩种子。理由是：

- fixed-command 口径下，mixed 相对 baseline 有效，但单位距离正向能耗没有低于 `rotor_positive_8` seed0。
- mixed 的 `E_pos/d_x` 只比 `rotor_abs_8` seed0 略好，优势约 `0.11%`，不足以抵消训练分布稳定性劣势。
- 训练分布中 mixed seed0 成功率 `88%`，低于 `rotor_positive_8` seed0 的 `92%`；摔倒数 `12`，高于主方法的 `8`。
- mixed seed0 的成功回合成本 `41.6067` 高于阈值 `40.8955`，成功回合 cost（成本）未通过。

因此当前更稳的表述是：mixed alpha 0.5 证明了“加入半权重负机械功惩罚仍能节能”，但还没有证明它优于 `rotor_positive_8` 或 `rotor_abs_8`。

## 汇总表

下表使用 seed0 fixed-command 20 回合评估和 seed0 训练分布 100 回合校准。baseline 的训练分布成功率来自原始 `model_3000` 在 mixed cost 口径下的 baseline calibration（基线校准），没有对应阈值通过判断。

| 方法 | cost 定义 | 阈值 | 固定成功率 | $E_{pos}/d_x$ | $E_{neg}/d_x$ | 训练分布成功率 | 摔倒数 | 成功回合 cost 是否通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline `model_3000` | 原始策略 | - | 20/20 | 8.0737 | 8.3499 | 89% | 11 | - |
| `rotor_positive_8` seed0 | $P^+$ | 30.8848 | 20/20 | 7.3911 | 7.8181 | 92% | 8 | 否，30.9197 > 30.8848 |
| `rotor_abs_8` seed0 | $P^+ + P^-$ | 50.8560 | 20/20 | 7.4331 | 7.7111 | 86% | 14 | 否，51.5796 > 50.8560 |
| `rotor_mixed_8_alpha050` seed0 | $P^+ + 0.5P^-$ | 40.8955 | 20/20 | 7.4248 | 7.8655 | 88% | 12 | 否，41.6067 > 40.8955 |
| `legacy_joint_abs_10` seed0 | 旧 ECO，10 关节 $|\tau\dot q|$ | 5540.5193 | 20/20 | 7.4846 | 7.8795 | 88% | 12 | 是，5397.6021 < 5540.5193 |
