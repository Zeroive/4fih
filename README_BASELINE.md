# HiF4 优化基线与迭代记录

本文档记录 `solution_collection/solution/solution.py` 的本地迭代结果。后续只有在精度、通过数或精度/时延折中有明确价值时，才保存新的 `solution_vxxx_.py` 快照，并在此追加记录。

## 测试方法

```bash
uv run python self_check_.py --solution_dir solution_collection/solution
```

- 数据：`mini_sample/linear.pt`、`mini_sample/attn.pt`
- Linear 通过阈值：MSE ≤ `1e-3`
- Attention 通过阈值：MSE ≤ `1e-3`
- MSE 数值用于比较算法变化；本地数据只用于调试，不用于编写样本内容特判。
- 时延受系统负载影响，只比较明显趋势。动态路径未增加计算时，数个百分点的差异视为运行波动。

## 版本总览

| 版本 | 核心变化 | 通过数 | Linear 平均 MSE | Attention 平均 MSE | 十项平均 MSE | 平均校准时延 | 平均动态时延 | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v162` | 原始 per-KV-head rotation 搜索 | 6/12 | `1.66782e-3` | `3.91960e-4` | `1.02989e-3` | `4565.62 ms` | `1102.59 ms` | `23.90 s` |
| `v163_` | Weight H256 精修从 4 次增加到 8 次 | 6/12 | `1.64312e-3` | `3.91960e-4` | `1.01754e-3` | `6911.44 ms` | `1537.80 ms` | `33.44 s` |
| `v164_` | Weight H256 精修改为 6 次，平衡精度和时延 | 6/12 | `1.64980e-3` | `3.91960e-4` | `1.02088e-3` | `4905.49 ms` | `1200.06 ms` | `25.94 s` |
| `v165_` | Attention 固定 signed-H64，移除昂贵 rotation 搜索 | 6/12 | `1.64980e-3` | `3.60080e-4` | `1.00494e-3` | `2724.27 ms` | `1108.13 ms` | `20.14 s` |
| `v166_` | 短序列 K 使用 partner-covariance quotient，跳过额外 H256 repair | 7/12 | `1.64980e-3` | `3.32902e-4` | `9.91351e-4` | `2778.13 ms` | `1134.24 ms` | `20.37 s` |
| `v167_` | K metric 按 KV head 做可靠性 covariance shrinkage | 7/12 | `1.64980e-3` | `3.32258e-4` | `9.91029e-4` | `2900.99 ms` | `1141.93 ms` | `20.82 s` |
| `v168_` | 融合 v111 直接 per-head Attention 路径，移除 partner-Hessian | 7/12 | `1.64980e-3` | `2.58932e-4` | `9.54366e-4` | `2498.94 ms` | `669.64 ms` | `15.19 s` |

当前推荐版本：`solution_collection/solution_v168_.py`。当前 `solution.py` 与该版本一致。

## 逐项 MSE

### Linear

| 版本 | 10 tokens | 128 tokens | 512 tokens | 1024 tokens A | 1024 tokens B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v162` | `2.1942e-3` | `1.9064e-3` | `1.4630e-3` | `1.4059e-3` | `1.3696e-3` |
| `v163_` | `2.1857e-3` | `1.8841e-3` | `1.4340e-3` | `1.3701e-3` | `1.3417e-3` |
| `v164_`–`v168_` | `2.1898e-3` | `1.8901e-3` | `1.4411e-3` | `1.3791e-3` | `1.3489e-3` |

结论：8 次精修的 Linear MSE 最低，但校准时延增长过大；6 次精修是当前折中方案。

### Attention

| 版本 | 10 tokens | 128 tokens | 512 tokens | 1024 tokens A | 1024 tokens B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v162`–`v164_` | `1.1904e-3` | `2.9510e-4` | `1.7900e-4` | `1.4928e-4` | `1.4602e-4` |
| `v165_` | `1.0805e-3` | `2.5601e-4` | `1.7477e-4` | `1.4020e-4` | `1.4892e-4` |
| `v166_` | `9.4461e-4` | `2.5601e-4` | `1.7477e-4` | `1.4020e-4` | `1.4892e-4` |
| `v167_` | `9.4461e-4` | `2.5696e-4` | `1.7064e-4` | `1.4017e-4` | `1.4891e-4` |
| `v168_` | `7.4076e-4` | `1.9035e-4` | `1.3298e-4` | `1.1700e-4` | `1.1357e-4` |

结论：`v166_` 首次使短序列 Attention 通过；`v168_` 融合 v111 的直接 per-head 路径后，同时显著降低 Attention MSE 和动态时延。

## 各版本说明

### v162：基线

- 六个公开接口完整。
- Q/K rotation 在 calibration 阶段按 KV head 搜索，但当前数据最终全部选择 identity。
- Attention calibration 搜索开销较大。
- 代码精简前后 MSE 完全一致；精简后从 5680 行降到约 1913 行。

### v163_：8 次 Weight H256 精修

- 五项 Linear MSE 全部下降。
- Linear 平均 MSE 相对 v162 下降约 1.48%。
- 校准开销明显增加，不作为低时延推荐版本。

### v164_：6 次 Weight H256 精修

- Linear 平均 MSE 相对 v162 下降约 1.08%。
- 相比 8 次方案牺牲少量精度，显著降低校准开销。

### v165_：固定 signed-H64

- 所有 Q/K head 使用相同 signed-H64，不改变各头边界。
- 移除未产生收益的 per-head rotation calibration 搜索。
- Attention calibration 从数秒降低到约 0.42 秒。
- Attention 平均 MSE 相对 v162 下降约 8.13%。

### v166_：短序列 K quotient

- 当 `sequence_length <= 64` 时，保留 partner-covariance quotient 结果，跳过额外 H256 hierarchy repair。
- 依据是低 token softmax 区域中额外二次近似精修的可靠性较低，而非针对具体样本内容。
- 10-token Attention MSE 降到 `9.4461e-4`，通过数从 6/12 提升到 7/12。

### v167_：K per-head covariance reliability

- Q partner covariance 保持不收缩，避免短序列退化。
- K partner covariance 根据 calibration 样本间的 off-diagonal 波动，为每个 KV head 独立估计可靠性。
- 当前两个 KV head 的可靠性约为 `0.9159`、`0.9178`。
- 只改变 calibration state 中的 metric，不增加动态量化步骤。

### v168_：v111 直接 per-head Attention

- A/B 确认 v111 的直接 per-head Q/K/V 路径明显优于 v167 的 partner-Hessian 路径。
- 保留当前固定 signed-H64，避免 v111 原版约 7 秒的 rotation calibration 搜索。
- Q 使用变换后的 tensor-self MSE；K 使用 translation quotient；V 使用 tensor-self MSE。
- 不再保存 Smooth 或 partner covariance state，Attention calibration 约为 `0.10 ms`。
- Attention 平均 MSE 相对 v167 下降约 22.1%，十项平均 MSE 下降约 3.7%。
- 平均动态时延从约 `1141.93 ms` 降到 `669.64 ms`。
- 删除失效路径后，`solution.py` 从约 1930 行进一步精简到 1381 行。

## 未保存实验

| 实验 | 结果 | 处理 |
| --- | --- | --- |
| Attention per-head reconstruction rotation | 两个 KV head 均选择 identity；校准慢且端到端不如 signed-H64 | 放弃 |
| Attention-output importance rotation | 两个 KV head 均选择普通 H64；Attention 平均 MSE 约 `3.7424e-4`，比 v166 退化约 12.4% | 放弃 |
| 不同 head 使用 `[H64, signed-H64]` | 本地 checker 将展平 Q/K 直接计算，跨头基变换发生错配，10-token MSE 约 `9.92e-3` | 不用于当前 checker |
| Q/K covariance 同时可靠性收缩 | 中长序列改善，但短序列退化；Attention 平均 MSE 约 `3.3701e-4` | 放弃 |
| 仅 Q covariance 收缩 | 10/128 token 退化，平均无收益 | 放弃 |
| K hierarchy candidate 接受阈值 | 所有长序列 candidate 的 metric 改善均远超阈值，head 选择完全不变 | 不增加无效逻辑 |

## 历史版本复用验证

| 来源 | Attention MSE（10 / 128 / 512 / 1024A / 1024B） | 结论 |
| --- | --- | --- |
| 原始 `solution_v111.py` | `7.6911e-4 / 1.9068e-4 / 1.3316e-4 / 1.1590e-4 / 1.1386e-4` | 精度有效，但 rotation calibration 约 7 秒 |
| v168 低时延融合 | `7.4076e-4 / 1.9035e-4 / 1.3298e-4 / 1.1700e-4 / 1.1357e-4` | 保留 v111 收益，移除冗余搜索，并进一步改善平均 MSE |

## 后续记录模板

新增版本时至少记录：

1. 相对上一版本的单一核心改动及通用性依据。
2. Linear 与 Attention 的五项 MSE。
3. 通过数、平均校准时延、平均动态时延和总耗时。
4. 是否增加动态计算步骤。
5. 若候选失败，记录退化原因但不保存版本文件。
