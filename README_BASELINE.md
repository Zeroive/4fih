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
- 2026-08-21 起，Attention checker 按 `[batch, heads, seq, head_dim]` 执行真实 GQA；更早结果来自展平 fallback，不可与新结果直接比较。

## 真实 GQA 基线

2026-08-21 使用修正后的 checker 串行重测全部 baseline 版本：

| 版本 | 通过数 | Linear 平均 MSE | Attention 平均 MSE | 十项平均 MSE | 平均校准时延 | 平均动态时延 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v162` | 7/12 | `1.66782e-3` | `4.05644e-4` | `1.03673e-3` | `4256.08 ms` | `1039.04 ms` |
| `v163_` | 7/12 | `1.64312e-3` | `4.05658e-4` | `1.02439e-3` | `5212.44 ms` | `1120.76 ms` |
| `v164_` | 7/12 | `1.64980e-3` | `4.05658e-4` | `1.02773e-3` | `4809.19 ms` | `1129.33 ms` |
| `v165_` | 7/12 | `1.64980e-3` | `3.98754e-4` | `1.02428e-3` | `2782.36 ms` | `1162.74 ms` |
| `v166_` | 7/12 | `1.64980e-3` | `4.01088e-4` | `1.02544e-3` | `2727.73 ms` | `1188.58 ms` |
| `v167_` | 7/12 | `1.64980e-3` | `4.00804e-4` | `1.02530e-3` | `2821.35 ms` | `1112.32 ms` |
| `v168_` | 7/12 | `1.64980e-3` | `4.51886e-4` | `1.05084e-3` | `2705.60 ms` | `670.33 ms` |
| `v169_` | 7/12 | `1.64686e-3` | `4.51886e-4` | `1.04937e-3` | `2473.78 ms` | `699.84 ms` |
| `v170_` | 7/12 | `1.64686e-3` | `3.98754e-4` | `1.02281e-3` | `2648.22 ms` | `1120.25 ms` |
| `v171_` | 7/12 | `1.64686e-3` | `3.98149e-4` | `1.02250e-3` | `3220.61 ms` | `1137.96 ms` |
| `v172_` | 7/12 | `1.30276e-3` | `3.98149e-4` | `8.50454e-4` | `10066.14 ms` | `2335.49 ms`* |
| `v173_` | 7/12 | `1.30202e-3` | `3.98149e-4` | `8.50085e-4` | `7260.20 ms` | `1137.85 ms` |
| `v174_` | 7/12 | `1.30202e-3` | `3.98149e-4` | `8.50085e-4` | `7447.54 ms` | `1164.91 ms` |

Attention 逐项结果：

| 版本 | 10 tokens | 128 tokens | 512 tokens | 1024 tokens A | 1024 tokens B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v162` | `7.6651e-4` | `4.3267e-4` | `2.9092e-4` | `2.7889e-4` | `2.5923e-4` |
| `v163_`–`v164_` | `7.6651e-4` | `4.3267e-4` | `2.9095e-4` | `2.7892e-4` | `2.5924e-4` |
| `v165_` | `7.7489e-4` | `4.1870e-4` | `2.7845e-4` | `2.7061e-4` | `2.5112e-4` |
| `v166_` | `7.8656e-4` | `4.1870e-4` | `2.7845e-4` | `2.7061e-4` | `2.5112e-4` |
| `v167_` | `7.8656e-4` | `4.1918e-4` | `2.7843e-4` | `2.6880e-4` | `2.5105e-4` |
| `v168_`–`v169_` | `7.9804e-4` | `4.7382e-4` | `3.4094e-4` | `3.3432e-4` | `3.1231e-4` |
| `v170_` | `7.7489e-4` | `4.1870e-4` | `2.7845e-4` | `2.7061e-4` | `2.5112e-4` |
| `v171_` | `7.7125e-4` | `4.1857e-4` | `2.8147e-4` | `2.6975e-4` | `2.4970e-4` |
| `v172_` | `7.7125e-4` | `4.1857e-4` | `2.8147e-4` | `2.6975e-4` | `2.4970e-4` |
| `v173_` | `7.7125e-4` | `4.1857e-4` | `2.8147e-4` | `2.6975e-4` | `2.4970e-4` |
| `v174_` | `7.7125e-4` | `4.1857e-4` | `2.8147e-4` | `2.6975e-4` | `2.4970e-4` |

所有版本的 5 个 Attention 样本仍全部通过阈值。`v171_` 首次得到当前最低真实 GQA Attention，`v172_` 完整保留该结果并显著降低 Linear/十项平均 MSE；低时延基线仍为 `v168_`。

\* `v172_` 完整 checker 运行时 Attention 同路径时延也异常升高；单独受控 Linear A/B 中，8轮版本平均动态时延约 `817.79 ms`，v171 Linear 约 `700.09 ms`，增幅约 16.8%。表中 `2335.49 ms` 是该次整轮实测值，不能全部归因于新增 Linear 逻辑。

## 旧版展平 fallback 版本总览

以下历史 Attention MSE 和十项平均 MSE 使用旧 checker，仅保留用于追溯，不能代表真实 GQA 收益。

| 版本 | 核心变化 | 通过数 | Linear 平均 MSE | Attention 平均 MSE | 十项平均 MSE | 平均校准时延 | 平均动态时延 | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v162` | 原始 per-KV-head rotation 搜索 | 6/12 | `1.66782e-3` | `3.91960e-4` | `1.02989e-3` | `4565.62 ms` | `1102.59 ms` | `23.90 s` |
| `v163_` | Weight H256 精修从 4 次增加到 8 次 | 6/12 | `1.64312e-3` | `3.91960e-4` | `1.01754e-3` | `6911.44 ms` | `1537.80 ms` | `33.44 s` |
| `v164_` | Weight H256 精修改为 6 次，平衡精度和时延 | 6/12 | `1.64980e-3` | `3.91960e-4` | `1.02088e-3` | `4905.49 ms` | `1200.06 ms` | `25.94 s` |
| `v165_` | Attention 固定 signed-H64，移除昂贵 rotation 搜索 | 6/12 | `1.64980e-3` | `3.60080e-4` | `1.00494e-3` | `2724.27 ms` | `1108.13 ms` | `20.14 s` |
| `v166_` | 短序列 K 使用 partner-covariance quotient，跳过额外 H256 repair | 7/12 | `1.64980e-3` | `3.32902e-4` | `9.91351e-4` | `2778.13 ms` | `1134.24 ms` | `20.37 s` |
| `v167_` | K metric 按 KV head 做可靠性 covariance shrinkage | 7/12 | `1.64980e-3` | `3.32258e-4` | `9.91029e-4` | `2900.99 ms` | `1141.93 ms` | `20.82 s` |
| `v168_` | 融合 v111 直接 per-head Attention 路径，移除 partner-Hessian | 7/12 | `1.64980e-3` | `2.58932e-4` | `9.54366e-4` | `2498.94 ms` | `669.64 ms` | `15.19 s` |
| `v169_` | Linear 加入 block-64 cross-target 补偿（`lambda=0.25`） | 7/12 | `1.64686e-3` | `2.58932e-4` | `9.52896e-4` | `2771.01 ms` | `677.14 ms` | `16.11 s` |

旧 fallback 口径当时推荐 `solution_collection/solution_v169_.py`；该推荐已被上面的真实 GQA 重测结论取代。当前 `solution.py` 以 `v174_` 为精度基线，并额外加入了下文记录的通用 full-Hessian refinement；该改动不另编号，因为本地 2048 维 MSE 没有变化。

## 逐项 MSE

### Linear

| 版本 | 10 tokens | 128 tokens | 512 tokens | 1024 tokens A | 1024 tokens B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v162` | `2.1942e-3` | `1.9064e-3` | `1.4630e-3` | `1.4059e-3` | `1.3696e-3` |
| `v163_` | `2.1857e-3` | `1.8841e-3` | `1.4340e-3` | `1.3701e-3` | `1.3417e-3` |
| `v164_`–`v168_` | `2.1898e-3` | `1.8901e-3` | `1.4411e-3` | `1.3791e-3` | `1.3489e-3` |
| `v169_` | `2.1868e-3` | `1.8840e-3` | `1.4396e-3` | `1.3767e-3` | `1.3472e-3` |
| `v172_` | `1.9021e-3` | `1.4990e-3` | `1.0881e-3` | `1.0093e-3` | `1.0153e-3` |
| `v173_` | `1.9053e-3` | `1.4998e-3` | `1.0864e-3` | `1.0056e-3` | `1.0130e-3` |
| `v174_` | `1.9053e-3` | `1.4998e-3` | `1.0864e-3` | `1.0056e-3` | `1.0130e-3` |

结论：8 次精修的 Linear MSE 最低，但校准时延增长过大；`v169_` 在 6 次精修基础上使五项 MSE 全部下降，动态时延仅小幅增加。

### Attention（旧展平 fallback）

| 版本 | 10 tokens | 128 tokens | 512 tokens | 1024 tokens A | 1024 tokens B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v162`–`v164_` | `1.1904e-3` | `2.9510e-4` | `1.7900e-4` | `1.4928e-4` | `1.4602e-4` |
| `v165_` | `1.0805e-3` | `2.5601e-4` | `1.7477e-4` | `1.4020e-4` | `1.4892e-4` |
| `v166_` | `9.4461e-4` | `2.5601e-4` | `1.7477e-4` | `1.4020e-4` | `1.4892e-4` |
| `v167_` | `9.4461e-4` | `2.5696e-4` | `1.7064e-4` | `1.4017e-4` | `1.4891e-4` |
| `v168_`–`v169_` | `7.4076e-4` | `1.9035e-4` | `1.3298e-4` | `1.1700e-4` | `1.1357e-4` |

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

### v169_：Linear cross-target 补偿

- 在每个 64 维块上，用 calibration 统计量求解 `T = (H + eps I)^-1 (B + eps I)`，近似补偿量化权重输出到目标输出之间的系统误差。
- 动态阶段以 `lambda=0.25` 混合普通输出与补偿输出，再沿用原有 H64/H256 分层量化；不依赖样本内容或固定 token 数。
- 五项 Linear MSE 全部下降，平均值相对 `v168_` 降低约 0.18%；Attention 路径及结果完全不变。
- 稳定复测中平均动态时延从 `669.64 ms` 增至 `677.14 ms`，约增加 1.1%；主要额外总耗时来自 calibration。
- `lambda=0.75` 的本地平均 MSE 略低，但首项退化；为降低本地过拟合风险，保留五项一致改善的 `lambda=0.25`。

### v170_：v169 Linear + v165 Attention

- 完整保留 v169 的 Linear block-64 cross-target 补偿，五项 Linear MSE 不变。
- Attention 恢复 v165 的 reciprocal Q/K Smooth、固定 signed-H64、partner covariance 和 H256 refinement 路径。
- 真实 GQA 下五项 Attention MSE 与 v165 精确一致，平均值相对 v169 降低约 11.8%。
- 十项平均 MSE 从 v169 的 `1.04937e-3` 降至 `1.02281e-3`，为当前最低。
- 平均动态时延从约 `699.84 ms` 回升到 `1120.25 ms`，因此定位为精度优先版本，而非低时延版本。

### v171_：极值稳健 signed-H64 seed2

- 保留 v170 的 Linear、Smooth、partner covariance 与 refinement，只将 Q/K 共同使用的确定性 signed-H64 符号种子从 1 改为 2。
- 同一 KV group 的 Q/K 始终共享完全相同的正交基，因此量化前 GQA logits 不变；动态步骤数没有增加。
- calibration Attention 平均 MSE 从 `4.34315e-4` 降至 `4.33899e-4`，test 平均从 `3.98754e-4` 降至 `3.98149e-4`，最坏样本也改善。
- 收益很小且本地仅有一个 Attention group，作为低风险实验快照保留，是否泛化必须以线上分数为准。

### v172_：H1024/H2048 Linear 精度优先折中

- Attention 完整保留 v171 的 reciprocal Smooth、signed-H64 seed2、partner Hessian 和 K refinement，五项 MSE逐项一致。
- Linear Weight calibration 增加 pairwise hierarchy、H1024 activation covariance 与 quantized-weight Gram refinement；动态 activation 使用 self-MSE 初始化、1轮 H1024 refinement 和8轮 full-H2048 pair-block refinement。
- 根据消融结果，最终 activation state 不保存 H64/H256；同时移除两者比完整小块+大块层级的 Linear 平均 MSE再降低约 0.99%，并减少约 1.25 MiB state。
- Linear 平均 MSE相对 v171 降低约20.9%，十项平均降低约16.8%；本地通过数仍为7/12，因为8轮折中下最后两项略高于 `1e-3`。
- 受控 Linear 动态时延约增加16.8%，但 Weight calibration 明显增加；完整实测 calibration 平均 `10066.14 ms`，定位为精度优先实验版本。
- 逻辑针对任意可被1024整除的维度保留 H1024 refinement，但 full-H2048 pair-block refinement 仅在输入维度恰为2048时生效，线上泛化需重点验证。

### v173_：直接 full-H2048 十轮

- 移除动态 activation 的 H1024 warm start，直接从 self-MSE 初始化进入 full-H2048 refinement，并将轮数从8增至10。
- calibration state 不再保存 `super1024_hessian_blocks`，每个 2048维 Linear state 减少约4 MiB；动态目标只保留完整 `Wq^T Wq`，避免分块目标与最终目标切换。
- Linear 平均 MSE从 `1.30276e-3` 小幅降至 `1.30202e-3`（约0.057%）；前两项略退，后三项改善，Attention 完全不变。
- 同环境相邻复测中，Linear 动态均值从 v172 的约 `769.17 ms` 增至约 `782.71 ms`（约1.76%）；完整 checker 总耗时从约 `30.39 s` 降至 `29.43 s`，但时延仍应视为有负载噪声的参考值。
- 直接 H2048 八轮/九轮平均 MSE分别为 `1.32244e-3` / `1.31090e-3`，说明 H1024 warm start 在相同8轮预算下有效；十轮才能略微超过原组合。

### v174_：动态 Hessian 按输入维度降级

- `k==2048` 完整保留 v173 的 full-H2048 十轮路径；否则互斥选择最大的受支持 block Gram：`k%1024==0 -> H1024`、`k%256==0 -> H256`、`k%64==0 -> H64`，均执行10轮 group refinement，其他情况回退 Self-MSE。
- 最终 state 只保存一个层级，避免重复内存和 H64/H256/H1024/full-H2048 多目标串联冲突。
- 修复 `k<256` 时 H256 reliability 不存在却读取 `mean()` 的原有错误；合成测试确认 `k=64/256/1024` 分别生成并使用正确形状的 block Hessian。
- 本地正式数据维度为2048，因此五项 Linear、五项 Attention 与 v173 逐项一致；完整复测为7/12，平均 calibration `7447.54 ms`、平均动态 `1164.91 ms`、总耗时 `30.18 s`。

### 当前 solution.py：full-Hessian 通用化（未另存版本）

- 将仅支持 2048 维的 `_v181b_pairblock_refine` 泛化为 `_refine_full_hessian_batched`，支持任意 `k % 64 == 0`；最后不足 `block_batch` 的 block 也按实际数量处理。
- `k <= 2048` 时统一保存完整 `k×k` quantized-weight Gram，因此 64/256/512/1024/1536/2048 等维度都保留全部通道相关性；`k > 2048` 才继续按 H1024/H256/H64 降级，控制 state 和计算量。
- 合成验证覆盖 `k=64/256/512/1024/1536`；2048 维新旧 refinement 的所有输出参数逐元素完全一致。
- 完整 checker 的 Linear 五项仍为 `1.9053e-3 / 1.4998e-3 / 1.0864e-3 / 1.0056e-3 / 1.0130e-3`，Attention 五项也与 v174 完全一致。该次平均 calibration/dynamic 为 `9219.11/1461.41 ms`、总耗时 `37.32 s`；由于 2048 路径新旧数值和运算结构等价，将本次时延上浮记为 CPU 负载噪声，不作为算法时延退化结论。
- 本次是通用性和命名改进，没有本地 MSE 收益，因此不创建新的 `solution_vxxx_.py`。

### 当前 solution.py：full-H Weight `chunk_rows=1024` 时延与 MSE 基准

- 将离线 Weight full-H refinement 的行分块从 `128` 扩大到 `1024`；对于本地 `8192×2048` 权重，chunk 数由 `64` 降为 `8`。该参数只改变校准阶段的批处理粒度，不改变动态 activation 路径。
- 完整 checker 为 `9/12`。Linear 五项 MSE 为 `1.9861e-3 / 1.4720e-3 / 1.0194e-3 / 9.3353e-4 / 9.4470e-4`，平均 `1.27115e-3`；Attention 五项为 `7.6906e-4 / 4.5506e-4 / 3.1220e-4 / 3.0811e-4 / 2.8484e-4`，平均 `4.25854e-4`。
- 相对同一实现 `chunk_rows=128` 的相邻运行，Linear calibration 从 `24928.78 ms` 降至 `18509.44 ms`，约下降 `25.8%`；平均 calibration 为 `9490.43 ms`，总耗时从 `47.80 s` 降至 `42.17 s`。
- 平均动态时延为 `1687.61 ms`。chunk 大小不参与动态量化，因此与 `128` 行基准 `1756.90 ms` 的差异视为 CPU 运行波动，不记为算法收益。
- 该结果作为当前 `solution.py` 后续时延与 MSE 对照基准；未单独创建版本快照。
- 随后删除 Attention 动态路径的 `_v159_dynamic_q/_v159_dynamic_k` 转发层，将 `_v158_dynamic_tensor_h256` 重命名为描述实际行为的 `_quantize_attention_tensor_hessian`，并移除已不参与计算的 `lv3_iters/base_mant_iters` 参数。完整复测十项 MSE逐项不变，仍为 `9/12`；平均 calibration/dynamic 为 `9875.73/1540.48 ms`、总耗时 `41.50 s`。此次只减少函数转发和失效参数，不减少张量计算，时延差异仍按 CPU 波动处理。
- Coverage 精简删除旧 `_v39/_v40/_v42` Attention state/permutation 兼容链，并将 Linear geometry 固定到当前实际使用的 `mass_act + max pressure + pmax phase`，移除未选择的 permutation、RMS phase 与 geom pressure 分支。非 coverage 完整复测十项 MSE逐项不变，仍为 `9/12`；coverage 可执行语句从 `1215` 降至 `1142`、未覆盖语句从 `244` 降至 `183`，行覆盖率从 `80%` 提升到 `84%`。剩余整段未覆盖代码主要是主动保留的 K state/Hessian 缺失 fallback。
- 继续移除全部旧 K fallback 兼容链：删除 feature-block candidate merge/score、旧短序列 translation、多轮无 Hessian K quantizer 及其常量；Q/K 动态接口现在直接消费本轮 calibration 生成的 `scale + partner_h64/h256` state。完整非 coverage 复测十项 MSE仍逐项不变，`9/12`，平均 calibration/dynamic 为 `9283.93/1470.88 ms`、总耗时 `38.01 s`。coverage 可执行语句进一步降至 `987`、未覆盖仅 `52`，覆盖率提升到 `95%`，已不存在整个函数完全未调用的情况；剩余未覆盖行都是输入尺寸、空数据和可选返回值防御分支。
- 合并单层内部包装：将 `_safe111_attach`、`_safe111_fixed_attention_base`、`_safe108_attach_cov_state` 直接并入公开 `hif4_calibration_attention`，固定当前实际使用的 H64/H256 state，并使 calibration QKV 从重复 decode 两次降为一次；唯一调用的一行 `_safe130_q` 也内联到 scale refinement。AST 扫描后不再存在“唯一调用且函数体仅一到两条转发语句”的内部 helper。完整复测十项 MSE逐项不变，仍为 `9/12`；本次平均 calibration/dynamic 为 `9925.74/1855.21 ms`、总耗时 `43.67 s`，计时差异按 CPU 波动处理。

### v175_：Linear permutation RMS mass

- 将 Linear `mass_act` permutation 的通道质量从 calibration mean-absolute 改为等样本权重 RMS，即 `sqrt(mean(x²))`；严格 L2 norm 在各通道统计元素数相同时只相差公共尺度，因此 permutation 的实质变化是 L1 importance 改为更强调极值的 L2 importance。
- Linear 五项 MSE 为 `2.0117e-3 / 1.4517e-3 / 1.0111e-3 / 9.2490e-4 / 9.4106e-4`，平均 `1.26809e-3`，相对 mean-absolute 基准 `1.27115e-3` 改善约 `0.24%`；Attention 五项逐项不变，平均仍为 `4.25854e-4`。
- 10-token Linear 首项退化约 `1.29%`，其余四项改善，因此该版本属于平均精度收益而非逐项稳健改善；RMS 对 calibration outlier 更敏感，线上泛化需要重点验证。
- 完整 checker 仍为 `9/12`，平均 calibration/dynamic 为 `7658.51/1478.25 ms`、总耗时 `34.68 s`。仅改变离线 permutation，动态步骤数不变；时延下降主要按 CPU 波动处理。
- 改善版本已保存为 `solution_collection/solution_v175_.py`，并保留在当前 `solution.py`。

### 当前 solution.py：Attention Hessian 按真实 head_dim 泛化

- 移除 Attention scale refinement 中固定的 `256` group：calibration state 改为保存通用 `partner_hessian`，其 Q/K 形状分别为 `[q_num_heads, head_dim, head_dim]` 和 `[kv_num_heads, head_dim, head_dim]`。
- `_refine_attention_scales` 根据 state 中 Hessian 的真实宽度推导 `head_width`、head 数和每 head 的 64-block 数，不再假设每个 head 恰好包含4个 H64 block。
- 合成测试已覆盖 `head_dim=64/128/256/512`，均成功完成 calibration 和动态 Q/K 量化；例如 `head_dim=512, kv_heads=2` 保存 `[2,512,512]` K Hessian。
- 本地真实 `head_dim=256` checker 的十项 MSE逐项与 v175 一致，仍为 `9/12`；平均 calibration/dynamic 为 `7788.92/1368.89 ms`、总耗时 `33.64 s`。动态步骤数没有增加，时延差异按 CPU 波动处理。
- 当前实现要求 `head_dim` 为64的倍数，使 HiF4 的64元素 block 与 head 边界对齐；非64倍数需要额外 padding/permutation 设计，不能直接按当前参数布局表达独立 per-head Hessian。

### v176_：Attention Full-H 初始化

- 将 Q 和 K 中心化候选从“按 H64 独立选择 E6M2 初值，再用 full-head Hessian 调 scale”改为“Self-MSE 物化合法参数、逐 head full-H mantissa 坐标初始化、同一 full-H scale refinement”。mantissa 更新的梯度包含同一 head 内全部通道相关性。
- 删除不再使用的 `_v37_quantize_hessian`、H64 Hessian 拆分逻辑和 Q/K state 中的 `partner_h64`；Attention state 只保存真实 `[num_heads, head_dim, head_dim]` 的 `partner_hessian`，减少冗余 state 和初始化候选搜索。
- Attention 五项 MSE 从 `7.6906e-4 / 4.5506e-4 / 3.1220e-4 / 3.0811e-4 / 2.8484e-4` 改善为 `7.7307e-4 / 4.3115e-4 / 2.9642e-4 / 2.9037e-4 / 2.6766e-4`，平均从 `4.25854e-4` 降至 `4.11734e-4`，改善约3.32%。10-token 样本轻微退化约0.52%，其余四项均改善。
- Linear 五项逐项不变，完整 checker 保持 `9/12`。清理后复测平均 calibration/dynamic 为 `8534.89/1447.28 ms`、总耗时 `35.66 s`；与此前运行的 CPU 波动区间相当，没有观察到明确时延增加。
- 合成验证覆盖 `head_dim=64/128/256/512` 的 full-H mantissa 初始化。改善版本保存为 `solution_collection/solution_v176_.py`，并保留在当前 `solution.py`。

### v177_：Linear permutation 采用 8-channel mass balancing

- 将 `_safe139_massdiff_perm(mass, 64)` 改为 `_safe139_massdiff_perm(mass, 8)`。实际 HiF4 block、Hadamard 和 Hessian 尺寸仍保持不变，只把 permutation 的负载均衡桶从32个64-channel桶细化为256个8-channel桶，使异常通道先在 HiF4 的8-value层级均匀分散，再组合为H64。
- Linear 五项从 `2.0117e-3 / 1.4517e-3 / 1.0111e-3 / 9.2490e-4 / 9.4106e-4` 全部改善为 `1.9790e-3 / 1.3913e-3 / 9.5518e-4 / 8.7744e-4 / 8.9350e-4`；平均从 `1.26809e-3` 降至 `1.21928e-3`，改善约3.85%。通过数从9/12提升到10/12。
- Attention 五项与 v176 逐项完全一致，平均仍为 `4.11734e-4`。完整复测平均 calibration/dynamic 为 `9917.49/1623.70 ms`、总耗时 `42.20 s`；本轮未修改的 Attention 计时也同步上浮，因此主要按 CPU 负载波动记录。动态计算步骤没有增加。
- 改善版本保存为 `solution_collection/solution_v177_.py`，并保留在当前 `solution.py`。

### v178_ / v179_：Attention per-head Full-H lv3/lv2

- 在 v177 的 Attention `Self-MSE → per-head Full-H mantissa → Full-H scale` 后增加逐 head `Full-H lv3 → Full-H lv2`。每个4-value/8-value toggle均使用完整 `head_dim×head_dim` partner Hessian评分，严格下降才接受，并立即用 `H[changed,:]` 更新同一 head 的全局梯度。
- `v178_` 为不含最终 mantissa repair 的保留版本。Attention五项为 `7.7220e-4 / 4.2836e-4 / 2.9302e-4 / 2.8435e-4 / 2.6341e-4`，平均 `4.08268e-4`，相对 v177 的 `4.11734e-4` 改善约0.84%。完整复测平均 calibration/dynamic 为 `10627.86/2232.88 ms`、总耗时 `48.87 s`。
- `v179_` 在 lv2 后再增加1轮 per-head Full-H mantissa repair，五项均进一步改善到 `7.6859e-4 / 4.2175e-4 / 2.8478e-4 / 2.7506e-4 / 2.5351e-4`，平均 `4.00738e-4`，相对 v177 改善约2.67%。Linear五项逐项不变，完整 checker保持10/12；平均 calibration/dynamic 为 `10025.07/2131.23 ms`、总耗时 `47.08 s`。
- 两次相邻运行中，无 repair 版本没有表现出时延优势，计时差异被 CPU负载波动覆盖。因此当前 `solution.py` 采用精度更好的 v179完整路径；用户要求保留的无 repair折中版本单独保存为 `solution_collection/solution_v178_.py`，完整版本保存为 `solution_collection/solution_v179_.py`。

### v180_：每个 KV group 的公共 Q/K permutation（实验快照）

- reciprocal Q/K scale 后，为每个 KV group联合统计1个 K head与关联8个 Q heads的 RMS pressure，生成一个公共 permutation；同组 Q/K在 signed-H64前应用相同置换，因此标准 GQA点积保持不变。随机置换等价性验证的最大 QK数值差约 `1.0e-5`。
- 8-channel mass balancing 的 Attention五项为 `7.8412e-4 / 4.2697e-4 / 2.8201e-4 / 2.6867e-4 / 2.4708e-4`，平均 `4.01770e-4`，比 v179 identity退化约0.26%。
- H64 mass balancing 的五项为 `7.7885e-4 / 4.2349e-4 / 2.8022e-4 / 2.6829e-4 / 2.5073e-4`，平均 `4.00316e-4`，相对 v179 的 `4.00738e-4` 仅改善约0.105%；10/128-token退化，三个长序列改善。完整复测平均 calibration/dynamic 为 `9935.15/1859.81 ms`、总耗时 `43.49 s`。
- H64版本按“有提升则保存”规则保留为 `solution_collection/solution_v180_.py`，但收益过小、短序列方向不一致且在线增加逐 head gather，因此不替换通用性优先的当前 v179主线。

### v181_：Linear `WqᵀW` cross-target 补偿

- 保留当前 RMS smooth geometry，并修正新增表达式为跨 calibration 样本先平均平方、再开根。动态 Linear 的二次项仍为归一化 `H=WqᵀWq`，额外从同一变换域原始权重 `Wt` 构造 `C=WqᵀWt`，使用 `g=(q-y)H+αy(H-Cᵀ)`，使 activation 量化误差能够补偿一部分 weight 量化误差；mantissa、scale、lv3、lv2 全部共享同一 cross-target gradient。
- 相同 RMS geometry 下，`α=0` 五项为 `1.8092e-3 / 1.2767e-3 / 8.4706e-4 / 7.7725e-4 / 7.8322e-4`，平均 `1.09869e-3`；直接保存 BF16 `C` 时，`α=0.5` 平均 `1.07900e-3`，改善约1.79%；`α=1.0` 平均 `1.17359e-3`，退化约6.82%，说明完整补偿过强。
- 最终不分别量化两个大矩阵再在线相减，而是在 calibration 以 FP32 计算并保存 `D=H-Cᵀ`。`α=0.5` 五项进一步改善为 `1.6707e-3 / 1.2029e-3 / 8.0901e-4 / 7.4965e-4 / 7.4965e-4`，平均 `1.03638e-3`，相对同 geometry 的 `α=0` 改善约5.67%，五项全部改善；Attention 五项保持 v179 不变，完整 checker 为10/12。
- `αyD` 在线只计算一次，并由全部 hierarchy stage复用。activation-only 同进程交错计时中，512-token 为 `1188.19/1184.18 ms`（α=0/0.5），两个1024-token分别为 `2150.80/2207.01 ms` 与 `2134.55/2233.90 ms`，长序列开销约2.6%与4.7%。删除从未读取的重复 `full_hessian` state 后，新增 BF16 `D` 替代该矩阵，state 的大矩阵总容量不增加。该收益版本保存为 `solution_collection/solution_v181_.py`，并作为当前 `solution.py`。

### v177_：W/A/Q/K/V 反量化重建 MSE

口径为本地 NVFP4 反量化张量经过当前完整变换和 HiF4 量化后再反量化。`变换域`直接比较量化器实际输入，`还原域`撤销 Smooth/permutation/Hadamard/QK reciprocal scale 后与原逻辑张量比较。

| 张量 | 变换域 MSE | 还原域 MSE |
| --- | ---: | ---: |
| W | `2.45529e-6` | `2.47934e-6` |
| A（五组平均） | `2.53220e-3` | `2.77844e-3` |
| Q（五组平均） | `8.62629e-3` | `1.06086e-2` |
| K（五组平均） | `6.43602e-1` | `8.39369e-1` |
| K centered（五组平均） | — | `7.01218e-3` |
| V（五组平均） | `1.49745e-3` | `1.49745e-3` |

K普通 MSE受 softmax-invariant translation 主导，不能直接代表 Attention 损失，因此主要参考 K centered。

| tokens | A还原域 | Q还原域 | K centered | V | Softmax逐元素 | Softmax每行平方L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `3.36453e-3` | `9.88417e-3` | `3.46045e-3` | `1.78178e-3` | `6.65531e-5` | `6.65531e-4` |
| 128 | `3.04896e-3` | `1.04005e-2` | `6.36481e-3` | `1.54803e-3` | `5.03604e-6` | `6.44613e-4` |
| 512 | `2.54128e-3` | `1.08353e-2` | `8.03085e-3` | `1.39023e-3` | `1.34978e-6` | `6.91087e-4` |
| 1024-A | `2.54255e-3` | `1.09262e-2` | `8.58219e-3` | `1.41029e-3` | `6.93882e-7` | `7.10536e-4` |
| 1024-B | `2.39487e-3` | `1.09969e-2` | `8.62262e-3` | `1.35691e-3` | `6.69370e-7` | `6.85435e-4` |
| 五组平均 | `2.77844e-3` | `1.06086e-2` | `7.01218e-3` | `1.49745e-3` | `1.48604e-5` | `6.79440e-4` |

Softmax平均逐行 KL 为 `4.87319e-3`。普通逐元素 MSE随序列长度自然下降，因此跨长度更适合比较每行概率向量平方L2。

#### Q/K/V/Softmax 大值区域

按原始绝对值选择每个测试张量的 top 10%～50%；Softmax 对每个 query/head 概率行独立选择最大概率。表内为五组测试平均 selected-position MSE。

| Top比例 | Q | K centered | K raw | V | Softmax概率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | `2.40018e-2` | `1.01464e-2` | `4.79968` | `3.63171e-3` | `8.03137e-5` |
| 20% | `1.83264e-2` | `8.98858e-3` | `2.86287` | `2.60357e-3` | `5.62979e-5` |
| 30% | `1.56686e-2` | `8.40658e-3` | `2.10628` | `2.18728e-3` | `4.31876e-5` |
| 40% | `1.41389e-2` | `8.02919e-3` | `1.69441` | `1.95989e-3` | `3.45545e-5` |
| 50% | `1.31180e-2` | `7.77197e-3` | `1.42622` | `1.81722e-3` | `2.84654e-5` |

#### W/A/MatMul/Attention output 大值区域

按原始绝对值选择 top 10%～50%。W为单个权重张量，其余为五组测试平均 selected-position MSE。

| Top比例 | W | A | W@A输出 | Attention输出 |
| ---: | ---: | ---: | ---: | ---: |
| 10% | `3.17257e-6` | `5.08736e-3` | `1.84313e-3` | `1.11108e-3` |
| 20% | `2.91530e-6` | `3.88348e-3` | `1.65359e-3` | `8.41196e-4` |
| 30% | `2.83249e-6` | `3.44643e-3` | `1.53053e-3` | `7.13353e-4` |
| 40% | `2.72717e-6` | `3.21932e-3` | `1.43022e-3` | `6.29313e-4` |
| 50% | `2.65311e-6` | `3.07200e-3` | `1.34909e-3` | `5.67431e-4` |

| Top10%相对全部位置平均 | W | A | W@A输出 | Attention输出 | Softmax概率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 倍率 | `1.28×` | `1.83×` | `1.51×` | `2.70×` | `5.4×` |

这些是本地 mini-sample 的 reconstruction诊断，不用于推断线上数据分布。

## 未保存实验

| 实验 | 结果 | 处理 |
| --- | --- | --- |
| Attention per-head reconstruction rotation | 两个 KV head 均选择 identity；校准慢且端到端不如 signed-H64 | 放弃 |
| Attention-output importance rotation | 两个 KV head 均选择普通 H64；Attention 平均 MSE 约 `3.7424e-4`，比 v166 退化约 12.4% | 放弃 |
| 不同 head 使用 `[H64, signed-H64]` | 本地 checker 将展平 Q/K 直接计算，跨头基变换发生错配，10-token MSE 约 `9.92e-3` | 不用于当前 checker |
| Q/K covariance 同时可靠性收缩 | 中长序列改善，但短序列退化；Attention 平均 MSE 约 `3.3701e-4` | 放弃 |
| 仅 Q covariance 收缩 | 10/128 token 退化，平均无收益 | 放弃 |
| K hierarchy candidate 接受阈值 | 所有长序列 candidate 的 metric 改善均远超阈值，head 选择完全不变 | 不增加无效逻辑 |
| 用 cross-target objective 完全替换现有 Linear hierarchy | `lambda=0.25` 相对该简化路径自身略有改善，但整体明显差于 `v168_` | 保留现有 hierarchy，仅叠加补偿 |
| cross-target 补偿仅保留对角项 | 各有效 `lambda` 均劣于完整 64x64 补偿；`lambda=0.25` 平均约 `1.66436e-3` | 放弃 |
| 真实 GQA per-KV 独立 rotation | calibration 选择 `(seed2, seed1)`，但 test 平均约 `4.00307e-4`，差于统一 seed1/seed2 | 放弃，避免逐 head 过拟合 |
| Q/K Smooth `beta={0,0.25,0.75}` | calibration 与 test 均由 `beta=0.5` 最优 | 保留 `beta=0.5` |
| K MAD-clipped robust translation center | test 平均略降，但 calibration 退化且 128-token 明显变差 | 放弃，泛化信号不一致 |
| Q/K/V tail-aware scale（top-value 加权） | Attention 平均退化到约 `4.04882e-4`，短序列最明显 | 放弃 |
| 仅 V tail-aware scale | Attention 平均退化到约 `4.04333e-4` | 放弃 |
| V attention-mass importance | HiF4 参数按 token/block 独立，统一 token 权重不改变该 block 的 argmin | 当前格式下不增加无效逻辑 |
| softmax-sensitivity covariance | calibration 平均改善约 2.7%，test 平均退化到约 `4.00864e-4` | 放弃，calibration 过拟合 |
| E6M2 anchor 向下扩展到 `-2/-4/-8` | 三组真实 GQA 的十项 MSE 均与 v171 一致；最激进 `-8..+4` 的 Linear/Attention 平均仍为 `1.64686e-3` / `3.98148e-4`。对变换后 test Q/K/V 的 base-SSE 逐 block 统计中，`<-1` 被选次数均为 0（Q 172672 blocks，K/V 各 21584 blocks） | 放弃；额外低 scale 候选无端到端收益 |
| Attention 两轮 permutation | 第一轮按 robust outlier score 分组后 test 平均 `4.00243e-4`；第二轮按 Hessian-weighted residual 集中分组为 `4.02958e-4`，跨 block 均衡为 `4.01525e-4`，均差于 v171 的 `3.98149e-4`；两轮版 calibration 约 `6.9–11.3 s` | 实验实现保留，不合入 solution；存在 calibration 过拟合且开销过大 |
| `solution_tmp.py` 动态 Linear 移除小块 Hessian | 完整 H64+H256 平均 `1.24839e-3`；仅去 H64 为 `1.24224e-3`；仅去 H256 为 `1.24924e-3`；同时去 H64/H256 最佳，为 `1.23598e-3`。离线 Weight 参数保持相同，只消融 activation state/动态路径 | 小块目标与 H1024/H2048 存在冲突；值得形成 tmp 精简版后串行复测时延 |
| Linear `full-H2048 -> H64` 后处理 | 在当前 full-H2048 mantissa + scale/lv3/lv2 完整 refinement 后，额外按 full Hessian 的 64×64 对角块执行1轮 mantissa refinement。Linear 五项从 `2.0117e-3 / 1.4517e-3 / 1.0111e-3 / 9.2490e-4 / 9.4106e-4` 退化为 `2.1304e-3 / 1.7093e-3 / 1.2789e-3 / 1.2153e-3 / 1.1955e-3`，平均从 `1.26809e-3` 升至 `1.50588e-3`，退化约18.75%；通过数由9/12降至7/12 | 放弃并恢复；H64 忽略跨 block 项，局部接受的更新会破坏已经优化好的 full-H2048 全局目标 |
| Linear H8 Hadamard | 保留 v177 的8-channel mass balancing，只将 Linear 成对 Hadamard 从H64改成H8；Linear 五项为 `2.0048e-3 / 1.4210e-3 / 9.8926e-4 / 9.0786e-4 / 9.2953e-4`，平均 `1.25049e-3`，比 v177 H64 的 `1.21928e-3` 退化约2.56%。Attention 五项逐项不变 | 放弃并恢复H64；permutation适合按8通道细分负载，但旋转需要覆盖完整64-value量化块，才能摊平不同8-value小组之间的极值 |
| Attention signed-H256 rotation | `head_dim=256` 时将每个 head 的四个独立 signed-H64 改为一个完整 signed-H256；Q/K 成对旋转最大点积数值差约 `9.5e-6`。Attention 五项为 `8.0868e-4 / 4.3170e-4 / 2.9300e-4 / 2.8511e-4 / 2.5922e-4`，平均 `4.15542e-4`，比 v177 signed-H64 的 `4.11734e-4` 退化约0.93%。后三个长序列样本改善，但10-token退化约4.61%，128-token也轻微退化 | 放弃并恢复 signed-H64；H256全局平滑有利于长序列，但短序列的少量关键方向被扩散到更多64-value量化块，平均精度与稳健性不如H64 |
| v179 Full-H lv2/lv3 + signed-H256 | 在包含 per-head Full-H mantissa/scale/lv3/lv2/mantissa repair 的 v179 上重新测试完整 signed-H256。Attention五项为 `7.9753e-4 / 4.2398e-4 / 2.8187e-4 / 2.7254e-4 / 2.4870e-4`，平均 `4.04924e-4`，仍比 v179 signed-H64 的 `4.00738e-4` 退化约1.04%。后三个长序列样本改善，但10-token退化约3.77%，128-token也略差 | 放弃并恢复 signed-H64；Full-H hierarchy能补偿部分跨H64误差，但没有消除全head旋转的短序列损失 |
| Attention signed-H8 rotation | 将每个 Attention head 的旋转从若干独立 signed-H64 改为更多独立 signed-H8，其他 v177 逻辑不变。Attention 五项为 `7.8872e-4 / 4.4053e-4 / 3.0530e-4 / 2.9627e-4 / 2.7314e-4`，平均 `4.20792e-4`，比 signed-H64 的 `4.11734e-4` 退化约2.20%；五项全部变差 | 放弃并恢复 signed-H64；H8 混合范围不足，无法在共享一级E6M2 scale的完整64-value block内充分摊平异常值 |
| Linear `H64 -> global permute -> H64` | 关闭 H1024/H2048 时，固定 perfect-shuffle 平均 MSE `2.48232e-3`、数据驱动 pressure-balanced permutation 为 `2.50572e-3`，Linear 动态均约 `121–129 ms`；恢复与 v172 相同的 H1024 + 8轮 H2048 后，数据驱动版平均 `1.49348e-3`、动态约 `793.66 ms`，仍比 v172 的 `1.30276e-3` 退化约14.6%，且 Linear calibration 增至约 `15.29 s` | 放弃；第二层 H64 使第一层形成的量化友好局部结构和 covariance 分块重新变密，额外 refinement 只能部分追回精度 |
| Linear normalized Gram `Diagonal + Low-rank` | 以量化权重 `Wq` 的随机 SVD 构造 `D + BB^T`，替换动态 H1024/full-H2048，均使用8轮 refinement。rank-32/64/128 的 Linear 平均 MSE分别为 `1.64958e-3` / `1.60632e-3` / `1.55216e-3`，state 约 `132/260/516 KiB`；rank-128 仍比 v172 的 `1.30276e-3` 退化约19.1%。各次原始 Linear 动态均值约 `274–438 ms`，明显低于 full-H2048 的受控约 `817.79 ms`，但 CPU负载波动较大；随机低秩分解使 Linear calibration 约为 `14.3–15.5 s` | 不合入精度优先主线；权重 Gram 的谱不够低秩，适合作为低内存/低时延分支，而不能无损替代 full-H2048 |
| full-Hessian refinement 32轮 | Linear 五项为 `1.8303e-3 / 1.4319e-3 / 1.0342e-3 / 9.5436e-4 / 9.6453e-4`，平均 `1.24306e-3`，相对10轮的 `1.30202e-3` 改善约 `4.53%`；通过数从 `7/12` 提升到 `9/12`。相邻运行 Linear 动态均值从约 `1053 ms` 增至 `2839 ms`，约为 `2.70×`；整轮平均动态时延从 `1461.41 ms` 增至 `2451.19 ms` | 精度有收益，但时延增长过大；恢复10轮，不保存版本 |
| per-Q-head Attention sensitivity Hessian | 前3个 calibration 样本按 `p²·||V-O||²` 构造每个 Q head 的 Value-aware Hessian，后2个样本按真实非因果 GQA output MSE，在 shared-H 与 sensitivity-H 的 `α=0/0.25/0.5/0.75/1` shrinkage 中逐 head 选择。16 heads 中仅4个选择 `α=0.25`，其余12个保留 shared-H；测试五项为 `7.7818e-4 / 4.1972e-4 / 2.8201e-4 / 2.6927e-4 / 2.5017e-4`，平均 `3.99868e-4`，比 v171/v174 的 `3.98149e-4` 退化约0.43%；Attention calibration 增至约 `25.30 s` | 放弃；held-out calibration 的 per-head 选择仍未泛化，且校准开销过大 |
| 每个 Q head 提供 K-Hessian候选 | 为2个 KV heads 分别保存其关联8个 Q heads 的独立 `Q_hᵀQ_h`，动态 K 对每个 Hessian各生成一套候选，再使用该 KV group 的聚合 Q-Hessian逐 KV head统一评分；保留原 quotient 与 aggregate-H 候选。新增 state 形状为 `[2,8,256,256]`。五项 Attention MSE 与基线逐项完全一致，平均仍为 `3.98149e-4`；专项动态均值约 `3108.74 ms`，明显高于相邻基线路径约 `1.87–2.15 s` | 放弃；独立 Q-head Hessian候选未击败原候选（或生成相同参数），只有额外时延和约2 MiB state |
| V attention-weighted feature Full-H | 用校准 `softmax(QK)` 的平方列和为 token 加权，构造每个 KV head 的 `head_dim×head_dim` V covariance；动态 V 比较 Self-MSE 与该 Full-H hierarchy 候选并逐 head 选优。Attention 五项为 `9.9915e-4 / 5.1384e-4 / 3.3836e-4 / 3.1996e-4 / 2.9458e-4`，平均 `4.93181e-4`，比 v179 的 `4.00738e-4` 退化约23.1%；动态均值升至 `2427.46 ms` | 放弃并恢复 v179；真实 V Hessian 位于 token 维且为 `P^T P`，把 token importance 投影为特征 covariance 引入了错误的跨特征耦合 |

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
