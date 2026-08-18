# HiF4 迭代版本说明

所有版本保留 v0 的六个公开接口，后一个版本在前一个版本之上增加一项可独立消融的能力。

| 文件 | 相比前版新增内容 | 校准选择目标 | 相对开销 |
| --- | --- | --- | --- |
| `hif4_solution_v0.py` | 多起点 scale、精确 lv2/lv3、固定 Smooth、固定 magnitude sort | 加权张量 MSE | 1x |
| `hif4_solution_v1.py` | 小样本搜索 Linear Smooth alpha，并允许自动退回不做 Smooth | 量化后 `XWᵀ` 相对 MSE | 8 个轻量候选 + 1 次完整量化 |
| `hif4_solution_v2.py` | 在 identity、magnitude sort、zigzag balance 中选择 regroup | 量化后 `XWᵀ` 相对 MSE | 24 个轻量候选 + 1 次完整量化 |
| `hif4_solution_v3.py` | 搜索 Smooth-QK 强度，联合评估完整校准 token 的 QK logits | 量化后 `QKᵀ` 相对 MSE | 额外 5 个 64-token 候选 |
| `solution_awq.py` | 固定闭式 AWQ saliency + W/A RMS balance | 无搜索 | 一次统计 + 一次直接 HiF4 转换 |
| `solution_permute_second_moment.py` | 固定 α=0.5 的 W/A joint second-moment sort | 无候选搜索 | 一次排序 |
| `solution_permute_alpha6.py` | 6 个 α 排序，以真实 activation-weighted HiF4 loss 选优 | 只搜索 permutation | 6 次离线 Weight 转换 |
| `solution_permute_local_swap.py` | 最佳 α sort 初始化 + targeted two-group swap | 只搜索 permutation | 有界局部搜索 |

## 建议实验顺序

1. 先对 v0、v1 做 A/B，确认 output-aware alpha search 的收益。
2. 再跑 v2，读取 `activation_state["permutation_strategy"]` 和
   `activation_state["calibration_relative_output_mse"]`，观察不同层选择是否稳定。
3. 最后跑 v3，读取 `q_state["smooth_beta"]` 和
   `k_state["calibration_relative_qk_mse"]`。

## 设计约束

- 搜索只使用最多 64 个 Linear calibration token、64 个 Attention token和 64 行 Weight；胜出策略才对完整 Weight 做一次正常量化。
- v1–v3 的动态 A/Q/K/V 将 hierarchy 搜索从 v0 的 21 次降到 6 次；完整 Weight 仍使用 v0 的高质量搜索。
- v1/v2 会保留 no-smoothing 候选；搜索结果不会被迫采用 Smooth。
- v2 的 permutation 对 Weight 与动态 Activation 使用同一个索引，因此 full-precision GEMM 等价。
- v3 的 Q/K 缩放互为倒数，因此 full-precision QK logits 等价。
- V 不做 rotation/permutation，因为当前接口没有 O projection 或逆变换 hook。

每个版本都是完整独立的单文件，不依赖目录中的其他 solution。

## AWQ 快速消融

`solution_awq.py` 默认使用 `beta=1, gamma=0.25`。若要做纯 W/A balance，设置
`AWQ_SALIENCY_GAMMA = 0.0`；若要更接近 activation-only fixed AWQ，设置
`AWQ_BALANCE_BETA = 0.0, AWQ_SALIENCY_GAMMA = 0.5`。
