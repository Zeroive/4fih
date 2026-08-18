# 当前六接口无法独立实现的论文

以下方法不能仅靠 `solution_demo.py` 的六个返回接口忠实实现。这里不生成看似可运行、实际删掉论文核心路径的 solution。

## Atom

Atom 需要将固定 outlier channels 保留为 INT8/FP16，并计算低精度主 GEMM 与高精度 outlier GEMM 的和。当前接口只能为整个 Tensor 返回 HiF4 五级参数，无法返回高精度 Weight/Activation 旁路或第二个 GEMM。

至少需要新增：

- outlier channel indices；
- high-precision Weight slice；
- high-precision dynamic Activation slice；
- 双 GEMM 及累加 hook。

## OSC

OSC 同样依赖 profile-driven outlier LUT、HiF4 主路径和高精度补偿路径。当前 state 虽然能保存 LUT，但动态函数无法返回高精度 outlier values，GEMM 接口也没有 correction accumulation hook。

至少需要新增：

- 每个 64-group 的离线 outlier LUT；
- dynamic high-precision outlier return；
- Weight correction table；
- compensation GEMM/accumulation hook。

## FPTQuant

FPTQuant 的关键变换跨越当前量化函数边界，包括 pre-RoPE Q/K transform、V 与 Output projection 的互逆 transform，以及 residual token scaling。当前接口拿到的是 projection 后的 Q/K/V，且没有 RoPE、Output projection、residual 或 inverse-transform hook。

至少需要新增：

- pre-RoPE Q/K projection hook；
- V projection 与 Output projection 的成对变换 hook；
- residual 输入和逆 scaling hook。

只在现有 Q/K 输出上乘一个普通旋转或只把 outlier channel 重排后仍全部量化成 HiF4，都不等价于上述论文方法。
