# CM2026 Project 2 实验报告

## 实验设置

- 阿里云ecs.gn7i-c16g1.4xlarge 实例
- GPU: NVIDIA A10
- PyTorch: 2.8.0+cu128
- Triton: 3.4.0
- 推理配置: `dpm-solver`, 20 steps, 20 个预提取 prompt
- 预训练权重: `PixArt-XL-2-1024-MS.pth` + `sd-vae-ft-ema`
- benchmark 配置: batch=2, dtype=fp16, `N={2048,4096,8192,16384}`, `H={8,16}`, `D={64,128}`, warmup=10, iters=50

## 实现概述

本实验实现了四类 attention。

- Vanilla attention: 直接计算 `softmax(QK^T/sqrt(d))V`，作为纯 PyTorch 正确性基线。
- Triton FA2: 使用分块加载 Q/K/V 和 online softmax，避免显式构造完整 attention matrix。
- Block-Sparse attention: 先对 Q/K 按 block 做 mean pooling，再计算 block-level score，为每个 query block 选择 top-k key blocks，Triton kernel 只访问被选中的 K/V block。
- Sparse Int8 attention: 在 sparse attention 基础上对 Q/K 做 per-block symmetric int8 量化，V 保持 fp16。Q 的 scale 合并 `1/log(2)/sqrt(d)`，kernel 中使用 `exp2` 完成 softmax。

## 任务 1: T2I 生成与速度

### 1.1 生成结果

以下是 2 个代表性 prompt 的生成对比。

#### Prompt 000

| SDPA | Vanilla | FA2 |
| --- | --- | --- |
| ![](output/report_assets/p000_sdpa.jpg) | ![](output/report_assets/p000_vanilla.jpg) | ![](output/report_assets/p000_fa2.jpg) |

| Sparse 0.3 | Sparse 0.5 | Sparse 0.8 |
| --- | --- | --- |
| ![](output/report_assets/p000_sparse03.jpg) | ![](output/report_assets/p000_sparse05.jpg) | ![](output/report_assets/p000_sparse08.jpg) |

| Sparse 0.9 | Sparse 1.0 | Int8 0.5 |
| --- | --- | --- |
| ![](output/report_assets/p000_sparse09.jpg) | ![](output/report_assets/p000_sparse10.jpg) | ![](output/report_assets/p000_int805.jpg) |

#### Prompt 001

| SDPA | Vanilla | FA2 |
| --- | --- | --- |
| ![](output/report_assets/p001_sdpa.jpg) | ![](output/report_assets/p001_vanilla.jpg) | ![](output/report_assets/p001_fa2.jpg) |

| Sparse 0.3 | Sparse 0.5 | Sparse 0.8 |
| --- | --- | --- |
| ![](output/report_assets/p001_sparse03.jpg) | ![](output/report_assets/p001_sparse05.jpg) | ![](output/report_assets/p001_sparse08.jpg) |

| Sparse 0.9 | Sparse 1.0 | Int8 0.5 |
| --- | --- | --- |
| ![](output/report_assets/p001_sparse09.jpg) | ![](output/report_assets/p001_sparse10.jpg) | ![](output/report_assets/p001_int805.jpg) |

### 1.2 平均采样时间

| mode | avg s/image |
| --- | ---: |
| sdpa | 6.877 |
| vanilla | 24.508 |
| triton_fa2 | 7.713 |
| sparse topk=0.3 | 7.060 |
| sparse topk=0.5 | 7.722 |
| sparse topk=0.8 | 8.807 |
| sparse topk=0.9 | 9.223 |
| sparse topk=1.0 | 9.610 |
| sparse_int8 topk=0.3 | 7.086 |
| sparse_int8 topk=0.5 | 7.822 |
| sparse_int8 topk=0.8 | 8.940 |
| sparse_int8 topk=0.9 | 9.261 |
| sparse_int8 topk=1.0 | 9.623 |

### 1.3 结论

- `vanilla` 正确但明显慢于 SDPA。
- `triton_fa2` 与 SDPA 生成效果接近，平均采样时间为 SDPA 的 1.12 倍，满足不超过 2.5 倍的要求。
- `sparse topk=1.0` 平均 9.610 s/image，为 SDPA 的 1.40 倍，精度近似 dense attention。
- `sparse topk=0.5` 平均 7.722 s/image，相比 `topk=1.0` 加速约 19.6%，满足至少 10% 的加速要求。
- `sparse` 和 `sparse_int8` 在 topk 较小时更快，topk 越大越接近 dense 行为。
- `sparse_int8` 五组 topk 均成功生成 20 张图像，无 crash 或 NaN。

## 任务 2: Benchmark

自动阈值检查覆盖 192 条数据，FA2、Sparse topk=1.0、Sparse Int8 topk=1.0 的速度和精度阈值均通过，Sparse/Sparse Int8 topk=0.8 相比 topk=1.0 均有加速。

### 2.1 代表性结果

| B | H | N | D | backend | time ms | vs SDPA | CosSim | RelL1 | RMSE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 8192 | 64 | sdpa | 7.8071 | 1.00x | ref | ref | ref |
| 2 | 16 | 8192 | 64 | triton_fa2 | 7.3631 | 1.06x | 1.000000 | 2.79e-04 | 6.99e-06 |
| 2 | 16 | 8192 | 64 | sparse(topk=0.8) | 7.8238 | 1.00x | 0.897365 | 4.93e-01 | 9.05e-03 |
| 2 | 16 | 8192 | 64 | sparse(topk=1.0) | 9.5902 | 0.81x | 1.000000 | 2.28e-04 | 6.21e-06 |
| 2 | 16 | 8192 | 64 | sparse_int8(topk=0.8) | 7.4216 | 1.05x | 0.897293 | 4.93e-01 | 9.06e-03 |
| 2 | 16 | 8192 | 64 | sparse_int8(topk=1.0) | 8.9155 | 0.88x | 0.999923 | 1.23e-02 | 2.28e-04 |
| 2 | 16 | 16384 | 128 | sdpa | 59.7646 | 1.00x | ref | ref | ref |
| 2 | 16 | 16384 | 128 | triton_fa2 | 60.4197 | 0.99x | 1.000000 | 2.85e-04 | 4.99e-06 |
| 2 | 16 | 16384 | 128 | sparse(topk=0.8) | 104.5481 | 0.57x | 0.896698 | 4.96e-01 | 6.39e-03 |
| 2 | 16 | 16384 | 128 | sparse(topk=1.0) | 130.2458 | 0.46x | 1.000000 | 2.34e-04 | 4.43e-06 |
| 2 | 16 | 16384 | 128 | sparse_int8(topk=0.8) | 88.7326 | 0.67x | 0.896621 | 4.96e-01 | 6.39e-03 |
| 2 | 16 | 16384 | 128 | sparse_int8(topk=1.0) | 110.0046 | 0.54x | 0.999916 | 1.29e-02 | 1.66e-04 |

### 2.2 Int8 长序列

| N | sparse(fp16) | sparse_int8 | speedup |
| ---: | ---: | ---: | ---: |
| 1024 | 0.359 ms | 0.494 ms | 0.73x |
| 2048 | 0.645 ms | 0.772 ms | 0.84x |
| 4096 | 2.153 ms | 2.173 ms | 0.99x |
| 8192 | 7.742 ms | 9.277 ms | 0.83x |
| 16384 | 36.639 ms | 28.358 ms | 1.29x |
| 32768 | 184.285 ms | 143.531 ms | 1.28x |

### 2.3 分析

#### 为什么 Flash Attention 2 更快

Vanilla attention 需要显式构造 `[B,H,N,N]` 的 attention score，并将 softmax 后的概率矩阵写回显存。FA2 将 Q/K/V 按 tile 加载到片上 SRAM，并使用 online softmax 在扫描 K/V block 的同时维护每行最大值和归一化系数，减少了中间矩阵的显存读写。因此 FA2 在长序列上更接近计算受限，而不是显存带宽受限。

#### Block-Sparse 在什么 topk ratio 下达到速度-质量平衡

从 T2I 速度看，`topk=0.3` 最快，但近似最强，图像细节更容易丢失；`topk=0.8/0.9` 质量更接近 `topk=1.0`，但速度收益较小。综合速度和质量，`topk=0.5` 是本次实验中较好的折中点：平均 7.722 s/image，相比 `topk=1.0` 的 9.610 s/image 有明显加速，同时生成图像仍保持 prompt 的主体语义。

#### Int8 短序列慢、长序列快的原因

Int8 attention 每次 forward 都需要额外执行 Q/K 量化、scale 读写和反量化计算。短序列时，attention matmul 本身不够大，量化固定开销占主导，所以 sparse_int8 反而慢。长序列时，矩阵乘和访存占主导，int8 tensor core 吞吐和更低带宽压力开始抵消量化开销。本次数据中 `N=16384` 开始 speedup > 1.0，并达到 1.29x；`N=32768` 为 1.28x。

## 总结

实验完成四种 attention 的实现、T2I 推理验证、benchmark 评测和 int8 长序列加速验证。T2I 共生成 13 组、每组 20 张图像；benchmark 覆盖任务要求的所有配置；int8 长序列在 `N>=16384` 达到超过 1.2x 的加速。实验结果表明，FA2 适合 dense exact attention，block-sparse 可通过 topk 控制速度和质量折中，int8 量化主要适合较长序列场景。
