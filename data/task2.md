# 任务 2：Attention Benchmark 测评

**总分：20 分**

本任务要求对已实现的 Attention 变体进行系统性的性能测评，量化分析各实现在不同配置下的速度和精度表现。

---

## 子任务 2.1：速度与精度评估（15 分）

### 要求

使用 `benchmark_attention.py` 对 `triton_fa2`、`sparse`、`sparse_int8` 三种实现进行 benchmark，与 SDPA baseline 对比。

**测试配置**（脚本默认值）：
- **序列长度**：N ∈ {2048, 4096, 8192, 16384}
- **Head 数量**：H ∈ {8, 16}
- **Head 维度**：D ∈ {64, 128}
- **数据类型**：fp16
- **Batch size**：2
- **topk_ratio**：{0.3, 0.5, 0.8, 0.9, 1.0}（sparse / sparse_int8 各跑一组）
- **Warmup**：10，**Iterations**：50

**运行命令**：

```bash
# 完整 benchmark（使用脚本默认配置）
python benchmark_attention.py --txt benchmark_results.txt --csv benchmark_results.csv

# 快速测试（可选，用于调试）
python benchmark_attention.py --seq-lens 2048 --num-heads 8 --head-dims 64 --warmup 3 --iters 10
```

**输出指标**：
- **时间**：每个 backend 的前向传播延迟（ms/forward）
- **vs SDPA**：相对 SDPA 的加速比（speedup = time_sdpa / time_backend）
- **CosSim**：Cosine Similarity（越接近 1.0 越好）
- **RelL1**：Relative L1 = Σ|O-O'| / Σ|O|（越小越好）
- **RMSE**：Root Mean Square Error（越小越好）

### 评分标准（15 分）

#### (1) Triton Flash Attention 2 vs SDPA（5 分）

**要求**：
- **速度**（3 分）：在所有 (H, N, D) 配置下，`triton_fa2` 的速度**不低于 SDPA 的 40%**（即 speedup ≥ 0.4，time ≤ 2.5x SDPA）。
- **精度**（2 分）：
  - CosSim > 0.99
  - RelL1 < 1e-3

**示例输出**（RTX A6000 实测，期望通过的 case）：
```
[B=2, H=8, N=8192, D=64]
  backend                  time (ms)   vs SDPA        CosSim         RelL1          RMSE
  sdpa                        2.6768     1.00x         (ref)         (ref)         (ref)
  triton_fa2                  3.4213     0.78x      1.000000      2.75e-04      7.03e-06
```
→ 速度 0.78x（> 0.4 ✓），CosSim 1.000000（> 0.99 ✓），RelL1 2.75e-4（< 1e-3 ✓）

**提示**：
- 如果速度不达标，检查：(1) Triton kernel 是否正确编译并在 GPU 运行；(2) BLOCK_M / BLOCK_N / num_warps 参数是否合理。
- CosSim 和 RelL1 应该非常接近（Flash Attention 是精确算法，只有舍入误差）。

---

#### (2) Sparse Attention (topk=1.0 / 0.8) vs SDPA（5 分）

**要求**：

**topk=1.0（选择所有 blocks）**：
- **速度**（1.5 分）：不低于 SDPA 的 40%。
- **精度**（1.5 分）：
  - CosSim > 0.99
  - RelL1 < 1e-3

**topk=0.8（稀疏 20%）**：
- **速度**（1 分）：相比 topk=1.0 有加速（speedup vs topk=1.0 > 1.0）。
- **精度**（1 分）：
  - CosSim > 0.8
  - RelL1 < 1.0

**示例输出**（RTX A6000 实测）：
```
[B=2, H=8, N=8192, D=64]
  backend                  time (ms)   vs SDPA        CosSim         RelL1          RMSE
  sdpa                        2.6768     1.00x         (ref)         (ref)         (ref)
  sparse(topk=0.8)            2.7829     0.96x      0.899157      4.87e-01      9.07e-03
  sparse(topk=1.0)            3.4392     0.78x      1.000000      2.25e-04      6.25e-06
```
→ topk=1.0 速度 0.78x（> 0.4 ✓），CosSim 1.000000（> 0.99 ✓），RelL1 2.25e-4（< 1e-3 ✓）  
→ topk=0.8 相比 1.0 更快（2.78 vs 3.44，speedup 1.24x > 1.0 ✓），CosSim 0.90（> 0.8 ✓），RelL1 0.49（< 1.0 ✓）

**提示**：
- Sparse Attention 的 block selection 开销在当前实现中较大（PyTorch `topk`），导致 topk=1.0 也比 FA2 慢。
- topk=0.8 vs 1.0 的加速幅度取决于 N：N 越大，稀疏的收益越明显。
- CosSim 和 RelL1 在 topk < 1.0 时会下降，这是稀疏近似的必然代价。

---

#### (3) Sparse Int8 (topk=1.0 / 0.8) vs SDPA（5 分）

**要求**：

与子任务 (2) 相同，但测试的是 `sparse_int8` backend：

**topk=1.0**：
- **速度**（1.5 分）：不低于 SDPA 的 40%。
- **精度**（1.5 分）：
  - CosSim > 0.99（int8 量化会引入额外误差）
  - RelL1 < 2e-2（int8 量化误差比 fp16 高一个数量级）

**topk=0.8**：
- **速度**（1 分）：相比 topk=1.0 有加速。
- **精度**（1 分）：
  - CosSim > 0.8
  - RelL1 < 1.0

**示例输出**（RTX A6000 实测）：
```
[B=2, H=8, N=8192, D=64]
  backend                  time (ms)   vs SDPA        CosSim         RelL1          RMSE
  sdpa                        2.6768     1.00x         (ref)         (ref)         (ref)
  sparse_int8(topk=0.8)       2.2471     1.19x      0.899103      4.87e-01      9.07e-03
  sparse_int8(topk=1.0)       2.7186     0.98x      0.999925      1.21e-02      2.27e-04
```
→ topk=1.0 速度 0.98x（> 0.4 ✓），CosSim 0.999925（> 0.99 ✓），RelL1 1.21e-2（< 2e-2 ✓）  
→ topk=0.8 相比 1.0 更快（2.25 vs 2.72，speedup 1.21x > 1.0 ✓），CosSim 0.90（> 0.8 ✓），RelL1 0.49（< 1.0 ✓）

**注意**：
- Int8 的耗时由「Q/K 量化（与 topk 无关的固定开销）+ block selection + attention kernel（随 topk 变化）」组成。在长序列（如 N=8192）下 attention 计算占主导，因此 sparse_int8 的耗时随 topk 正常变化。
- 但在**短序列**（如 N ≤ 2048）下，固定的量化开销占主导，会出现「不同 topk 下耗时几乎相同」的现象——这是正常的（量化整个 Q/K 的成本与选多少 block 无关），并非 bug。因此本子任务建议在 N ≥ 4096 上评测。
- Int8 的精度要求比 fp16 宽松（topk=1.0 时 CosSim≈0.9999、RelL1≈1.2e-2）。

---

## 子任务 2.2：Int8 长序列加速验证（5 分）

### 要求

使用 `test_sparse_int8.py` 测试 `sparse_int8` 相比 `sparse` (fp16) 在长序列下的加速表现。

**测试配置**（脚本默认值）：
- **序列长度**：N ∈ {1024, 2048, 4096, 8192, 16384, 32768}
- **topk_ratio**：0.8
- **Batch size**：2
- **Head 数量**：16
- **Head 维度**：64
- **Warmup**：10，**Iterations**：30

**运行命令**：

```bash
# 使用脚本默认配置
python test_sparse_int8.py
```

结果会输出到终端并保存到 `output/sparse_int8_results.txt`。可以用 `--txt` 指定其他路径。

**输出指标**：
- 每个 N 下，`sparse` (fp16) 和 `sparse_int8` 的时间（ms）
- Speedup = time_fp16 / time_int8（> 1.0 表示 int8 更快）

### 评分标准（5 分）

**要求**：
- 当 **N ≥ 16384**（16k tokens）时，`sparse_int8` 的速度比 `sparse` (fp16) **快 20% 以上**（即 speedup ≥ 1.20，time_int8 ≤ 0.83 * time_fp16）。
- **提交分析**：在实验报告中分析以下问题：
  1. **为什么短序列时 int8 反而比 fp16 慢？** 从量化开销和计算开销两方面解释。
  2. **为什么长序列时 int8 能加速？** 说明 int8 的加速来源（访存带宽、tensor core 吞吐）。
  3. **从你的实验数据看，int8 加速的"拐点"在哪里？**（即从哪个 N 开始 speedup > 1.0）

**示例输出**（RTX A6000 实测，B=2, H=16, D=64, topk=0.8）：
```
     N |   sparse(fp16) |    sparse_int8 |    speedup |     CosSim |        RelL1 |         RMSE
  1024 |      0.330 ms |      0.433 ms |    0.76x ✗ |   0.872375 | 5.59e-01 | 2.94e-02
  2048 |      0.555 ms |      0.597 ms |    0.93x ✗ |   0.887944 | 5.18e-01 | 1.90e-02
  4096 |      1.492 ms |      1.284 ms |    1.16x ✓ |   0.898414 | 4.89e-01 | 1.28e-02
  8192 |      5.431 ms |      4.387 ms |    1.24x ✓ |   0.896805 | 4.95e-01 | 9.06e-03
 16384 |     22.044 ms |     16.837 ms |    1.31x ✓ |   0.897492 | 4.93e-01 | 6.41e-03
 32768 |     92.716 ms |     68.073 ms |    1.36x ✓ |   0.900066 | 4.86e-01 | 4.51e-03
```
→ 短序列（N ≤ 2048）int8 比 fp16 慢（0.76x / 0.93x），N=4096 起开始反超；
  N=16384 时 speedup 1.31x（> 1.20 ✓），N=32768 时 1.36x（> 1.20 ✓）

**提示**：
- Int8 的加速在长序列上才显现（计算量大，量化开销被摊销）。
- 如果 N=16384 仍未达标，检查：
  1. 量化是否在每次 forward 都重新计算？（应该只量化一次）
  2. Block selection 是否成为瓶颈？（PyTorch `topk` 在长序列上很慢）
- 报告中需附上 **N vs speedup 曲线**，展示 int8 的加速随序列长度增长的趋势。

---

## 提交清单

将以下内容整理到**实验报告**中：

| 内容 | 子任务 | 说明 |
| ---- | ---- | ---- |
| `benchmark_attention.py` 结果表 | 2.1 | 所有 backend 在不同 (H, N, D) 下的速度/精度对比 |
| 速度达标情况说明 | 2.1 | 各 backend 是否满足 "不低于 SDPA 40%" 要求 |
| 精度达标情况说明 | 2.1 | CosSim / RelL1 是否满足阈值 |
| `test_sparse_int8.py` 结果 | 2.2 | N vs speedup 表格或曲线图 |
| Int8 长序列加速分析 | 2.2 | N ≥ 16k 时是否达到 1.2x 加速，原因分析 |

---

## 调试建议

1. **对比 SDPA**：如果某个 backend 的 CosSim < 0.99，用小数据（N=256）逐元素对比输出，定位数值错误。
2. **速度异常慢**：检查 Triton kernel 是否正确编译（看终端是否有 warning），用 `nvidia-smi` 确认 GPU 利用率。

祝实验顺利！
