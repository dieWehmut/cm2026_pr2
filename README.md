# 实验 2：CM2026 Project 2

本实验的任务是实现并优化多种 Attention 机制，并将其应用于 Text-to-Image 模型 PixArt-Alpha，在此过程中你将实现不同的 Attention 变体（Vanilla Attention、Flash Attention 2、Block-Sparse Attention、Int8 量化 Attention），并通过实验分析不同设计对生成效果和推理速度的影响。

本仓库提供了可运行的基线（使用 PyTorch SDPA），以及 Attention 接口和测试脚本，供任务 1 和任务 2 使用。

**硬件要求**：本项目已经过显存优化（VAE 和 DiT 模型动态 offloading），推荐使用 **NVIDIA RTX 3070 (8GB)** 或更高显存的显卡。峰值显存约 7GB，可在 8GB 显存的 GPU 上运行。

---

## 示例生成结果

<p align="center">
  <img src="data/sample0.jpg" width="240" alt="Sample 0">
  <img src="data/sample1.jpg" width="240" alt="Sample 1">
  <img src="data/sample2.jpg" width="240" alt="Sample 2">
  <br>
  <em>PixArt-Alpha 使用不同 prompts 的生成示例</em>
</p>

**注意**：本项目直接使用预先提取好的 T5 embeddings（保存在 `data/prompt_embeddings/`），无需在线调用 T5 encoder。PixArt-Alpha 原本使用 T5-XXL encoder（参数量较大，需要 >20GB 显存），为了方便实验，我们已为 20 个测试 prompts 预先提取了 embeddings（`prompt_000.pt` 到 `prompt_019.pt`），可以在显存较小的 GPU（如 RTX 3070）上运行。

---

## 一、快速开始

### 1. 安装依赖

详细的环境安装说明请参考 [data/env_install.md](data/env_install.md)。

### 2. 运行默认推理（使用 SDPA baseline）

```bash
python test_t2i.py
```

默认使用 `--attention_mode sdpa`（PyTorch Scaled Dot-Product Attention）。运行结束后，生成的图像和计时结果会保存到 `output/<date>_attention_sdpa/` 目录下。

### 3. 切换 Attention 实现

```bash
# 使用 Vanilla Attention
python test_t2i.py --attention_mode vanilla

# 使用 Triton Flash Attention 2
python test_t2i.py --attention_mode triton_fa2

# 使用 Block-Sparse Attention (topk=0.5)
python test_t2i.py --attention_mode sparse --topk_ratio 0.5

# 使用 Int8 量化 Sparse Attention
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.5
```

生成的图像将保存到 `output/<date>_attention_<mode>_topk<ratio>/` 目录下，同时输出目录中会包含 `timing.txt` 记录每张图的采样时间。

---

## 二、实验要求

### 任务概览

本实验包含两个任务：

- **任务 1**（Attention 实现 + T2I 生成评估，80 分）：在 `attention/` 目录下实现四种 Attention，并评估其在 PixArt-Alpha 上的生成效果和速度。详见 [data/task1.md](data/task1.md)。
- **任务 2**（Attention Benchmark 测评，20 分）：使用 `benchmark_attention.py` 和 `test_sparse_int8.py` 系统性评估各 Attention 实现的速度和精度。详见 [data/task2.md](data/task2.md)。

### 实现约束

- **Triton kernel 实现**：Flash Attention 2、Block-Sparse Attention、Int8 Attention 需要使用 Triton 编写 kernel，不得直接调用第三方封装库（如 `flash-attn`、`xformers` 等）。
- **可以使用**：PyTorch 张量运算、Triton JIT、CUDA 基础库。
- **Vanilla Attention** 可以使用纯 PyTorch 实现，不得调用 `F.scaled_dot_product_attention`。

### 建议完成顺序

1. 运行默认 SDPA baseline，确认 T2I 生成流程正常。
2. 按顺序实现 `attention/vanilla.py`、`attention/fa2.py`、`attention/sparse.py`、`attention/sparse_int8.py`。
3. 使用 `test_t2i.py` 验证各 Attention 在 T2I 上的生成效果。
4. 使用 `benchmark_attention.py` 和 `test_sparse_int8.py` 完成任务 2 的性能测评。
5. 整理实验数据并撰写报告。

### 评分构成

| 部分 | 分值 |
| ---- | ---- |
| **任务 1：Attention 实现 + T2I 生成** | **80** |
| &emsp;子任务 1.1：Vanilla Attention | 10 |
| &emsp;子任务 1.2：Triton Flash Attention 2 | 30 |
| &emsp;&emsp;生成效果与 SDPA 接近 | 15 |
| &emsp;&emsp;速度不低于 SDPA 的 40% | 15 |
| &emsp;子任务 1.3：Block-Sparse Attention | 30 |
| &emsp;&emsp;topk=1.0 效果/速度验证 | 10 |
| &emsp;&emsp;topk=0.5 加速验证 | 10 |
| &emsp;&emsp;消融实验：topk vs 效果/速度 | 10 |
| &emsp;子任务 1.4：Int8 Quantized Sparse Attention | 10 |
| **任务 2：Attention Benchmark 测评** | **20** |
| &emsp;子任务 2.1：速度与精度评估 | 15 |
| &emsp;&emsp;Triton FA2 vs SDPA | 5 |
| &emsp;&emsp;Sparse (topk=1.0 / 0.8) vs SDPA | 5 |
| &emsp;&emsp;Sparse Int8 (topk=1.0 / 0.8) vs SDPA | 5 |
| &emsp;子任务 2.2：Int8 长序列加速验证 | 5 |
| **合计** | **100** |

---

## 三、提交要求

### 1. 实验报告

实验报告请以 `pdf` 格式提交。建议至少包含以下内容：

- **实验设置**：简要说明你实现了哪些 Attention，使用的 Triton 版本、CUDA 版本、GPU 型号。
- **任务 1 结果**：
  - 每个子任务的 T2I 生成结果（图像质量对比）。
  - 每张图的采样时间（从 `timing.txt` 汇总）。
  - 消融实验：不同 `topk_ratio` 下的速度 vs 生成效果折线图。
- **任务 2 结果**：
  - `benchmark_attention.py` 的速度、CosSim、RelL1、RMSE 对比表。
  - `test_sparse_int8.py` 的长序列加速曲线（N vs speedup）。
- **结果分析**：
  - 为什么 Flash Attention 2 相比 Vanilla Attention 更快？
  - Block-Sparse 在什么 topk ratio 下达到速度-质量平衡？
  - Int8 量化在短序列慢、长序列快的原因是什么？
- **总结**：概括哪些设计有效，哪些设计的适用场景，以及对 Attention 优化的主要观察。

### 2. 代码

代码部分应保证可直接运行。提交文件说明如下：

| 文件 | 是否必交 | 说明 |
| ---- | ---- | ---- |
| `attention/vanilla.py` | **必交** | 任务 1.1：纯 PyTorch Attention |
| `attention/fa2.py` | **必交** | 任务 1.2：Triton Flash Attention 2 |
| `attention/sparse.py` | **必交** | 任务 1.3：Triton Block-Sparse Attention |
| `attention/sparse_int8.py` | **必交** | 任务 1.4：Int8 量化 Sparse Attention |

即使某部分实验没有完全做完，上述必交文件仍需全部提交（可保留未实现的 stub），以便统一验收。不需要提交 `output/` 下的结果文件，图表和数值直接整理进报告即可。

### 3. 压缩包组织示例

```text
姓名_学号_Project2.zip
|
|-- report.pdf
|
|-- code/
|   |-- attention/
|   |   |-- vanilla.py
|   |   |-- fa2.py
|   |   |-- sparse.py
|   |   |-- sparse_int8.py
|   |   |-- __init__.py
|   |   |-- sdpa.py  (可选，如有修改可提交)
```

---

## 四、文件说明

| 文件 / 目录 | 说明 |
| ---- | ---- |
| [test_t2i.py](test_t2i.py) | T2I 推理主脚本，`--attention_mode` 选择 Attention 实现 |
| [benchmark_attention.py](benchmark_attention.py) | 任务 2.1：Attention 速度/精度 benchmark |
| [test_sparse_int8.py](test_sparse_int8.py) | 任务 2.2：Int8 vs FP16 长序列加速测试 |
| [attention/](attention/) | **作业**：实现 Attention 变体 |
| &emsp;[attention/vanilla.py](attention/vanilla.py) | 子任务 1.1：纯 PyTorch Attention |
| &emsp;[attention/fa2.py](attention/fa2.py) | 子任务 1.2：Triton Flash Attention 2 |
| &emsp;[attention/sparse.py](attention/sparse.py) | 子任务 1.3：Triton Block-Sparse Attention |
| &emsp;[attention/sparse_int8.py](attention/sparse_int8.py) | 子任务 1.4：Int8 量化 Sparse Attention |
| &emsp;[attention/sdpa.py](attention/sdpa.py) | SDPA baseline（已提供） |
| [PixArt-alpha/](PixArt-alpha/) | PixArt-Alpha 模型代码（已提供） |
| [data/task1.md](data/task1.md) | 任务 1 详细说明 |
| [data/task2.md](data/task2.md) | 任务 2 详细说明 |
| [data/prompt_embeddings/](data/prompt_embeddings/) | 预提取的 T5 embeddings（20 个 prompts） |
| [requirements.txt](requirements.txt) | Python 依赖列表 |

---

## 五、快速测试

### 任务 1：T2I 生成测试

```bash
# 测试各 Attention 的生成效果
python test_t2i.py --attention_mode sdpa
python test_t2i.py --attention_mode vanilla
python test_t2i.py --attention_mode triton_fa2
python test_t2i.py --attention_mode sparse --topk_ratio 0.5
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.5
```

查看 `output/<date>_attention_<mode>/timing.txt` 获取每张图的采样时间。

### 任务 2.1：Attention Benchmark

```bash
# 默认配置：N ∈ {512, 1024, 2048, 4096}, H ∈ {8, 16}, D ∈ {32, 64, 128}
python benchmark_attention.py --txt benchmark_results.txt --csv benchmark_results.csv

# 快速测试
python benchmark_attention.py --seq-lens 1024 --num-heads 8 --head-dims 64 --warmup 3 --iters 10
```

### 任务 2.2：Int8 长序列测试

```bash
# 测试 N ∈ {1024, 2048, 4096, 8192, 16384, 32768}
python test_sparse_int8.py --seq-lens 1024 2048 4096 8192 16384 32768
```

---

## 补充说明

- **速度测量**：所有速度测量均在**相同 GPU、相同 prompt、相同随机种子**下进行，以保证可比性。
- **Triton 版本**：本仓库使用 Triton 3.4.0，CUDA 12.6。如使用其他版本请在报告中说明。
- **硬件环境**：建议在 **NVIDIA RTX 3070** 或同等算力 GPU 上测试。报告中需注明实际使用的 GPU 型号。
- **预提取 embeddings**：本项目使用 `data/prompt_embeddings/` 中预先提取好的 T5 embeddings（20 个 prompts），无需在线调用 T5 encoder，可在 8GB 显存的 GPU 上运行。
- **助教将会抽查提交的代码**。对于与报告内容严重不符、抄袭、违反诚信的内容予以严肃处理。

---

## 参考文献

### Attention 优化

- **Flash Attention**: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. https://arxiv.org/abs/2205.14135
- **Flash Attention 2**: Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024. https://arxiv.org/abs/2307.08691
- **SpargeAttention**: Zhang, J., et al. (2025). *SpargeAttention: Sparse Attention for Long Context LLM Inference*. https://arxiv.org/abs/2502.18137 | GitHub: https://github.com/thu-ml/SpargeAttn
- **SageAttention (Int8)**: Zhang, J., et al. (2024). *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration*. https://arxiv.org/abs/2410.02367 | GitHub: https://github.com/thu-ml/SageAttention

### Text-to-Image 模型

- **PixArt-α**: Chen, J., et al. (2023). *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*. ICLR 2024. https://arxiv.org/abs/2310.00426 | GitHub: https://github.com/PixArt-alpha/PixArt-alpha

### Triton 编程

- **Triton Language**: Tillet, P., Kung, H.-T., & Cox, D. (2019). *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. MAPL@PLDI 2019. https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- **Triton Documentation**: https://triton-lang.org/

---

## 学术诚信

请独立完成本次作业。

- 可以阅读课程提供的代码、文档与参考资料（包括上述参考文献的论文和开源实现）。
- 可以与同学讨论思路，但不得直接交换代码、实验结果或报告文本。
- 不得抄袭他人实现后冒充为自己的工作。参考开源实现时需在报告中明确说明。
- 提交的代码、结果和分析必须与本人实际实现一致。如出现报告数值与本人实现不一致的情况，将严肃处理。

祝实验顺利！
