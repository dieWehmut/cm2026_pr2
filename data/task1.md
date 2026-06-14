# 任务 1：Attention 实现 + Text-to-Image 生成评估

**总分：80 分**

本任务要求在 `attention/` 目录下实现四种 Attention 变体，并将其应用于 PixArt-Alpha Text-to-Image 模型，评估各实现在生成效果和推理速度上的表现。

---

## 准备工作：下载模型权重

在开始实验前，需要下载 PixArt-Alpha 的预训练模型权重并放到 `pretrained_models/` 目录下：

**需要下载的文件**：
1. **PixArt-XL-2-1024-MS.pth** — PixArt-Alpha 主模型权重
2. **sd-vae-ft-ema/** — Stable Diffusion VAE 权重（整个文件夹）

**下载地址**：
> **[南大网盘](https://box.nju.edu.cn/d/a3270e44552b4dc491ac/)**  

**放置位置**：
```
/share/ws/projects/cm2026/
├── pretrained_models/
│   ├── PixArt-XL-2-1024-MS.pth          # 主模型权重
│   └── sd-vae-ft-ema/                   # VAE 权重目录
│       ├── config.json
│       ├── diffusion_pytorch_model.bin
│       └── ...
```

下载完成后即可运行 `test_t2i.py` 进行推理。

---

## 子任务 1.1：Vanilla Attention（10 分）

### 要求

在 `attention/vanilla.py` 中实现纯 PyTorch 版本的 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

**实现约束**：
- 使用 PyTorch 张量运算实现，不得调用 `F.scaled_dot_product_attention` 或其他封装的 Attention 函数。
- 输入输出格式：`(B, num_heads, N, head_dim)`，与 SDPA 保持一致。
- 可以忽略 `attn_mask`、`dropout_p`、`training` 参数（保留接口但不实现）。

**评分标准**（10 分）：
- **生成效果**（10 分）：使用 `test_t2i.py --attention_mode vanilla` 生成图像，与 SDPA baseline 对比，生成结果在视觉上接近（允许细节差异，但主体结构、语义一致）。

**提示**：
- Vanilla Attention 的数值精度应与 SDPA 非常接近，因为都是标准的 softmax attention。
- 速度不做要求（Vanilla 会比 SDPA 慢，这是预期的）。

---

## 子任务 1.2：Triton Flash Attention 2（30 分）

### 要求

在 `attention/fa2.py` 中使用 **Triton** 实现 Flash Attention 2。

**Flash Attention 2 核心思想**：
- 将 Attention 计算分块（tiling），按块加载 Q/K/V 到 SRAM，在线计算 softmax 累加输出。
- 使用 online softmax 技巧（记录累积的 max 和 sum）避免重新扫描整个序列。
- 2D grid `(cdiv(M, BLOCK_M), B * H)`，每个线程块负责一个 Q block 的所有 K blocks。

**实现约束**：
- 必须使用 Triton JIT 编写 kernel，不得调用 `flash-attn`、`xformers` 等封装库。
- 需要支持非 power-of-2 的 `head_dim`（如 PixArt-XL 的 head_dim=72），通过 `BLOCK_K = next_power_of_2(head_dim)` + masking 处理。
- 输入输出格式：`(B, num_heads, N, head_dim)`。

**评分标准**（30 分）：
- **生成效果**（15 分）：使用 `test_t2i.py --attention_mode triton_fa2` 生成图像，与 SDPA baseline 对比，生成结果在视觉上接近。
- **推理速度**（15 分）：在 PixArt-Alpha T2I 推理中，平均每张图的采样时间（`timing.txt` 中记录）**不超过 SDPA 的 2.5 倍**（即速度不低于 SDPA 的 40%）。

**提示**：
- 参考 Flash Attention 2 论文的 Algorithm 1。
- 调试时可以先在小的 (B, H, N, D) 上测试（如 N=256），对比 PyTorch SDPA 的输出验证正确性。

**参考文献**：
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. https://arxiv.org/abs/2307.08691

---

## 子任务 1.3：Block-Sparse Attention（30 分）

### 要求

在 `attention/sparse.py` 中实现基于 block selection 的稀疏 Attention：

1. **Block selection（PyTorch）**：
   - 将 Q, K 沿序列维度划分为 block（默认 block_size=64）。
   - 对每个 block 计算 pooled representation（block 内 mean）。
   - 计算 block-level attention score，每个 Q block 选择 top-k 个 K blocks（`topk_ratio` 控制比例）。
   - 返回选中的 K block 索引（`int32` tensor）。
   
   **Block indices shape 说明**：
   ```python
   # 输入：q, k, v 的 shape 都是 [B, H, N, D]
   # 假设 N=1024, block_size=64, topk_ratio=0.5
   
   num_q_blocks = N // block_size          # 1024 // 64 = 16
   num_k_blocks = N // block_size          # 1024 // 64 = 16
   topk = int(num_k_blocks * topk_ratio)   # 16 * 0.5 = 8
   
   # Block selection 返回的 indices shape:
   # [B, H, num_q_blocks, topk]  = [B, H, 16, 8]
   #  ↑  ↑       ↑          ↑
   #  |  |       |          └─ 每个 Q block 选出的 top-8 个 K block 的索引
   #  |  |       └─ Q block 的数量
   #  |  └─ 每个 head 独立选择
   #  └─ batch 内每个样本独立
   
   # indices[b, h, i, :] 包含第 b 个样本、第 h 个 head、
   # 第 i 个 Q block 选择的 K block 索引（范围 0 到 num_k_blocks-1）
   ```

2. **Sparse Attention kernel（Triton）**：
   - 基于 Flash Attention 2 kernel，但只加载选中的 K/V blocks（根据索引）。
   - 其余逻辑与 FA2 相同（online softmax 累加）。

**实现约束**：
- Block selection 可以用 PyTorch 实现（`torch.topk`），kernel 部分必须用 Triton。
- 输入输出格式：`(B, num_heads, N, head_dim)`。
- `topk_ratio=1.0` 时应该选择所有 blocks，此时行为接近 dense attention。

**评分标准**（30 分）：

### (1) topk=1.0 验证（10 分）

使用 `test_t2i.py --attention_mode sparse --topk_ratio 1.0` 生成图像：
- **生成效果**（5 分）：与 SDPA baseline 接近（允许因 block selection 的舍入差异导致的轻微变化）。
- **推理速度**（5 分）：平均每张图的采样时间不超过 SDPA 的 2.5 倍（速度不低于 40%）。

### (2) topk=0.5 加速验证（10 分）

使用 `test_t2i.py --attention_mode sparse --topk_ratio 0.5` 生成图像：
- **生成效果**（5 分）：能生成与 prompt 语义大致一致的图像（允许细节丢失、风格变化，但不能完全错误或崩溃）。
- **加速效果**（5 分）：相比 `topk=1.0` 的 sparse，采样时间减少至少 10%（即 `time(0.5) ≤ 0.9 * time(1.0)`）。

### (3) 消融实验：topk_ratio vs 效果/速度（10 分）

运行 `topk_ratio ∈ {0.3, 0.5, 0.8, 0.9, 1.0}` 五组实验，报告中需包含：
- **速度曲线**（5 分）：横轴 topk_ratio，纵轴平均采样时间，展示稀疏度如何影响速度。
- **效果对比**（5 分）：选择至少 2 个 prompt，展示不同 topk 下的生成图像，分析稀疏度对图像质量的影响（如：topk=0.3 时哪些细节丢失？topk=0.9 与 1.0 是否视觉上无差异）。

**提示**：
- Sparse Attention 在长序列（N ≥ 8192）上的加速效果更显著；PixArt-Alpha 的序列长度较小，加速相对有限。
- 参考 SpargeAttention 论文的 block selection 策略和实现细节。

**参考文献**：
- Zhang, J., et al. (2024). *SpargeAttention: Sparse Attention for Long Context LLM Inference*. https://arxiv.org/abs/2502.18137 | GitHub: https://github.com/thu-ml/SpargeAttn

---

## 子任务 1.4：Int8 Quantized Sparse Attention（10 分）

### 要求

在 `attention/sparse_int8.py` 中实现对 Q、K 进行 **per-block int8 量化** 的 Sparse Attention：

1. **量化**（Triton kernel）：
   - 对 Q、K 按 block（默认 64×64）进行 symmetric int8 量化：`scale = max(|x|) / 127`。
   - Q 的 scale 额外乘上 `1.44269504 / sqrt(d)`（= 1/ln(2)/sqrt(d)），使 kernel 可以用 `exp2` 替代 `exp`。

2. **Int8 matmul**：
   - 在 Triton kernel 中用 `tl.dot(q_int8, k_int8)` 计算（累积到 int32），再乘 `q_scale * k_scale` 反量化。
   - V 保持 fp16（量化 V 对精度损失较大）。

3. **可选的 smooth_k**：
   - 在量化 K 之前减去 per-channel mean（`K.mean(dim=-2)`），可以改善 int8 动态范围。
   - 数学上这是安全的：per-channel 常数偏移会被 softmax 抵消。

**实现约束**：
- 必须使用 Triton 实现量化 kernel 和 int8 attention kernel。
- Block selection 逻辑与 `sparse.py` 相同（可以复用）。

**评分标准**（10 分）：

运行 `topk_ratio ∈ {0.3, 0.5, 0.8, 0.9, 1.0}` 五组实验：

```bash
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.3
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.5
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.8
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.9
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 1.0
```

- **完整性**（10 分）：成功跑出 5 组结果，生成图像保存到对应目录，无 crash / NaN。不要求生成质量（int8 + sparse 双重近似，质量下降是预期的），只要求能正常运行并输出合理图像（非全黑 / 全白 / 噪声）。

**提示**：
- Int8 量化会引入额外误差（相比 fp16 sparse），在短序列（N ≤ 2048）上可能比 fp16 sparse 更慢（量化开销 > int8 matmul 加速）。
- 任务 2.2 会测试 int8 在长序列（N ≥ 8192）上的加速，这里只要求能跑通。
- 参考 SageAttention 的实现细节（per-block quant、exp2 技巧）。

**参考文献**：
- Zhang, J., et al. (2024). *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration*. https://arxiv.org/abs/2410.02367 | GitHub: https://github.com/thu-ml/SageAttention

---

## 提交清单

| 文件 | 子任务 | 说明 |
| ---- | ---- | ---- |
| `attention/vanilla.py` | 1.1 | Vanilla Attention |
| `attention/fa2.py` | 1.2 | Triton Flash Attention 2 |
| `attention/sparse.py` | 1.3 | Triton Block-Sparse Attention |
| `attention/sparse_int8.py` | 1.4 | Int8 量化 Sparse Attention |

所有子任务的图像对比、速度数据、消融实验结果整理进**实验报告**，无需提交 `output/` 目录下的图像文件。

---

## 调试建议

1. **从小数据开始**：先在 (B=1, H=4, N=256, D=64) 上测试，对比 PyTorch SDPA 输出，确认数值正确性。
2. **检查 NaN**：Triton kernel 中的 softmax 需要正确处理 `-inf` mask 和空选择（会导致除零）。

祝实验顺利！
