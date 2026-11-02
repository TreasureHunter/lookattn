# 动态视觉 Token 压缩方案笔记

## 1. 任务目标

输入图像经过 Vision Encoder 得到视觉特征：

\\[ X\in\mathbb R^{N\times D} \\]

例如：

\\[ X\in\mathbb R^{576\times1024} \\]

目标是将其压缩为：

\\[ X'\in\mathbb R^{N_1\times D} \\]

其中 \\(N_1\\) 可以变化，例如 32、64、128、300。

核心思路： 不直接从原始视觉 Token 中选择 Top-K，而是学习一个动态投影矩阵 \\(P\\)，将原始 Token 加权聚合为新的视觉 Token。

## 2. 模型结构

### Step 1：提取全局视觉信息

初始化 16 个可学习的 Latent Queries：

\\[ U\in\mathbb R^{16\times256} \\]

使用 Multi-Head Cross-Attention，从原始视觉特征 \\(X\\) 中提取信息。

\\[ Q_u=UW_Q \\]

\\[ K_x=XW_K,\qquad V_x=XW_V \\]

\\[ Z=\operatorname{Softmax} \left(\frac{Q_uK_x^\top}{\sqrt{256}}\right)V_x \\]

得到：

\\[ Z\in\mathbb R^{16\times256} \\]

目的： 将原始视觉特征提炼为 16 个全局视觉表示，为后续生成压缩 Token 提供信息。

### Step 2：构造输出 Query

根据目标 Token 数量 \\(N_1\\)，生成对应数量的位置编码：

\\[ R=\operatorname{PE}(N_1) \in\mathbb R^{N_1\times256} \\]

使用 Cross-Attention，让每个输出位置从 16 个全局视觉表示中读取信息：

\\[ C=\operatorname{CrossAttn}(R,Z,Z) \\]

得到：

\\[ C\in\mathbb R^{N_1\times256} \\]

融合位置编码和视觉信息：

\\[ Q=\operatorname{MLP}(R+C) \\]

得到：

\\[ Q\in\mathbb R^{N_1\times256} \\]

目的： 为每个输出 Token 生成与图像内容相关的 Query。输出数量由 \\(N_1\\) 决定。

### Step 3：计算动态投影矩阵

将原始视觉特征映射为 Key：

\\[ K=XW_K^{p} \\]

其中：

\\[ W_K^{p}\in\mathbb R^{1024\times256} \\]

因此：

\\[ K\in\mathbb R^{576\times256} \\]

计算输出 Query 与原始视觉 Key 的相似度：

\\[ P=\operatorname{Softmax} \left(\frac{QK^\top}{\sqrt{256}}\right) \\]

得到：

\\[ P\in\mathbb R^{N_1\times576} \\]

Softmax 沿原始视觉 Token 维度进行，每一行权重之和为 1。

目的： 决定每个输出 Token 应该从哪些原始视觉 Token 中聚合信息。

### Step 4：生成压缩视觉特征

通过投影矩阵加权聚合原始视觉特征：

\\[ \boxed{X'=PX} \\]

得到：

\\[ X'\in\mathbb R^{N_1\times1024} \\]

目的： 生成新的视觉 Token，而不是简单保留原始 Token 的子集。

## 3. 整体流程

Vision Encoder 输出

\\(X:576\times1024\\)

Step 1：Latent Cross-Attention

16 个 Latent Queries 提取全局信息

\\(Z:16\times256\\)

Step 2：生成输出 Query

位置编码 + 全局视觉信息

\\(Q:N_1\times256\\)

Step 3：计算动态投影矩阵

\\(P:N_1\times576\\)

Step 4：加权聚合

\\(X'=PX:N_1\times1024\\)

MLLM 输入

压缩视觉特征 + 文本问题

## 4. 训练方案

训练数据： 单轮 VQA 数据，每条样本包含图像、一个问题和对应答案。

1. 冻结 Vision Encoder，提取视觉特征 \\(X\\)。
2. 随机采样目标 Token 数量 \\(N_1\\)。
3. 通过上述压缩模块生成 \\(X'\\)。
4. 将 \\(X'\\) 和文本问题输入冻结的 MLLM。
5. 使用答案生成交叉熵损失训练压缩模块：

\\[ \mathcal L\_{\mathrm{VQA}} =-\sum_t\log p(y_t\mid X',Q,y\_{\<t}) \\]

训练参数：

- 冻结：Vision Encoder、MLLM。
- 训练：Latent Queries、Cross-Attention、MLP、Key Projection 等压缩模块参数。
- 第一版只使用 VQA Loss，不额外引入辅助损失。

## 5. 多轮推理方式

训练使用单轮 VQA 数据；多轮能力在推理阶段验证。

对于同一张图像：

1. Vision Encoder 提取 \\(X\\)。
2. 压缩模块生成一次 \\(X'\\)。
3. 后续多个问题共享同一份 \\(X'\\)。

\\[ Y_i=\operatorname{MLLM}(X',Q_i) \\]

不需要针对每个新问题重新压缩图像。

## 6. 第一版实现配置

| 模块                     | 初始配置                         |
| ---------------------- | ---------------------------- |
| 原始视觉特征                 | \\(576\times1024\\)          |
| Latent Queries 数量      | 16                           |
| 隐藏维度                   | 256                          |
| Latent Cross-Attention | 1 层 Multi-Head               |
| 输出 Query               | 位置编码 + Cross-Attention + MLP |
| 投影矩阵                   | Query-Key 点积 + Softmax       |
| 特征聚合                   | \\(X'=PX\\)                  |
| 训练损失                   | 单轮 VQA Loss                  |
| 训练预算                   | 随机采样多个 \\(N_1\\)             |

实现时需要注意： Step 1 的 Latent Cross-Attention 和 Step 3 的 Query-Key 投影是两个不同的操作，建议使用不同的参数矩阵，避免代码中混淆。第一版暂不加入预算编码、Latent Self-Attention 和额外辅助损失。

这份笔记可以直接作为 Codex 的初始实现说明。当前设计属于待验证方案，尤其需要通过消融实验检查 16 个 Latent 是否足以提供有效的全局信息，以及不同压缩预算下的性能表现。
