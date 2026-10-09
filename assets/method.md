## Token-Sparse GRPO 具体方案

1. 采样与奖励计算：每个 prompt 采样 GGG 个回答，根据每个回答的最终 reward 计算标准 GRPO advantage。同一回答的所有 token 共用一个 advantage。

2. Token 重要性评估：利用当前训练 step 已有的前向计算信息，为每个 token 计算重要性分数，不额外计算逐 token 的完整参数梯度。

3. 筛选关键 Token：按照重要性分数排序，在每个回答的有效 token 中选择 Top-kkk，例如 10%、25% 或 50%。

4. 构造稀疏策略损失：仅对选中的 token 计算策略损失，未选中的 token 不直接贡献策略损失，但仍保留在原始序列中作为上下文。

Lsparse=−∑t=1TmtAlog⁡πθ(yt∣x,y<t)\mathcal L_{\text{sparse}} =-\sum_{t=1}^{T}m_t A \log\pi_\theta(y_t\mid x,y_{<t})Lsparse=−t=1∑TmtAlogπθ(yt∣x,y<t)

其中，AAA 是该回答共享的 GRPO advantage，mt∈{0,1}m_t\in\{0,1\}mt∈{0,1} 表示 token 是否被选中。

5. 反向传播与参数更新：使用稀疏策略损失执行标准反向传播和优化器更新。

6. 实验设置：对比标准 GRPO、随机 Token 筛选和重要性分数筛选，测试不同保留比例下的任务准确率、每步训练时间及总训练耗时。
