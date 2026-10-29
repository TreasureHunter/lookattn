# LongBench Pro：25 个子集的评测方法

## T1. Retrieval & Ranking（检索与排序）

### T1.1 Global Cohesive Retrieval
- **任务：** 从长文档中检索相关内容，并对检索结果进行组织。
- **上下文依赖：** Full。
- **指标：** NDCG@k。
- **评测方式：** 根据检索结果的相关性及排序计算 NDCG@k，越相关的结果排在越前面，得分越高。

### T1.2 Key-Snippet Retrieval
- **任务：** 在指定段落或局部范围内找到目标片段。
- **上下文依赖：** Partial。
- **指标：** NDCG@k。
- **评测方式：** 根据检索结果的相关性及排序计算 NDCG@k，重点评估局部信息检索的质量。

## T2. Sequencing & Structure Reconstruction（时序与结构重建）

### T2.1 Global Timeline Reconstruction
- **任务：** 将全文中打乱顺序的事件重新排列。
- **上下文依赖：** Full。
- **指标：** Pairwise Accuracy。
- **评测方式：** 比较预测顺序中任意两个事件的相对先后关系与标准顺序是否一致，计算正确事件对的比例。

### T2.2 Local Causal Chain Sorting
- **任务：** 对指定段落中的内容进行因果或逻辑顺序重建。
- **上下文依赖：** Partial。
- **指标：** Pairwise Accuracy。
- **评测方式：** 比较预测的局部排序中，各内容对的相对顺序是否正确。

## T3. Evidence-Grounded QA（证据支持的问答）

### T3.1 Multi-Doc Integration QA
- **任务：** 整合来自多个文档或不同位置的证据，通过多跳推理回答问题。
- **上下文依赖：** Full。
- **指标：** Accuracy。
- **评测方式：** 判断模型最终答案是否正确，计算正确答案占总样本的比例。

### T3.2 Single-Hop Fact QA
- **任务：** 根据局部段落中的信息回答事实性问题。
- **上下文依赖：** Partial。
- **指标：** Accuracy。
- **评测方式：** 判断模型最终答案是否正确，计算正确答案占总样本的比例。

## T4. Summarization & Synthesis（摘要与综合）

### T4.1 Global-Coverage Constrained Summary
- **任务：** 在给定约束下概括全文，要求覆盖整体内容。
- **上下文依赖：** Full。
- **指标：** SemSim 和 ROUGE-L。
- **评测方式：** 每个样本有 3 个参考摘要，分别计算生成摘要与各参考摘要的 SemSim 和 ROUGE-L，每个指标取 3 个参考结果中的最大值，再进行等权组合：

  \[
  S=0.5\max_i \operatorname{SemSim}_i
  +0.5\max_i \operatorname{ROUGE\text{-}L}_i
  \]

### T4.2 Query-Focused Summary
- **任务：** 围绕指定主题或问题，生成针对性的摘要。
- **上下文依赖：** Partial。
- **指标：** SemSim 和 ROUGE-L。
- **评测方式：** 与 T4.1 使用相同的计算方法，但重点评估摘要是否覆盖指定主题。

## T5. Attribution & Citation Alignment（归因与引用对齐）

### T5.1 Full-Sentence Citation Alignment
- **任务：** 为生成的所有句子匹配正确的来源或证据。
- **上下文依赖：** Full。
- **指标：** F1。
- **评测方式：** 将预测的来源归属与标准来源归属比较，计算 Precision、Recall 和 F1。

### T5.2 Key-Statement Citation Alignment
- **任务：** 为指定的关键陈述匹配来源。
- **上下文依赖：** Partial。
- **指标：** F1。
- **评测方式：** 比较关键陈述的预测来源与标准来源，计算 Precision、Recall 和 F1。

## T6. Aggregation & Clustering（聚合与聚类）

### T6.1 Large-Scale Document Clustering
- **任务：** 对文档内容进行分类，并输出各类别所占比例。
- **上下文依赖：** Full。
- **指标：** SubEM。
- **评测方式：** 使用 SubEM 评估模型输出与规范答案的匹配程度。论文未在正文中给出该子集的完整字段级匹配规则。

### T6.2 Targeted Subset Cluster Identification
- **任务：** 根据查询，找出属于指定类别的内容实例。
- **上下文依赖：** Partial。
- **指标：** F1。
- **评测方式：** 比较预测实例集合与标准实例集合，综合考虑误报和漏报。

### T6.3 Global Frequency Analysis
- **任务：** 统计全文中的词频，并按照频率排序。
- **上下文依赖：** Full。
- **指标：** Pairwise Accuracy。
- **评测方式：** 比较预测排序中任意两个词的相对顺序与标准词频排序是否一致。

## T7. Consistency & Compliance Checking（一致性与合规检查）

### T7.1 Global Conflict & Inconsistency Localization
- **任务：** 找出全文中相互矛盾或不一致的内容，并定位相关片段。
- **上下文依赖：** Full。
- **指标：** F1。
- **评测方式：** 比较模型定位的冲突片段与标准标注，计算 Precision、Recall 和 F1。

### T7.2 Targeted Rule or Condition Violation Detection
- **任务：** 根据给定规则或条件，找出违反规则的内容。
- **上下文依赖：** Partial。
- **指标：** F1。
- **评测方式：** 比较模型识别的违规内容与标准违规内容，综合衡量误报与漏报。

### T7.3 Comprehensive Error & Anomaly Sweep
- **任务：** 扫描全文，找出拼写错误。
- **上下文依赖：** Full。
- **指标：** F1。
- **评测方式：** 将模型标记的错误与标准错误标注进行比较，计算 Precision、Recall 和 F1。

## T8. Structured & Numeric Reasoning（结构化与数值推理）

### T8.1 Structured Multi-Source Consistency Verification
- **任务：** 综合多个来源中的结构化信息，完成数值计算或一致性验证。
- **上下文依赖：** Full。
- **指标：** SubEM。
- **评测方式：** 将模型计算出的规范答案与标准答案进行匹配。

### T8.2 Single-Source Targeted Aggregation
- **任务：** 根据单个来源中的信息，完成指定的聚合或数值计算。
- **上下文依赖：** Partial。
- **指标：** SubEM。
- **评测方式：** 比较模型输出的计算结果与标准答案是否匹配。

### T8.3 Long-Context Procedural State Tracking
- **任务：** 根据长文本中的操作或事件，追踪实体状态随时间或步骤发生的变化。
- **上下文依赖：** Full。
- **指标：** 论文存在不一致：第 2 节表 2 将指标列为 SubEM，附录 A 则列为 F1。
- **评测方式：** 需核对作者评测代码才能确定实际采用的指标及具体计算方式。

## T9. Version & Code Diff Analysis（版本与代码差异分析）

### T9.1 Dependency-Aware Multi-Version Impact Analysis
- **任务：** 比较多个版本之间的变化，并分析这些变化对依赖关系或相关组件的影响。
- **上下文依赖：** Full。
- **指标：** F1。
- **评测方式：** 将模型识别出的变化项或影响项与标准标注比较，计算 Precision、Recall 和 F1。

### T9.2 Localized Interface Change Detection
- **任务：** 检测不同版本之间局部接口的变化。
- **上下文依赖：** Partial。
- **指标：** F1。
- **评测方式：** 比较模型识别的接口变化与标准变化集合，计算 Precision、Recall 和 F1。

## T10. Rule Induction & In-Context Learning（规则归纳与上下文学习）

### T10.1 Large-Scale In-Context Rule Induction
- **任务：** 从全文或大量示例中归纳规则，并据此做出判断。
- **上下文依赖：** Full。
- **指标：** SubEM。
- **评测方式：** 将模型归纳出的规则或判断结果与规范答案匹配。

### T10.2 Targeted Example-Based Rule Induction
- **任务：** 从指定示例中归纳规则，并将规则用于相关判断。
- **上下文依赖：** Partial。
- **指标：** SubEM。
- **评测方式：** 比较模型输出与标准答案是否匹配，重点在于从指定示例中归纳规则。

## T11. Dialogue Memory & Long-Horizon Tracking（对话记忆与长期跟踪）

### T11.1 Long-Range Entity & Commitment Tracking
- **任务：** 跨越较长的对话历史，跟踪实体状态、用户承诺或此前的决定。
- **上下文依赖：** Full。
- **指标：** Accuracy。
- **评测方式：** 根据对话历史回答问题，计算正确答案占总样本的比例。

### T11.2 Short-Range Reference Resolution & State Query
- **任务：** 利用局部对话上下文，解析指代对象并查询相关状态。
- **上下文依赖：** Partial。
- **指标：** Accuracy。
- **评测方式：** 判断模型是否正确识别指代对象或局部状态，计算正确答案比例。

## 统一的指标计算与实验设置

### 指标定义

| 指标 | 计算方式 | 衡量目标 |
|---|---|---|
| NDCG@k | DCG@k / IDCG@k | 检索结果的相关性及排序质量 |
| Pairwise Accuracy | 正确的成对顺序数 / 可比较的成对顺序总数 | 预测排序与标准排序的一致程度 |
| Accuracy | 正确答案数 / 样本总数 | 最终答案的正确率 |
| Precision | TP / (TP + FP) | 预测结果中正确项的比例 |
| Recall | TP / (TP + FN) | 标准答案中被识别项的比例 |
| F1 | 2 × Precision × Recall / (Precision + Recall) | Precision 与 Recall 的综合表现 |
| SubEM | 论文采用的子项匹配指标 | 规范答案的匹配程度；正文未给出完整公式 |
| SemSim | 语义相似度计算 | 生成内容与参考答案的语义一致程度 |
| ROUGE-L | 基于最长公共子序列计算 | 生成文本与参考文本的序列重合程度 |

其中，TP、FP、FN 分别表示正确预测、错误预测和漏掉的标准答案项。论文未在正文中完整说明 NDCG@k 的具体 k、相关性等级赋值、SubEM 的详细公式，以及所有任务的答案解析规则。

### 统一汇总方式

1. 对每个样本计算相应的任务指标。
2. 对所有样本的得分取平均。
3. 将平均分乘以 100，作为报告分数。
4. 摘要任务使用 SemSim 与 ROUGE-L 的等权组合。

### 推理设置

- 每个模型按默认推理参数运行 3 次。
- 常规表现报告多次生成的平均得分。
- Best-of-N 报告多次生成中最高的得分。
- Pass@N 衡量至少有一次回答完全正确的概率。
- 非 Thinking 模式的输出长度统一设为 1K tokens；Thinking 模式根据模型支持的上下文长度设置为 8K 或 32K。
- 当输入超过模型上下文限制时，从输入中间截断。

## 原文

- [LongBench Pro — Section 5 Evaluation](https://arxiv.org/html/2601.02872#S5)
- [Appendix A: Task Definitions](https://arxiv.org/html/2601.02872#A)

**注意：** T8.3 的指标在正文表格和附录中存在不一致。若需严格复现论文结果，应以作者公开的评测代码为准。
