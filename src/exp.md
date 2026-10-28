## exp1
### 下载
下载代码仓库https://github.com/NVIDIA/kvpress
下载模型Qwen3-8B
下载数据集https://huggingface.co/datasets/caskcsg/LongBench-Pro，你的内部平台没有的话就本地下载托上去，也就500多MB

### 和Agent交互：
* 进入kvpress目录，配置conda环境
* Qwen3-8B开 YaRN（factor=4.0，original_max_position_embeddings=32768）→ 确保支持 128K上下文；temperature=0.6, top_p=0.95
* kvpress中有SnapKV方法，使用Qwen3-8B thinking mode在LongBench-Pro数据集运行该压缩方法，仅在Prefill结束后进行一次压缩，Decode阶段不压缩，新生成的token直接加在压缩的token后边，压缩率设置为10%。也就是说这个数据集有不同输入长度，对于输入长度为8k,16k等，模型每层保留的KV数为0.8k,1.6k等。
* 其他设置和LongBench-Pro保持不变，生成长度为8k，输入的模板，评测方法等。使用8卡推理全部1500条样本。

## exp2
### 下载
下载模型Llama-3.1-8B-Instruct
下载数据集https://huggingface.co/datasets/THUDM/LongBench（使用 16 个任务评测：NrtvQA Qasper MF-en HotpotQA 2WikiMQA Musique GovReport QMSum MultiNews TREC TriviaQA SAMSum PCount PRe Lcc RB-P）具体可以查看SnapKV论文（https://arxiv.org/pdf/2404.14469）

### 和Agent交互：
* 使用Llama-3.1-8B-Instruct对LongBench 16个子集的每个样本做一次prefill，记录最近窗口 W=32 个query对全部key的注意力矩阵 A（layer × query_head × 32 × L）。对所有layer和head取平均，得到一条（32 × L）
* 局部有序性：对每个样本得到的（32 × L），计算窗口内相邻query注意力向量的cosine均值（31对），以及窗口内所有pair的cosine均值（496对）。按层、KV头、样本取平均，输出两个数（0.XX vs 0.YY）
* 结果按模型、按任务分别汇总成表格，并给出全样本均值；保存原始CSV

## exp3
### 下载
下载RPC代码（https://github.com/jiwonsong-dev/ReasoningPathCompression）
下载数据集AIME 2025（30题）https://huggingface.co/datasets/MathArena/aime_2025 与 MATH500 https://huggingface.co/datasets/HuggingFaceH4/MATH-500

### 和Agent交互：
* Qwen3-8B 以 thinking mode 运行（enable_thinking=True）。所有方法统一采样参数：temperature=0.6, top_p=0.95。
* 指标：pass@1。AIME 2025 跑全部30题，每题4次采样；MATH500 每题只采样1次。
* RPC可能不支持Qwen3-8B，因为他transformers版本的问题，你修改其源码让他支持Qwen3-8B
* 对比两组：
  1) Full cache：不压缩
  2) RPC：按官方打分（类似SnapKV，最近窗口注意力，对层和头取平均）
* RPC实现统一的周期性压缩调度（RPC设置：P=1024, R=32, c=4）：
  - 首次压缩：生成满1024个token后进行，从这1024个token中按最近32个token窗口的注意力打分，保留约256个重要token
  - 后续触发：每生成1024个新token，把"此前保留的token"与"新生成的1024个token"合并，联合评估后压回目标规模
* 生成到EOS或最大长度（建议32k），记录实际输出长度
* 输出表格：方法 × 数据集 × 准确率

## exp4
### 和Agent交互：
* Qwen3-8B (128L cache budget)模型（no thinking）跑不同输入长度的上下文（可以通过longbench构建），8k 16k 32k 64k 128k各一条
* 计算两种方法的end-to-end latency以及TTFT：
  - full cache不压缩
  - SnapKV (128L)
