# lookattn

在 [HAWK](https://arxiv.org/abs/2508.15882) 的 head-importance 视觉 token 剪枝之上，训练一组**可学习的前瞻行（lookahead rows）**，用它们代替指令文本作为打分查询。

HAWK 用「问题/指令」的隐状态去问「哪些视觉 token 重要」，所以保留下来的往往是**问题里提到的东西**。lookattn 让那组可学习的行去扮演**答案**，于是保留下来的变成**回答时会去看的东西**。只有这组行是学的（`32 × 3584 ≈ 115k` 个参数，占 7B 模型的 0.0016%），主干模型全程冻结。

本包是从研究仓库里抽出来的**可运行、可训练、可推理**版本：不含单元测试、不含论文里的复现实验脚本、不含数据集。

---

## 1. 方法

### 1.1 打分规则

学生（32 个伪 token）和教师（真实答案 token）**共用同一次读数**，取自解码器第 `L` 层（默认 `L = 1`）：

```text
H   = forward(prompt_embeds ‖ appended_embeds, mRoPE position_ids)  # 真实前向
H_L = 第 L 层的输入                                                  # hidden_states[L]
N   = layer_L.input_layernorm(H_L)
Q   = layer_L.self_attn.q_proj(N)      # 只取被追加的那些行
K   = layer_L.self_attn.k_proj(N)      # 只取视觉 token 的列
K   = repeat_kv(K, num_key_value_groups)   # GQA：kv 头展开到 query 头
S   = softmax( Q Kᵀ / sqrt(d_head) )   # [H, rows, V] —— 不加 RoPE，softmax 只在视觉列上
```

三个刻意的设计选择：

- **Pre-RoPE。** 前向本身跑的是真实 mRoPE，只有**打分那两个投影**跳过 RoPE。所以 `H_L` 是货真价实的隐状态，不是被 ablate 过的模型的产物。去掉 RoPE 之后打分是纯内容匹配，与 token 在哪儿无关；否则分数会依赖追加行到每个视觉 token 的距离，而这个距离随视觉 token 数 `V`（评测时 1280~16384）变化，Qwen 的 `(t, h, w)` 位置编码还会让分数带上位置结构——在某个 `V` 上拟合的打分器换个 `V` 就不是同一个东西了。
- **softmax 只在视觉列上归一化**，不是在所有 key 上。正因如此输出才是「对视觉 token 的一个分布」，KL 才有定义。因果性不需要 mask：所有视觉 token 都在追加行之前。
- **`output_attentions=False`**，否则会从 SDPA 掉回 eager，物化 `[H, T+P, T+P]` 的注意力矩阵（长 prompt 下是几个 GB）。

### 1.2 从读数里取出分数

`S` 是 `[H, rows, V]`，转成可排序的 `[V]` 向量永远走同一条链：

```text
per-head = renormalise(mean over rows of S^(1/T))     # T = temperature
score    = merge over heads of per-head               # head_merge
```

`temperature` 和 `head_merge` 都写进 checkpoint，因为**两个都会改变排序**——训练用 0.5、推理用 1.0 会静默地排出不同的结果，不会报错。`head_merge` 默认 `mean`；`sum` 只差一个常数 `H`，选出的 token 完全一样。

### 1.3 损失

教师行就是**真实答案 token**，走完全相同的读数规则。行数 `A` 不必等于伪 token 数 `P`——KL 之前 query 维已经被平均掉了。

```text
每个样本:
    loss = Σ_h KL( p_t[h] ‖ p_s[h] )
    p_s  = renormalise(mean over rows of student)   # [H, V]，有梯度
    p_t  = renormalise(mean over rows of teacher)   # [H, V]，detach
```

**注意损失是 28 个头的求和**（`per_head.sum()`），不是头平均。这一点在读数时很容易搞混：训练日志里的 `teacher_kl_uniform` 是**头平均**，两者单位不同，**不能相除**。

平均是在**概率空间**上做的（LookaheadKV 的顺序），之后重新归一化。如果改成对 log 概率取平均，得到的张量根本不是分布，`kl_div` 会静默返回一个不是 KL 的数。

主干被裁到 `score_layer + 1 = 2` 层且冻结，只有 `embed_pseudo` 有梯度。样本是**逐个**累积的（打分读数是 batch-1，因为各样本视觉 token 数不同），最后优化器只走一步——所以 `--batch-size` 花的是时间而不是显存。

### 1.4 推理

推理用的读数和合并方式与训练**完全相同**，所以离线测的 `recall@k` 就是部署时真正做出的选择。

```text
install:  载入 scorer → 存到内层 W 上 → 重绑 hawk.pruning.prune_qwen_prefill

prefill:
    visual ← input_ids == image_token_id | video_token_id   (2D mask 时相与)
    没有 scorer / 没有视觉 token / keep_ratio ≥ 1 → 原样回退到 HAWK
    text.layers 临时换成前 score_layer + 1 层          # try/finally
    S ← read(prompt_embeds, embed_pseudo, extend(position_ids, P))
    还原 text.layers
    scores ← mean over heads of renormalise(mean over P rows of S)      # [V]
    keep   ← 按 scores 取 top-k，k = max(min_visual_tokens, ceil(V × keep_ratio))，保持原序
    按索引裁剪 inputs_embeds / attention_mask / position_ids / cache_position
    存下生成状态给 decode 侧的补丁；返回 PrunedPrefill
```

`text.layers` 的替换是**纯省算力，不是正确性手段**：`hidden_states[L]` 是它下面各层的因果输出，和 28 层前向里的同一项逐位相同。生成本身仍然跑满 28 层。伪 token **绝不会**进入返回的 `inputs_embeds`，所以 cache position 和 `rope_deltas` 的记账不受影响。

---

## 2. 已测数字

在 Cambrian OCR_VQA 子集（5,000 训练 / 500 留出，`V ≈ 1315`，答案 `A ≈ 5.4` 行，问题 token 6~18 个、均值 9.89）上，`P = 32`、`L = 1`、2000 步、batch 32、lr 2e-4 cosine：

| 变体 | KL（28 头求和） | ÷ 均匀基线 |
|---|---|---|
| **伪 token（32 行，训练后）** | **0.2832** | 0.0119 |
| 单行最好的已学行 | 0.9728 | 0.0408 |
| **全部问题 token（~9.9 行）** | **1.8750** | 0.0787 |
| 32 行随机初始化 | 2.8045 | 0.1177 |
| 均匀分布基线（解析值） | 23.8371 | 1.0000 |

- **同单位下伪 token 比「全部问题 token」好 6.62 倍，500/500 个样本全部获胜**（逐样本比值中位数 0.123）。
- 逐行比较则**不吃亏也不占优**：k=1 时单个问题 token 是 3.2921，单个已学行是 3.6520——问题 token 反而好 11%。伪 token 的优势来自**32 行互不冗余**：已学行从 k=1 到 k=32 改善 12.9 倍，而问题行 k≈4 就饱和了（4 行到 9.9 行只改善 8%）。这也符合直觉——问题是一句连贯的话，行之间高度冗余。
- 把问题行重复到 32 行**不会改变任何数字**（概率空间求均值，重复行零信息），所以问题侧不存在「如果它也有 32 行」的更好版本。

**尚未测量：** 部署时保留率是 19.8%（`k ≈ 261` 个视觉 token），而 `train_layer1.py` 里 `eval_keys` 目前只测到 33/66/132，**`k ≈ 261` 附近的 `recall@k` 还没有数字**。KL 只是代理指标，真要判断掉点多少，需要把 264 补进 `eval_keys`（见 `scripts/train_layer1.py` 中 `eval_keys` 那一段）或者直接跑第 6.2 节的 VLMEvalKit 端到端评测。

---

## 3. 环境准备

要求 Python ≥ 3.10、一张能跑 7B bf16 的 GPU（约 20 GB 显存足够裁剪后的训练和推理）。

```bash
conda create -n lookattn python=3.10 -y
conda activate lookattn

# 按目标机器的 CUDA 版本选 torch wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

然后生成**打了补丁的隔离运行时**：

```bash
bash scripts/setup_runtime.sh
```

这一步会把 `transformers==4.52.0` / `tokenizers` / `huggingface-hub` / `qwen-vl-utils` 以 `--target` 方式装进项目内的 `.runtime/python`（约 155 MB），然后用 `scripts/patch_transformers.py` 往 `transformers` 里插入两处改动：Qwen2.5-VL 的 prefill 会调用 `hawk.pruning.prune_qwen_prefill`，以及生成过程中的记账会接受被裁短的 prompt。**你的 site-packages 不会被改动。**

成功时会打印：

```text
HAWK 0.1.0 Transformers 4.52.0
```

> **所有 Python 入口都必须用 `scripts/run.sh` 包一层**，它负责把 `src/` 和 `.runtime/python` 一起放进 `PYTHONPATH`：
>
> ```bash
> bash scripts/run.sh scripts/train_layer1.py --steps 20   # ✅
> bash scripts/run.sh python scripts/train_layer1.py       # ❌ run.sh 自己就是解释器
> ```
>
> 不经过 `run.sh` 直接 `python scripts/xxx.py` 会 import 到你 site-packages 里没打补丁的 transformers，症状是剪枝静默不生效。

下载模型：

```bash
bash scripts/run.sh scripts/download_model.py --local-dir models/Qwen2.5-VL-7B-Instruct
```

---

## 4. 快速验证：跑通推理（不需要任何数据）

本包自带一个已经训练好的 scorer（`checkpoints/scorer_best.pt`，228 KB，就是上表里 0.2832 那个），配一张示例图：

```bash
bash scripts/run.sh scripts/infer_lookhawk.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --scorer checkpoints/scorer_best.pt \
  --image assets/smoke_test.png \
  --keep-ratio 0.198
```

（等价于 `make infer`。）

通过的判据：打印里 `pruning.applied == true`，`kept_visual_tokens ≈ 0.198 × original_visual_tokens`，生成的文本非空。

`checkpoints/scorer_best.config.json` 是产出这个 checkpoint 的完整训练配置，供追溯。

想先看**没有** lookahead 的 HAWK 原始表现做对照：

```bash
make demo        # scripts/infer.py，不需要 checkpoint
```

---

## 5. 训练数据准备

### 5.1 语料组成

训练语料由 [Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M) 的三个子集拼成，做法参照 [VisionSelector](https://github.com/JulietChoo/VisionSelector) 的 *Dataset Preparation*：

| 数据集 | 训练量 | 内容 |
|---|---|---|
| **OCR_VQA** | 50K | 文档 / 书封文字，答案就是图里的字 |
| **ChartQA** | 16K | 图表数值问答（**池子只有 18,317 张图**，见第 5.3 节） |
| **CoCo** | 10K | 自然图像的通用描述 |
| **合计** | **76K** | |

> **`K` 的单位是「图片张数」**（`--train-count`），val 默认再取 10%。注意这和 `Cambrian737k.jsonl` 的**标注行数**不是一回事——ChartQA 一张图 1.54 行、CoCo 一张图 4.11 行，见第 5.3 节的对照表。

**与 [VisionSelector](https://github.com/JulietChoo/VisionSelector) 的关系：** 下载方式、数据清单、以及**切分的两阶段形状**都按它来——它先用 `datasets/filter_json.py` 把标注按 `image` 前缀分流成 `ocr_vqa_cambrian.jsonl` / `chartqa_cambrian.jsonl` / …（**全量输出，不采样、不切 train/val、不去重**），采样率则留到训练时用数据集名后缀 `%N` 交给 `data_qwen.py` 处理。本项目的 `--stage filter` 就是它的 `filter_json.py`，schema 逐字一致；区别在第二阶段：它**没有** train/val 切分（四个训练脚本全是 `--eval_strategy "no"`），而本项目的 scorer 训练要靠 val 的 `recall@k` / `kl` 选 `scorer_best.pt`，所以 `--stage split` 补上了这一步，并且**按图片去重**——`filter_json.py` 不做去重，照搬的话 CoCo 会有 9.3% 的 val 图片同时出现在 train 里。

### 5.2 下载

标注文件（766 MB）和三个图片归档都要下。国内直连 HF 很慢，走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd <项目根>

# 标注文件
hf download nyu-visionx/Cambrian-10M --repo-type dataset \
  --include "jsons/Cambrian737k.jsonl" --local-dir datasets

# 三个图片归档
hf download nyu-visionx/Cambrian-10M --repo-type dataset \
  --include "ocr_vqa.tar.gz" "chartqa.tar.gz" "coco.tar.gz" --local-dir datasets
```

如果没有 `hf` 命令（它在 `huggingface_hub ≥ 0.30` 里，本项目把它装进了 `.runtime/` 而不是基础环境），用 `pip install -U huggingface_hub`，或者直接用直链——VisionSelector README 给的就是这些：

```text
https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/jsons/Cambrian737k.jsonl
https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/ocr_vqa.tar.gz
https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/chartqa.tar.gz
https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/coco.tar.gz
```

体积（实测）：

| 文件 | 大小 |
|---|---|
| `jsons/Cambrian737k.jsonl` | 766 MB |
| `ocr_vqa.tar.gz` | 3.2 GB |
| `chartqa.tar.gz` | 708 MB |
| `coco.tar.gz` | **18.1 GB** |
| **合计** | **≈ 22.8 GB** |

> CoCo 归档 18 GB，而你只从里面取 10K 张图。tar.gz 的成员没法按 HTTP 范围按需下载，所以这 18 GB 躲不掉——但它**只用于下一步的选择性解压**，解完就能删。如果你对 CoCo 的来源不敏感，也可以换成 `lmms-lab/COCO-Caption` 或原版 COCO train2017，代价是它不再是 Cambrian 的子集，分布上会和另两个不一致。

### 5.3 切子集：两阶段

`scripts/prepare_cambrian_subset.py` 有两个阶段，`--dataset` 取 `ocr_vqa` / `chartqa` / `coco` / `textvqa`。它**只写文本**——几个 jsonl 加一个成员清单——从不碰归档，所以很便宜、可以反复跑。

**阶段 1 `--stage filter`：照 VisionSelector 的 `filter_json.py`，全量分流。**

```bash
bash scripts/run.sh scripts/prepare_cambrian_subset.py --stage filter \
  --cambrian-jsonl datasets/jsons/Cambrian737k.jsonl \
  --dataset ocr_vqa chartqa coco \
  --out-dir datasets
```

产出 `datasets/{ocr_vqa,chartqa,coco}_cambrian.jsonl`，schema 与 VisionSelector 逐字一致：

```json
{"id": "10095", "image": "chartqa/train/png/10095.png",
 "conversations": [{"from": "human", "value": "Is the value of Favorable 38 in 2015?\n<image>"},
                   {"from": "gpt", "value": "Yes"}]}
```

**全量输出**（实测 chartqa 28,138 行 / coco 364,100 行 / ocr_vqa 80,000 行；整份 766 MB 扫完约 5 秒）。这一步不采样、不切分、不去重——和 `filter_json.py` 一样。

**阶段 2 `--stage split`：按图片去重后切 train/val。**

```bash
for spec in "ocr_vqa 50000 5000" "chartqa 16000 1600" "coco 10000 1000"; do
  set -- $spec
  bash scripts/run.sh scripts/prepare_cambrian_subset.py --stage split \
    --cambrian-jsonl "datasets/$1_cambrian.jsonl" \
    --dataset "$1" --train-count "$2" --val-count "$3" \
    --image-root datasets \
    --out-dir "data/cambrian-$1"
done
```

（`--cambrian-jsonl` 也可以直接指回 `datasets/jsons/Cambrian737k.jsonl`，脚本会自己按 `--dataset` 前缀过滤；用阶段 1 的产物只是省一遍解析。）

每个 `data/cambrian-<名字>/` 下产出 `train.jsonl`、`val.jsonl`、`members.txt`（归档成员清单，喂给 `tar -T`）。`--image-root` 会被解析成**绝对路径**写进 `image` 字段，所以之后从哪个目录启动训练都行。

**两个保证，都是构造出来的，不是事后抽查：**

1. **`--train-count` / `--val-count` 数的是「图片」。** 两个集合从**同一个图片池**里无放回抽取，所以 val 图**不可能**出现在 train 里。脚本自己打印 `overlap=0` 作为证据。`--pairs-per-image`（默认 1）决定每张图取几对 q/a；取 1 时样本数 == 图片数。
2. **同一张图的所有 q/a 归到同一边。** 这是关键：corpus 里一张图有多行标注（ChartQA 1.54 行/图、CoCo 4.11 行/图），旧版本抽的是**行**，同一张图就会跨在切分线上。实测旧版 val 泄漏率 CoCo **9.3%**（991 张 val 图里 92 张也在 train）、ChartQA **1.4%**、OCR_VQA 0%；新版同一配置**都是 0%**。

**ChartQA 的池子上限。** chartqa 只有 **18,317 张去重图片**（28,138 行），所以上面写 16,000/1,600 而不是 25,000/2,500。超过池子会直接报错退出，不会静默截断：

```text
only 18317 distinct chartqa images available, need 27500 (25000 train + 2500 val).
Lower --train-count/--val-count; the pool is 18317 images.
```

想要更多 chartqa 样本就调 `--pairs-per-image 2`——实测整池 18,317 张图能出 **26,219 个样本**（1.43 对/图，不到 2 是因为部分图只有一对）。但那时样本数就不再等于图片数了。

各数据集真实可用量——这是我在本机对 `Cambrian737k.jsonl` **全量统计**的结果，不是估的（VisionSelector README 那张表只有数量级）。「去重图片」那一列就是 `--train-count` 的上限：

| 前缀 | 标注行 | 去重图片 | 行/图 |
|---|---|---|---|
| coco | 364,100 | **88,524** | 4.11 |
| vg | 86,417 | 86,417 | 1.00 |
| ocr_vqa | 80,000 | 80,000 | 1.00 |
| gqa | 72,140 | 72,140 | 1.00 |
| chartqa | 28,138 | **18,317** | 1.54 |
| textvqa | 21,953 | 21,953 | 1.00 |
| ai2d | 15,501 | — | — |
| docvqa | 14,999 | — | — |
| dvqa | 13,000 | — | — |

（`—` 表示我只统计了行数，没对这四万多个短前缀单独去重。全部 696,248 行里包含这 9 个前缀和少量更小的子集。）

**训练集 vs 评测基准：查过了，不冲突——但不是因为这个管线做了什么 filter，是因为归档本身就只装了 train split。**

| 训练来源 | 归档里实际有的 | 对应评测集用的 split | 是否重叠 |
|---|---|---|---|
| `chartqa.tar.gz` | **只有 `chartqa/train/png/`（18,317 张）**，`test/png/`、`val/png/` 是**空目录** | `ChartQA_TEST`（官方 test） | **否** |
| `textvqa.tar.gz` | **只有 `textvqa/train_images/`（21,953 张）** | VLMEvalKit 的 TextVQA 用 **val** split | **否** |
| `coco.tar.gz` | `coco/train2017/` | 本项目没评 COCO captioning | 不适用 |
| `ocr_vqa.tar.gz` | 平铺 80,000 张，**不带 split 信息** | 本项目没评 OCR-VQA | 不适用 |

两点保留意见：

- ChartQA 那一行只验证到「归档里没有 test 图片，所以训练管线**够不着** test」。至于官方 train/test 的图片本身是否图像级不相交，我**没在本机实测**（test 图片不在归档里，要比对得另外下 `lmms-lab-encoder/chartqa` 的 test parquet，见 `scripts/prepare_chartqa_hf.py`）。
- `ocr_vqa` 是**揉平**的：OCR-VQA 官方有 train/val/test，Cambrian 把它们混在一个 `images/` 里，路径上看不出归属。现在不评 OCR-VQA 所以无害，但**以后要加 OCR-VQA 基准，这批数据不能直接用**。

### 5.4 选择性解压

`members.txt` 里是归档内的成员名，`tar -T` 只解出你要的那几张图，不会展开整个归档：

```bash
for name in ocr_vqa chartqa coco; do
  tar -xzf "datasets/$name.tar.gz" -C datasets -T "data/cambrian-$name/members.txt"
done
```

实测 5,500 张 OCR_VQA 图约 243 MB（≈44 KB/张），76K 样本对应的图大约几个 GB。解完 22.8 GB 的归档就可以删了。

### 5.5 合并成一份语料

三份 jsonl 的 `image` 都指向同一个 `datasets/` 树下的不同子目录，schema 完全一样，直接拼接即可：

```bash
mkdir -p data/cambrian-mix
cat data/cambrian-ocr_vqa/train.jsonl \
    data/cambrian-chartqa/train.jsonl \
    data/cambrian-coco/train.jsonl    > data/cambrian-mix/train.jsonl
cat data/cambrian-ocr_vqa/val.jsonl \
    data/cambrian-chartqa/val.jsonl \
    data/cambrian-coco/val.jsonl      > data/cambrian-mix/val.jsonl

wc -l data/cambrian-mix/*.jsonl      # 期望 76000 / 7600
```

### 5.6 ⚠️ 76K 规模目前**跑不动**，先看这段再动手

`Layer1Trainer` 在**开训之前**就把所有样本编码好并全部常驻显存（`__init__` 里对每个样本调 `_prepare`，`self.items` 里持有 `prompt_embeds`）。按已实测的每行规模推算：

| | 每行 | × 76,000 行 |
|---|---|---|
| 显存（`prompt_embeds` `[1, ~1346, 3584]` bf16） | ≈ 9.6 MB | **≈ 734 GB** |
| 内存（编码期 `pixel_values` float32） | ≈ 25 MB | **≈ 1.9 TB** |

734 GB 在 8×A800（合计 640 GB）上都放不下。**这不是配置问题，是当前 trainer 的结构问题**：它假设整个语料能一次性驻留。

现有 5,000 行能跑通，对应约 48 GB 显存——已经在单卡 80 GB 的舒适区边缘了。可行的路只有三条：

1. **先降规模验证**（推荐先做这个）：把第 5.3 节的循环换成 `"ocr_vqa 3000 300" "chartqa 1500 150" "coco 600 60"`（单位是**图片**），train 合计 5,100 张 / 5,100 个样本，和已跑通的规模同量级，先把配方跑通再谈扩量。
2. **分片独立训练**：`--limit-train` 只是限制**编码**的行数，每个分片是一次**独立**训练、各有自己的 scorer，不会累积。要合起来用「取最好的一个」，或者把多个 scorer 的 `embed_pseudo` 平均（没验证过，别当默认做法）。
3. **改 trainer 做流式**：把 `self.items` 从「全部预编码常驻显存」改成「CPU/磁盘缓存 + 每个 batch 前搬到 GPU」。这是唯一能真正吃下 76K 的路，但要动 `src/lookhawk/layer1.py` 的 `Layer1Trainer`，属于代码改动。

---

## 6. 训练与评测

### 6.1 训练 scorer

**数据准备见第 5 节**（最小可跑规模的做法在第 5.6 节）。下面假设 jsonl 已经生成好了。

`--train-jsonl` / `--val-jsonl` 是 JSONL，一行一个样本，含一个图片路径，加上 `question`/`answer` 或者 `conversations`/`messages` 列表：

```json
{"image": "/abs/path/datasets/ocr_vqa/images/140031996X.jpg", "question": "Who is the author of this book?", "answer": "John Smith"}
```

图片路径**绝对或相对都行**（相对时按启动训练时的工作目录解析）。`prepare_cambrian_subset.py` 写的是绝对路径，所以从哪个目录启动都不会错。

每一行会**过两遍** chat template（只有 prompt / prompt + answer），并且要求编码后的 prompt ids 是完整 ids 的前缀；不一致会直接抛错，而不是静默把答案切错位置。答案就是后缀，去掉尾部的 `<|im_end|>` 和换行，再截到 `--teacher-max-rows`。

用自带的冒烟样例（`data/smoke/`，20 KB）先验证跑得通：

```bash
bash scripts/run.sh scripts/train_layer1.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --train-jsonl data/smoke/train.jsonl \
  --val-jsonl data/smoke/val.jsonl \
  --out-dir outputs/smoke \
  --min-pixels 3136 --steps 20 --batch-size 1 --limit-train 8 --limit-val 4
```

正式训练（Cambrian OCR_VQA 5,000/500，与第 2 节数字同配置）：

```bash
bash scripts/run.sh scripts/train_layer1.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --train-jsonl data/cambrian-ocr_vqa/train.jsonl \
  --val-jsonl data/cambrian-ocr_vqa/val.jsonl \
  --out-dir outputs/lookhawk_layer1 \
  --num-pseudo 32 --score-layer 1 \
  --lr 2e-4 --steps 2000 --batch-size 32 \
  --save-best
```

跑混合语料就把两行换成 `data/cambrian-mix/{train,val}.jsonl`——但先读第 5.6 节，76K 行当前跑不动。

输出目录里会有 `config.json`、`scorer.pt`，加了 `--save-best` 还会有按验证 KL 选出的 `scorer_best.pt`。

**启动时先看那行诊断：**

```text
[diag] sample 0: kl=3.420177 teacher_kl_uniform=0.810815 recall@1=1.0000 ...
```

`teacher_kl_uniform` 是教师各头对均匀分布的 KL（**头平均**，注意与损失的**头和**不同单位）。它接近 0 意味着教师本身在视觉 token 上就是平的，KL 再怎么训练也动不了——看到这个数再决定要不要信损失曲线。

**准备数据**：见第 5 节。最小可跑规模的做法在第 5.6 节。

### 6.2 评测

**离线 `recall@k` / KL**（不依赖任何外部评测框架）：

```bash
bash scripts/run.sh scripts/eval_recall_lookhawk.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --scorer checkpoints/scorer_best.pt \
  --jsonl data/cambrian-ocr_vqa/val.jsonl
```

`recall@k` 在合并后的 `[V]` 向量上算（`|top-k(student) ∩ top-k(teacher)| / k`），所以它评的就是实际部署的那个选择。

**VLMEvalKit 端到端**（native resolution，论文同款设置）：

```bash
pip install -r requirements-eval.txt
bash scripts/setup_vlmevalkit.sh          # 把 ms-vlmeval 复制到 .runtime/VLMEvalKit 并打补丁
make prepare-vlmeval                      # 准备 RealWorldQA / ScienceQA / ChartQA / TextVQA / MME 的本地 tsv
DATASETS=RealWorldQA KEEP_RATIO=0.20 CUDA_DEVICES=0 bash scripts/evaluate_vlmeval.sh
```

结果落在 `vlmeval_results/<RUN_NAME>/`，用 `scripts/summarize_vlmeval.py` 汇总。

> ⚠️ **`evaluate_vlmeval.sh` 目前走的是 HAWK 的 layer-0 分支加 head-weight 机制，和 lookahead scorer 是两套完全不同的打分。** 它会生成一个 `HawkQwen2VLChat` 的 config，里面只有 `hawk_keep_ratio` / `hawk_head_weights`，没有 scorer 的位置。想用 VLMEvalKit 端到端评 lookahead，需要往 `scripts/evaluate_vlmeval.sh` 里生成 config 的那段加上 scorer 路径，并让 `hawk.vlmeval_adapter` 在 `configure_model` 之后调用 `lookhawk.inference.load_lookahead_for_model(model, scorer_path)`。这一步**尚未接**，第 2 节的 KL 数字全部来自离线管线。

---

## 7. 关键 flag

`scripts/train_layer1.py`：

| Flag | 默认 | 说明 |
|---|---|---|
| `--model-path` | `models/Qwen2.5-VL-7B-Instruct` | |
| `--train-jsonl` / `--val-jsonl` |  `data/smoke/*.jsonl` | 见 6.1 |
| `--out-dir` | `outputs/lookhawk_layer1_smoke` | |
| `--num-pseudo` | 32 | `P`，追加的行数 |
| `--score-layer` | 1 | `L`，必须小于解码器层数 |
| `--teacher-max-rows` | 32 | 教师最多读多少个答案 token |
| `--temperature` | 1.0 | 写进 checkpoint |
| `--min-pixels` / `--max-pixels` | `1003520` / 不设 | **与评测 harness 对齐**，别随意改 |
| `--max-seq-len` | 2048 | prompt + answer 的总预算 |
| `--workers` | 8 | CPU 侧预处理进程数（解码/缩放/patchify） |
| `--steps` / `--batch-size` | 2000 / 32 | batch 花时间不花显存 |
| `--lr` / `--lr-schedule` / `--warmup-ratio` / `--min-lr` | `2e-4` / `cosine` / 0.02 / 0.0 | `--lr-schedule none` 保持恒定学习率 |
| `--max-grad-norm` | 1.0 | 注意损失是 28 头**求和**，梯度尺度比头平均大一个量级 |
| `--limit-train` / `--limit-val` | 不设 | 限制编码样本数 |
| `--log-every` / `--eval-every` | 10 / 50 | `--eval-every 0` 关闭验证 |
| `--save-best` | 关 | 按验证 KL 存 `scorer_best.pt` |
| `--device` / `--dtype` / `--seed` | `auto` / `auto` / 0 | |

`scripts/infer_lookhawk.py`：

| Flag | 默认 | 说明 |
|---|---|---|
| `--model-path` / `--scorer` / `--image` | 必填 | |
| `--keep-ratio` | 0.198 | 保留比例；`1 - 0.198` 就是 HAWK 主打的 80.2% 剪枝 |
| `--min-visual-tokens` | 1 | 保留数量的下限 |
| `--prompt` | 见 `--help` | 提示词 |
| `--max-new-tokens` | 64 | |
| `--min-pixels` / `--max-pixels` | 模型默认 | |
| `--device` / `--dtype` | `cuda:0` / `bfloat16` | |

---

## 8. 目录结构

```text
requirements.txt                    基础环境依赖
requirements-eval.txt               评测依赖（VLMEvalKit / 数据集）
checkpoints/scorer_best.pt          已训练好的 scorer（P=32, L=1, τ=1.0）
checkpoints/scorer_best.config.json 产出该 checkpoint 的训练配置

src/hawk/config.py                  HAWK 的比例与归一化头权重
src/hawk/pruning.py                 打分、归一化、保序 top-k
src/hawk/integration.py             公开的 configure_model API
src/hawk/vlmeval_adapter.py         VLMEvalKit 的 model / dataset 适配器

src/lookhawk/layer1.py              前瞻打分器的全部核心：读数、KL 损失、trainer、checkpoint
src/lookhawk/inference.py           把 scorer 装到 HAWK 剪枝入口上
src/lookhawk/model_graph.py         解析外层 wrapper 与 text decoder
src/lookhawk/data.py                jsonl schema 与 chat template 编码
src/lookhawk/loss.py                recall@k、rank loss、layer-0 KL
src/lookhawk/train.py               layer-0 trainer（已被 layer1.py 取代）
src/lookhawk/read.py                layer-0 条件读数
src/lookhawk/modules.py             layer-0 scorer（lookahead 行 + LoRA）
src/lookhawk/config.py              layer-0 路径的 LookhawkConfig

scripts/run.sh                      设置 PYTHONPATH 后 exec 解释器（所有入口都走它）
scripts/setup_runtime.sh            生成 .runtime/python 并打补丁
scripts/patch_transformers.py       Qwen2.5-VL + 生成状态的补丁本体
scripts/train_layer1.py             训练入口（本项目主路径）
scripts/infer_lookhawk.py           单图推理入口（本项目主路径）
scripts/eval_recall_lookhawk.py     离线 recall@k / KL 报告
scripts/train_lookhawk.py           layer-0 scorer 训练入口
scripts/infer.py                    HAWK 原始路径的端到端推理
scripts/download_model.py           下载 Qwen2.5-VL
scripts/create_smoke_image.py       生成示例图
scripts/prepare_cambrian_subset.py  两阶段：--stage filter 全量分流 / --stage split 按图去重切 train-val
scripts/prepare_*_hf.py             其他数据集（ChartQA / TextVQA / MME）预处理
scripts/evaluate.py                 本地端到端生成 + 判分
scripts/evaluate_vlmeval.sh         VLMEvalKit 分布式 native-resolution 评测
scripts/setup_vlmevalkit.sh         准备并补丁 .runtime/VLMEvalKit
scripts/patch_vlmevalkit.py         补丁本体
scripts/run_vlmeval.py / summarize_vlmeval.py   评测执行与汇总
scripts/prepare_vlmeval_datasets.py VLMEvalKit 本地 tsv 准备
scripts/diagnose_lookhawk.py        打分诊断（头间差异、行间可区分性）
scripts/diagnose_read_control.py    读数对照实验

assets/method.png                   方法示意图
assets/smoke_test.png               示例图（由 create_smoke_image.py 生成）
data/smoke/*.jsonl                  冒烟样例，随包发布，第 4 节直接用
data/cambrian-<名字>/               第 5 节切出的语料（train/val.jsonl + members.txt），不入库
data/cambrian-mix/                  三份语料合并结果，不入库
datasets/                           下载的 Cambrian 标注与图片归档，不入库
```

---

## 9. 注意事项

- **只支持 Qwen2.5-VL-7B-Instruct + Transformers 4.52.0**。换版本要重新核对 `patch_transformers.py` 的两个插入点。
- **剪枝路径 batch size 必须是 1，`num_beams=1`**。这是打分读数本身的性质（各样本视觉 token 数不同），不是训练的限制。
- **注意力后端用 SDPA**。
- **layer-1 分支完全不看 `HawkConfig.score_normalization`**（`l2`/`softmax`/`minmax`）。给一个 `pseudo_token` checkpoint 传 `--score-normalization l2` 不会有任何效果，也不会报错。它只对 layer-0 分支有效。
- **lookahead scorer 目前只认图像 token**（训练时的 span 只匹配 image token），视频不在范围内。
- 显存与内存：5,000 行 OCR_VQA 在 CPU 侧编码时，`pixel_values` 是 `[5280, 1176]` float32、每行约 25 MB，全部解码完约 115 GB 常驻内存。`src/lookhawk/data.py` 用固定窗口（`2 × workers`）喂进程池并把张量拷进匿名内存，把 `/dev/shm` 峰值压到 0.48 GB——**不要**把那个窗口改回 `pool.imap` 的无界写法，在共享机器上会直接把 `/dev/shm` 撑爆。
- 训练结束后 `.runtime/`（155 MB）、`models/`、`outputs/` 都已被 `.gitignore` 排除。

---

## 10. 引用与许可

如果这个工作对你的研究有帮助：

```bibtex
@inproceedings{zhu2026hawk,
  title     = {HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models},
  author    = {Zhu, Qihui and Zhang, Tao and Wang, Yuchen and Wen, Zijian and
               Zhang, Mengjie and Chen, Shuangwu and Tan, Xiaobin and Yang, Jian and
               Liu, Yang and Dong, Zhenhua and Yu, Xianzhi and Pan, Yinfei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

Apache License 2.0，见 [LICENSE](LICENSE)。Qwen2.5-VL、Transformers、VLMEvalKit 及各评测数据集仍归各自许可与条款约束。
