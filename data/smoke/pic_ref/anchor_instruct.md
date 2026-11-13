You are an expert Scientific Illustrator for top-tier AI conferences (NeurIPS/CVPR/ICML).
Your task is to generate a professional "Illustration" (main figure for the paper) based on a research paper abstract and methodology.
I have provided you with a reference image. Please use it as a stylistic reference, but do not copy it rigidly. 
Create the figure in an editable PowerPoint format so that I can continue modifying it later.
**Abstract:**
Multimodal large language models (MLLMs) incur substantial computational and memory overhead due to the large number of visual tokens and their associated KV cache. To mitigate these costs, various visual token pruning methods have been proposed. However, most methods are designed for single-turn VQA and struggle to preserve visual information relevant to diverse questions across multi-turns. To address this challenge, we propose AnchorMerge, a learnable visual token pruning method for multi-turn VQA. Specifically, AnchorMerge partitions visual tokens into contiguous local regions and selects a representative token from each region as an anchor. It then aggregates information from the remaining tokens into each anchor through learnable weighted aggregation and gated fusion. To further enhance these representations, we introduce a dual-objective framework that combines downstream VQA task optimization with visual feature reconstruction. The former retains task-relevant information, while the latter captures fine-grained visual details. Although trained with VQA objectives, AnchorMerge performs query-independent pruning at inference time, allowing a single pruned representation to be reused across multi-turns. Extensive experiments on five MLLMs across three multi-turn VQA benchmarks demonstrate the effectiveness of AnchorMerge. At a 10\% visual token retention rate, AnchorMerge outperforms the strongest baseline by 3.77, 4.25, and 4.60 points on LLaVA-1.5-7B, LLaVA-NeXT-Vicuna-7B, and Qwen2.5-VL-7B, respectively. Meanwhile, it achieves a 2.28$\times$ speedup in prefilling and reduces GPU memory to 15.1\% of the full cache setting.
**Methodology:**
\section{Methodology}\label{sec:method}
\subsection{Multimodal Large Language Models}
An MLLM takes a visual input \(X_v\) (e.g., an image or video) and a textual sequence \(X_t\) as inputs. The visual input is processed by a vision encoder to extract visual features, which are then mapped into the embedding space of the LLM through a multimodal projector, producing a sequence of visual token embeddings \(H_v = \{h_v^1, h_v^2, \ldots, h_v^M\}\), where \(M\) denotes the number of visual tokens. This projection enables the LLM to process visual and textual information within a shared embedding space.Meanwhile, the textual input is tokenized and converted into a sequence of text embeddings \(H_t = \{h_t^1, h_t^2, \ldots, h_t^N\}\), where \(N\) denotes the number of text tokens. 
The two embedding sequences are then combined to form a unified multimodal input \([H_v; H_t]\), which is fed into the LLM for joint processing. In practice, the number of visual tokens can be much larger than that of text tokens, i.e., \(M \gg N\), particularly for high-resolution images and long videos. The LLM processes the combined sequence through multiple transformer layers and generates the output sequence \(Y = \{y_1, y_2, \ldots, y_O\}\) in an autoregressive manner. Specifically, the probability of generating \(Y\) conditioned on the multimodal input is given by:
\begin{equation}
    P(Y \mid H_v, H_t) = \prod_{i=1}^{O} P(y_i \mid y_{<i}, H_v, H_t),
\end{equation}
where \(O\) denotes the output length, \(y_{<i}\) represents all previously generated tokens, and each output token is predicted based on the preceding outputs and the multimodal context.

\subsection{Objective of Visual Token Pruning}
Visual tokens exhibit varying importance for MLLM generation, with some containing redundant information and others carrying essential details for answering specific questions. Formally, given the original visual token sequence \(H_v\) and the text embedding sequence \(H_t\), the objective is to obtain a pruned visual token sequence \(\tilde{H}_v\) with a predefined token budget \(m\), such that the output of the MLLM remains close to that produced using the full visual token sequence. This objective can be formulated as:
\begin{equation}
\tilde{H}_v^* = \operatorname*{arg\,min}_{|\tilde{H}_v|=m}
\mathcal{L}\left(
f([\tilde{H}_v;H_t]),
f([H_v;H_t])
\right),
\label{eq:obj}
\end{equation}
where \(f\) denotes the LLM, \(\tilde{H}_v^*\) represents the optimal compressed visual representation derived from the original visual tokens, and \(\mathcal{L}\) measures the difference between the outputs generated using the compressed and original visual representations. The token budget \(m\) controls the compression level. The key challenge is to construct a compact visual representation that preserves the information necessary for model understanding and generation. In particular, token importance depends not only on visual content but also on the textual instruction and downstream task. Therefore, an effective pruning strategy should preserve informative and task-relevant visual information while reducing redundant representations.

\subsection{AnchorMerge}
The framework of AnchorMerge is illustrated in Figure~\ref{fig:framework}.

\textbf{Method workflow.} Given the visual token sequence $H_v\in\mathbb{R}^{M\times d}$ and a token budget $m$, AnchorMerge constructs a compact representation $C\in\mathbb{R}^{m\times d}$ that preserves information required by the LLM. Unlike methods that retain a subset of visual tokens and discard the rest~\cite{divprune,holov,dart}, AnchorMerge further incorporates information from the pruned tokens into the retained representations. Inspired by HoloV~\cite{holov}, which partitions visual tokens into spatial crops and performs crop-wise token selection to preserve holistic visual context, we divide the raster-ordered visual tokens into $m$ contiguous segments and select one original token from each segment as an anchor. This uniform segment-wise selection ensures spatial coverage across the entire image, which is particularly important for multi-turn VQA, where future questions are unknown and may focus on different visual regions.

Specifically, we divide the $M$ visual tokens into $m$ contiguous segments, denoted by $S_1,\ldots,S_m$, each containing approximately $M/m$ tokens. Within each segment, we select the token with the highest average cosine similarity to all tokens in the same segment as the anchor:
\begin{equation}
\begin{aligned}
z_i&=\frac{h_i}{\lVert h_i\rVert},\qquad
s_i=\frac{1}{L_k}\sum_{j\in S_k}z_i^\top z_j,\\
i_k^\star&=\operatorname*{arg\,max}_{i\in S_k}s_i,\qquad
a_k=h_{i_k^\star},
\end{aligned}
\label{eq:anchor}
\end{equation}
where $h_i$ denotes the original feature of the $i$-th visual token, $z_i$ is its normalized feature, and $L_k=|S_k|$ is the number of tokens in the $k$-th segment. Here, $s_i$ measures the average similarity of token $i$ to all tokens in its segment, $i_k^\star$ denotes the index of the selected anchor, and $a_k$ is the anchor's representation. 
An anchor is a single representative of its segment and cannot capture the remaining content, which would otherwise be discarded. Inspired by the contextual token merging strategy in VisionZip~\cite{visionzip}, which aggregates information from non-dominant tokens into target tokens based on feature similarity, we further incorporate information from the pruned tokens into their corresponding anchors. Specifically, we compute the residual between each non-anchor token and its anchor, and use a lightweight scoring network together with cosine similarity to determine the contribution of each residual:
\begin{equation}
u_i=\operatorname{MLP}(r_i)+\cos(h_i,a_k),\,
w_i=\frac{\exp(u_i)}{\sum\limits_{j\in \mathcal{R}_k}\exp(u_j)},
\label{eq:weights}
\end{equation}
where $\mathcal{R}_k=S_k\setminus{i_k^\star}$ denotes the set of non-anchor tokens in segment $k$, $r_i=h_i-a_k$ is the residual of token $i$ relative to its anchor, and $\operatorname{MLP}(\cdot)$ is a two-layer scoring network with an intermediate GELU activation. $u_i$ measures the contribution of each residual, while $w_i$ is the normalized value by applying softmax over the non-anchor tokens in the same segment, satisfying $\sum_{i\in \mathcal{R}_k}w_i=1$.

To reflect the distribution of residual information within each segment, we summarize the weighted residuals using their mean and standard deviation:
\begin{equation}
\mu_k=\sum_{i\in \mathcal{R}_k}w_i r_i,\qquad
\sigma_k=\sqrt{\sum_{i\in \mathcal{R}_k}w_i(r_i-\mu_k)^2},
\end{equation}
where $\mu_k$ reflects the dominant direction of the residuals and $\sigma_k$ reflects their spread. Together, these two statistics provide a compact summary of the residual distribution within segment $k$. 
To incorporate this statistical information into the anchor representation, we concatenate the two statistics into $v_k=[\mu_k\Vert\sigma_k]\in\mathbb{R}^{2d}$ and feed it into two lightweight networks. Specifically, a write-in network maps $v_k$ to the correction vector $\delta_k$, which specifies the feature update to be applied to the anchor. Meanwhile, a gating network predicts a scalar gate $g_k\in(-1,1)$ to adaptively control the contribution of this correction. The gate allows each segment to adjust its anchor representation with a different intensity. It is calculated as follows:
\begin{equation}
\delta_k=W_2\operatorname{GELU}(W_1v_k+b_1)+b_2,\,
g_k=\tanh\big(\phi(v_k)\big),
\label{eq}
\end{equation}
where the write-in network is a two-layer MLP, and $\phi$ is a two-layer MLP with a scalar output. The gating network is shared across all segments and takes their respective $v_k$ as input, producing segment-specific gates. We then integrate the correction vector $\delta_k$ into the anchor through gated fusion:
\begin{equation}
c_k=a_k+g_k\delta_k,\qquad
c_k\leftarrow c_k\cdot\frac{\lVert a_k\rVert}{\lVert c_k\rVert}.
\label{eq:fuse}
\end{equation}
The updated anchors are concatenated to form the compact representation $C=[c_1;\ldots;c_m]$ of the entire image, which is concatenated with subsequent queries and fed into the LLM for autoregressive generation across multi-turns.

\textbf{Training objective.} AnchorMerge is trained with a dual-supervision objective consisting of a VQA task loss and a visual feature reconstruction loss. Given an image-question pair, we first obtain the compressed visual representation $C\in\mathbb{R}^{m\times d}$ using AnchorMerge, where $m$ is fixed to 10\% of the original number of visual tokens $M$ during training. We then concatenate $C$ with the question tokens and feed them into the LLM to predict the answer. During training, the vision encoder, projector, and LLM are frozen, while only AnchorMerge and the reconstruction decoder are optimized. Gradients are backpropagated through the frozen LLM to update AnchorMerge. The VQA task loss is defined as the standard cross-entropy loss over the ground-truth answer tokens:
\begin{equation}
\mathcal{L}_{{ce}}
=-\frac{1}{|\mathcal{A}|}
\sum_{t\in\mathcal{A}}
\log P_\theta(y_t\mid y_{<t},Q,C),
\label{eq:ce}
\end{equation}
where $Q$ denotes the input question, $C$ is the compressed visual representation, $y_t$ is the ground-truth answer token, and $\mathcal{A}$ denotes the positions of answer tokens. 

Existing learning-based visual token pruning methods have explored leveraging downstream task supervision~\cite{lightvla}. However, in multi-turn VQA, the same image may be queried multiple times. Consequently, optimizing only for the current VQA task may cause the compressor to overlook visual details that are potentially useful for subsequent questions. To address this limitation, we introduce a visual feature reconstruction objective to encourage the compressed representation to preserve information from the original visual features.

Specifically, we employ a lightweight, single-layer Transformer decoder based on cross-attention to reconstruct the original visual features $H_v\in\mathbb{R}^{M\times d}$ from $C$. We first generate a fixed sinusoidal positional embedding matrix $P\in\mathbb{R}^{M\times d}$. The positional embedding at the $i$-th position is defined as:
\begin{equation}
P_{i,2j}=\sin\left(\frac{i}{10000^{2j/d}}\right),\,
P_{i,2j+1}=\cos\left(\frac{i}{10000^{2j/d}}\right).
\end{equation}
We then construct the inputs to the decoder as:
\begin{equation}
    Q=PW_p+b_p,\qquad U=CW_c+b_c,
\end{equation}
where $Q$ provides the positional queries, while $U$ serves as both the key and value input. The decoder performs cross-attention followed by a feed-forward network, with pre-normalization and residual connections. The reconstruction is formulated as:
\begin{equation}
R=W_o\mathcal{T}_{\psi}
\left(Q,U\right)+b_o,
\end{equation}
where $\mathcal{T}_{\psi}$ denotes the Transformer decoder, and $W_o,b_o$ project its output back to the original feature dimension. The resulting $R\in\mathbb{R}^{M\times d}$ represents the reconstructed visual features. Since the sinusoidal positional embeddings are generated on the fly, the decoder accommodates varying numbers of visual tokens without requiring a fixed positional embedding table. Importantly, the decoder is used only during training and is discarded at inference, introducing no additional inference overhead.
We use the original visual features $H_v$ as the reconstruction target and define the reconstruction loss as a normalized mean squared error (MSE):
\begin{equation}
\mathcal{L}_{{r}}=
\frac{\operatorname{MSE}(R,H_v)}
{\operatorname{Mean}(H_v^2)},
\end{equation}
where $\operatorname{MSE}(R,H_v)$ computes the mean squared error between the reconstructed and original features,and $\operatorname{Mean}(H_v^2)$ denotes the mean squared value of all elements in $H_v$. This normalization reduces the dependence of the loss on the magnitude of the target features. By minimizing this loss, the compressed representation is encouraged to preserve visual information from the original feature sequence.
The two supervision signals are jointly optimized using:
\begin{equation}
\mathcal{L}
=\mathcal{L}_{{ce}}
+\lambda\mathcal{L}_{{r}},
\label{eq:total_loss}
\end{equation}
where $\lambda$ controls the contribution of visual feature reconstruction. 
**Visual Style Requirements:**
1.  **Style:** Flat vector illustration, clean lines, academic aesthetic. Similar to figures in DeepMind or OpenAI papers.
2.  **Layout:** Organized flow (Left-to-Right, Top-to-Bottom, Circular and other shapes). Group related components logically.
3.  **Color Palette:** Professional pastel tones. White background.
4.  **Text Rendering:** You MUST include legible text labels for key modules or equations mentioned in the methodology (e.g., "Encoder", "Loss", "Transformer").
5.  **Negative Constraints:** NO photorealistic photos, NO messy sketches, NO unreadable text, NO 3D shading artifacts.
**Generation Instruction:**
Highlight the core novelty. Ensure the connection logic makes sense.
