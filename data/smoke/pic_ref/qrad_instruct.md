You are an expert Scientific Illustrator for top-tier AI conferences (NeurIPS/CVPR/ICML).
Your task is to generate a professional "Illustration" (main figure for the paper) based on a research paper abstract and methodology.
I have provided you with a reference image. Please use it as a stylistic reference, but do not copy it rigidly. 
Create the figure in an editable PowerPoint format so that I can continue modifying it later.

**Abstract:**

Multimodal large language models (MLLMs) have achieved remarkable progress in image and video understanding, but the large number of visual tokens introduces substantial computational and memory overhead. Visual token pruning offers an effective solution, yet existing methods may suffer from query-agnostic selection, unreliable attention signals, or indirect supervision for estimating token importance. 
To address these limitations, we propose \textbf{Q}uery-aware \textbf{R}etrieval-head \textbf{A}ttention \textbf{D}istillation (\textbf{QRAD}), a lightweight, plug-and-play framework for accurate and flexible visual token pruning. Motivated by the reliability of deeper-layer attention for estimating token importance, \textbf{QRAD} identifies deep-layer retrieval heads that localize visual evidence. It then distills their attention distributions into a lightweight scorer that predicts them directly from visual and textual features. We further introduce a ListMLE loss to enhance the scorer's ability to rank important visual tokens.
Extensive experiments on image and video benchmarks show that \textbf{QRAD} consistently delivers state-of-the-art performance across multiple MLLMs. Specifically, at a 20\% visual token retention budget, \textbf{QRAD} improves the average performance by 7.69 points over the strongest baseline on Qwen2.5-VL-7B, retaining 94.66\% of the full cache performance. Moreover, \textbf{QRAD} reduces end-to-end latency by 28\% compared with the full cache setting and decreases GPU memory usage, achieving efficient inference while maintaining strong performance. Additional experiments across different retention budgets and ablation studies demonstrate the adaptability of \textbf{QRAD} and the effectiveness of its individual components.

**Methodology:**

\section{Methodology}\label{sec:method}
\subsection{Objective of Visual Token Pruning}
During MLLM inference, only a small subset of visual tokens is typically relevant to the question, resulting in substantial redundancy among visual tokens. Given the original visual representations $H_v$ and text token representations $H_q$, the objective of visual token pruning is:
\begin{equation}
    \hat{H}_v^{\star} = \operatorname*{arg\,min}_{|\hat{H}_v| = k}\; \mathcal{L}\left(f([\hat{H}_v; H_q]), f([H_v; H_q])\right),
\label{eq:obj}
\end{equation}
where $\hat{H}_v$ denotes the retained visual representations, $k$ is the number of retained visual tokens, and $f$ is the LLM model. The challenge is to find the optimal subset $\hat{H}_v$ that minimizes the loss $\mathcal{L}$ while preserving the semantic information of the original visual representations.

\subsection{Observation} \label{sec:obs}
FastV~\cite{fastv} ranks visual tokens according to the attention they receive from the last token in the second LLM layer and retains only the top-ranked tokens for subsequent layers to reduce computational costs. This approach assumes that shallow-layer attention can effectively identify query-relevant visual tokens. However, shallow-layer attention may be affected by attention sinks and biases~\cite{sink,dsas}, making it less reliable for estimating token importance. We hypothesize deep-layer attention distributions could provide more reliable signals. To validate this, we conduct an analysis on 5,000 training samples from each of three datasets: OCRVQA~\cite{ocrvqa}, TextVQA~\cite{textvqa}, and COCO Karpathy~\cite{karpathy2015deep}, as they cover document-like text, in-the-wild text, and general visual scenes, and all involve short-answer QA. We use Qwen2.5-VL-7B~\cite{qwen2.5} for evaluation; the same analyses on Llava-1.5-7B~\cite{llava1.5} and Qwen3-VL-8B~\cite{qwen3vl} are provided in Appendix~\ref{app:1}.

Specifically, we first perform a full forward pass to extract the last-token attention over visual tokens at each layer. Based on these attention scores, we capture the top-20\% visual tokens at each layer. We then construct a reduced visual input using the selected tokens and feed it into the model again, performing a new prefilling pass followed by autoregressive decoding to generate the answer. By evaluating the resulting VQA performance, we measure how effectively the attention at each layer identifies the visual tokens necessary for answering questions. As shown in Figure~\ref{fig}, the results reveal that shallow-layer attention is insufficient for identifying visual evidence required for answering, while deep-layer attention generally provides better token selection signals, with the best results achieved at a later layer rather than the final layer. Appendix~\ref{app:1} shows similar trends on LLaVA-1.5-7B and Qwen3-VL-8B. We therefore select the layer that achieves the best average performance across the three datasets as the teacher layer in QRAD, providing consistently strong performance across different tasks.
These findings establish deep-layer attention as a more reliable signal for visual token importance estimation. This motivates us to explore whether a lightweight learnable module can \textbf{predict deep-layer attention scores from the initial text and visual representations}, enabling accurate visual token pruning.

\subsection{Retrieval-head Identification}
\label{sec:heads}
Attention heads in LLMs are functionally specialized~\cite{hawk,vera,baek2025large,park2026retrieval}: some heads retrieve relevant context, and some attend locally~\cite{duo}. Consequently, averaging the attention of all heads in the selected layer is not an optimal importance signal: the average could dilute the heads that localize the evidence. 
To identify retrieval heads, we use the training samples introduced in Section~\ref{sec:obs} with Qwen2.5-VL-7B. For each head, we first evaluate its ability to identify important tokens following Section~\ref{sec:obs}, replacing layer-level attention with the attention of an individual head for top-20\% visual token selection. We then measure how much attention each head assigns to visual tokens. Specifically, we sum the attention weights from the last token $r_{\mathrm{last}}$ over all visual tokens $N_v$ and average them across the sampled dataset:
\begin{align}
\alpha_h &= \frac{1}{|\mathcal{D}|}
\sum_{i\in\mathcal{D}}\sum_{j\in N_v}
A_{l^*,h}^{(i)}[r_{\mathrm{last}}, j], \\
A_{l^*,h}^{(i)} &=
\mathrm{softmax}\left(
\frac{Q_{l^*,h}^{(i)}(K_{l^*,h}^{(i)})^\top}
{\sqrt{d_h}}
\right),
\end{align}
where $\mathcal{D}$ denotes the sampled dataset, $l^*$ is the selected deep layer, and $Q_{l^*,h}^{(i)}$ and $K_{l^*,h}^{(i)}$ are the query and key matrices of head $h$ for sample $i$, respectively. $d_h$ is the head dimension. Here, $\alpha_h$ measures the average attention mass that head $h$ assigns to visual tokens, with a larger value indicating higher attention to visual content.
As shown in Figure~\ref{fig:heads}(a), $\alpha_h$ exhibits a clear separation between heads that attend more to visual tokens and those that do not. Interestingly, higher attention to visual tokens does not necessarily provide better token selection, as some heads with high $\alpha_h$ have relatively low accuracy. Nevertheless, heads with the highest $\alpha_h$ values still achieve high accuracy.

We therefore further examine the attention patterns of each head across question and generation positions. We define the generation-to-question attention ratio as:
\begin{align}
\rho_h = \frac{1}{|\mathcal{D}|}
\sum_{i\in\mathcal{D}}
\frac{
\sum\limits_{j\in N_v} A_{l^*,h}^{(i)}[r_{\mathrm{last}},j]
}{
\frac{1}{|\mathcal{R}_q^{(i)}|}
\sum\limits_{r\in\mathcal{R}_q^{(i)}}
\sum\limits_{j\in N_v} A_{l^*,h}^{(i)}[r,j]
},
\end{align}
where $\mathcal{R}_q^{(i)}$ denotes the positions of question tokens in sample $i$. A larger $\rho_h$ indicates that a head increases its attention to visual tokens more substantially at the generation position relative to the question positions.
As shown in Figure~\ref{fig:heads}(b), some heads with high accuracy exhibit a pronounced increase in attention to visual tokens at the generation position, i.e., a larger $\rho_h$. Similar experimental results are observed on LLaVA-1.5-7B and Qwen3-VL-8B, as shown in Appendix~\ref{app:2}.

Based on these observations, we find that the two attention signals capture complementary properties of visual information. Heads with higher $\alpha_h$ capture broader visual information, whereas those with higher $\rho_h$ more precisely localize visual evidence, as reflected by their final accuracy. Combining their attention distributions therefore provides a more comprehensive and precise importance signal for visual token pruning. 
To select the teacher layer, we consider the average performance across datasets to reduce the influence of data-specific variations. Within the selected layer, we further identify retrieval heads based on two complementary attention patterns rather than directly ranking individual heads by their inference accuracy. Specifically, we select one head with the highest $\alpha_h$ and another head with the highest $\rho_h$ among the top 25\% of heads ranked by $\alpha_h$. This restriction ensures that the latter head also exhibits substantial attention to visual tokens, while providing a complementary attention pattern to the former. Their combined attention distributions serve as the teacher signal for training the visual token scorer. \textbf{This provides a principled approach to identifying complementary retrieval heads based on their attention patterns.}
\subsection{QRAD Training and Inference}
\label{sec:qrad}

Based on the selected layer and retrieval heads, we propose QRAD to distill their attention distributions into a lightweight scorer. The scorer learns to predict visual token importance before LLM runs. It offers two benefits: (1) providing direct supervision from the LLM's task-relevant attention patterns, and (2) introducing negligible inference overhead. Figure~\ref{fig:framework} illustrates the overall framework.

\textbf{Teacher signal.}
Let $\mathcal{H}$ denote the retrieval heads selected in Section~\ref{sec:heads}, and let $l^*$ denote the selected deep layer. Given visual tokens $N_v$ and the last prompt token $r_{\mathrm{last}}$, we extract the attention scores $A_{l^*,h}$ for each $h\in\mathcal{H}$. We first average the attention scores across the selected heads and then normalize them to obtain the teacher distribution:
\begin{equation}
\bar A_j=\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}A_{l^*,h}[r_{\mathrm{last}},j],
\qquad
t_j=\frac{\bar A_j}{\sum\limits_{j'\in N_v}\bar A_{j'}}.
\label{eq:teacher}
\end{equation}
The resulting distribution ${\mathbf{t}}=(t_1,\ldots,t_{N_v})$ serves as the supervision target for training the visual token scorer.

\textbf{Scorer.}
We introduce a lightweight cross-attention scorer $g_\phi$ to estimate visual token importance. Given visual features $H_v\in\mathbb{R}^{N_v\times D}$ and question embeddings $H_q\in\mathbb{R}^{N_q\times D}$, the scorer outputs an importance score for each visual token. We first project the visual and question embeddings into a shared low-dimensional space:
\begin{equation}
\tilde{H}_v=H_v P_v+b_v,\qquad
\tilde{H}_q=H_q P_q+b_q,
\label{eq:input_projection}
\end{equation}
where $P_v,P_q\in\mathbb{R}^{D\times d}$ and $b_v,b_q\in\mathbb{R}^{d}$.
The projected visual and text features $\tilde{H}_v,\tilde{H}_q$ are then fed into a cross-attention block. The block consists of a cross-attention layer followed by a feed-forward network. Each operation uses pre-LayerNorm (LN) and a residual connection:
\begin{equation}
\begin{aligned}
Q &= W_Q\,\mathrm{LN}(\tilde{H}_v),
K = W_K\,\mathrm{LN}(\tilde{H}_q),
V = W_V\,\mathrm{LN}(\tilde{H}_q), \\
O &= \operatorname{MHA}(Q,K,V), \\
X &= \tilde{H}_v + O
+\operatorname{FFN}\left(\mathrm{LN}(\tilde{H}_v + O)\right), \\
\end{aligned}
\end{equation}
where $\operatorname{MHA}$ denotes multi-head cross-attention, $\operatorname{FFN}$ is a two-layer MLP with a GELU nonlinearity between the two linear layers, and $X\in\mathbb{R}^{N_v\times d}$ is the output. Finally, we apply an FFN and a Softmax operation to map each visual token independently to a student distribution:
\begin{equation}
\mathbf{s}=\operatorname{Softmax}\big(W_2\operatorname{GELU}
\left(
W_1\operatorname{LN}(X)+b_1
\right)+b_2\big),
\label{eq:token_scoring}
\end{equation}
where $W_1\in\mathbb{R}^{d\times d_1}$ and $W_2\in\mathbb{R}^{d_1\times 1}$. This produces a student distribution $\mathbf{s}\in\mathbb{R}^{N_v}$, with each entry corresponding to one visual token.

\textbf{Training.}
We freeze the vision encoder and the LLM and train only the scorer. One iteration consists of three steps:
\begin{enumerate}
  \item \emph{Teacher forward.} Compute the teacher distribution ${\mathbf{t}}$ from the retrieval heads in the selected deep layer.
  \item \emph{Scorer forward.} Compute the student distribution $\mathbf{s}=g_\phi(H_v,H_q)$ using the lightweight scorer.
  \item \emph{Loss.} Let $\pi=(\pi_1,\ldots,\pi_{N_v})$ denote the visual token indices sorted by the teacher scores ${\mathbf{t}}$ in descending order. We combine a distribution-matching term with a interceptive listwise ranking term:
  \begin{align}
  \mathcal{L}&=\mathrm{KL}({\mathbf{s}}\,\|\,\mathbf{t})
  +\beta\mathcal{L}_{\text{rank}}, \\
  \mathcal{L}_{\text{rank}}&=-\frac{1}{m}\sum_{i=1}^{m}
  \log\frac{s_{\pi_i}}{\sum_{j=i}^{N_v}s_{\pi_j}}.
  \label{eq:loss}
  \end{align}
\end{enumerate}
The KL term provides a dense signal by matching the full teacher distribution. For token pruning, however, we are less concerned with tokens with low scores. We therefore introduce an interceptive ListMLE loss $\mathcal{L}_{\text{rank}}$ over the top-$m$ teacher-ranked tokens ($m=10\%$), encouraging the scorer to prioritize important visual tokens.

\textbf{Inference.}
Given $H_v$ and $H_q$, the scorer $g_\phi$ produces $\mathbf{s}$; we retain the top-$k$ visual tokens, where $k=\left\lfloor \eta N_v\right\rfloor,$ and $\eta$ is the retention rate. Only the kept tokens are passed to the LLM for answer generation. Thus, QRAD introduces only one lightweight forward pass before LLM inference.

**Visual Style Requirements:**
1.  **Style:** Flat vector illustration, clean lines, academic aesthetic. Similar to figures in DeepMind or OpenAI papers.
2.  **Layout:** Organized flow (Left-to-Right, Top-to-Bottom, Circular and other shapes). Group related components logically.
3.  **Color Palette:** Professional pastel tones. White background.
4.  **Text Rendering:** You MUST include legible text labels for key modules or equations mentioned in the methodology (e.g., "Encoder", "Loss", "Transformer").
5.  **Negative Constraints:** NO photorealistic photos, NO messy sketches, NO unreadable text, NO 3D shading artifacts.
**Generation Instruction:**
Highlight the core novelty. Ensure the connection logic makes sense.
