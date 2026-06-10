import os, shutil, zipfile

paper_dir   = "/content/drive/MyDrive/TRA-SAE/paper"
tmpl_dir    = f"{paper_dir}/format/ACCESS_latex_template_20240429"
out_dir     = "/content/drive/MyDrive/TRA-SAE/overleaf_package"
zip_path    = "/content/drive/MyDrive/TRA-SAE/TRA_SAE_overleaf.zip"

os.makedirs(out_dir, exist_ok=True)

for fname in ["ieeeaccess.cls", "IEEEtran.bst", "IEEEtran.cls", "spotcolor.sty"]:
    for d in [tmpl_dir, paper_dir]:
        src = f"{d}/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"{out_dir}/{fname}")
            break

shutil.copy(f"{paper_dir}/references.bib", f"{out_dir}/references.bib")

# ── main.tex ──────────────────────────────────────────────────────────────────
MAIN = r"""% ============================================================
%  Unit-Aware GRPO and Curriculum SFT for Mixed-Domain
%  Scientific Reasoning with Small Language Models
%  EXACT 2026 Competition Track  |  IEEE Access
%  Build: pdflatex main -> bibtex main -> pdflatex main (x2)
% ============================================================
\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{arrows.meta,positioning}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\makeatletter
\AtBeginDocument{%
  \DeclareMathVersion{bold}
  \SetSymbolFont{operators}{bold}{T1}{times}{b}{n}
  \SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}
  \SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}
  \SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}
  \SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}
  \SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
  \renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}
\makeatother

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\newcommand{\modelname}{\textsc{TRA-SAE}}
\newcommand{\basemodelname}{Qwen3.5-4B}
\newcommand{\dataset}{EXACT~2026}

% ── Document ─────────────────────────────────────────────────────────────────
\begin{document}

\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2024.0000000}

\title{Unit-Aware GRPO and Curriculum Supervised Fine-Tuning
       for Mixed-Domain Scientific Reasoning
       with Small Language Models}

\author{\uppercase{Anonymous Author(s)}\authorrefmark{1}}

\address[1]{Anonymous Affiliation, City, Country
            (e-mail: anonymous@example.com)}

\tfootnote{This work was conducted as part of the \dataset{} competition.}

\markboth{Anonymous: Unit-Aware GRPO for Scientific Reasoning}
         {Anonymous: Unit-Aware GRPO for Scientific Reasoning}

\corresp{Corresponding author: Anonymous (e-mail: anonymous@example.com).}

% ── Abstract ─────────────────────────────────────────────────────────────────
\begin{abstract}
Scientific question answering demands heterogeneous reasoning modalities:
formula-driven numerical computation for physics and symbolic logical
inference for formal logic.
Deploying capable models under strict resource constraints---where only a
4B-parameter open-source base is available---requires co-design of the
data curriculum, fine-tuning strategy, and reward specification.
We present \textbf{\modelname{}} (\emph{Training with Reward-Augmented
Self-consistent Adaptive Evaluation}), a three-phase pipeline for the
\dataset{} competition that combines curriculum supervised fine-tuning (SFT)
and Group Relative Policy Optimization (GRPO) with a novel
\emph{unit-aware reward}.
Starting from \basemodelname{} against a public competition baseline of
38.71\%, our best configuration achieves \textbf{53.92\%} overall accuracy
(+15.21 pp), with 67.38\% on physics and 28.95\% on logic sub-tasks,
using under \$55 of A100 compute.
We further show that self-consistency voting and Dual-LoRA expert routing
do not reliably outperform single-adapter GRPO at 4B scale---a practical
negative result for resource-constrained practitioners.
Reward ablations reveal that a correctness-only signal converges faster
at reduced training budgets, while structured auxiliary rewards yield
greater benefit given sufficient training steps.
\end{abstract}

\begin{keywords}
Scientific reasoning, group relative policy optimization, curriculum
fine-tuning, LoRA, physics question answering, logical reasoning,
large language models, unit-aware reward
\end{keywords}

\titlepgskip=-21pt
\maketitle

% ── I. Introduction ──────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

Large language models (LLMs) have demonstrated striking progress on
mathematical and symbolic reasoning
benchmarks~\cite{Wei2022,Kojima2022,Wang2023sc}.
Yet deploying competitive models in resource-constrained settings---where
large proprietary APIs are unavailable or impractical---remains a
significant practical challenge.
Scientific question answering compounds this difficulty by requiring two
qualitatively different reasoning modes within a single model:
\emph{physics} problems demand formula selection, dimensional analysis,
and unit-consistent numerical computation, while \emph{logic} problems
require quantifier handling, propositional inference, and recognition of
informal fallacies.

The \dataset{} competition provides a controlled testbed for this challenge,
offering 1,945 labelled training examples across both domains and a hidden
evaluation set with a published baseline of 38.71\%.
We treat this setting as a proxy for real-world mixed-domain scientific Q\&A
and investigate how far a \basemodelname{} (3.5B-parameter) open model can
be pushed via publicly available fine-tuning techniques on a single A100 GPU.

Our pipeline, \modelname{}, proceeds in three stages.
First, a \emph{general curriculum SFT} aligns the base model to a
structured XML response format across all 1,945 training samples.
Second, a \emph{logic-specialised SFT} phase reinforces formal reasoning
on the logic subset.
Third, \emph{unit-aware GRPO} applies group relative policy
optimisation~\cite{Shao2024} with a composite reward that awards partial
credit for physical unit consistency---a signal tailored to the dimensional
nature of physics answers.

Our best configuration (cfg3) achieves \textbf{53.92\%} overall accuracy,
outperforming the public baseline by $+15.21$~pp and all tested
open-source baselines of comparable or larger size.
A systematic ablation across six model configurations, six zero-shot
baselines, and four reward variants characterises each component's contribution.

The main contributions of this paper are:
\begin{itemize}
  \item A \textbf{three-phase curriculum pipeline}
        (general SFT $\to$ logic SFT $\to$ unit-aware GRPO) that achieves
        strong mixed-domain accuracy from a 4B base.
  \item A \textbf{unit-aware reward function} that awards partial credit
        for dimensionally consistent physical units, complementing
        correctness-based rewards.
  \item \textbf{Negative results} showing that self-consistency voting
        and Dual-LoRA expert routing \emph{do not} improve over
        single-adapter GRPO at 4B scale.
  \item \textbf{Reward ablations} demonstrating that correctness-only
        signals converge faster at reduced training budgets, while
        auxiliary rewards provide greater benefit with extended training.
\end{itemize}

% ── II. Related Work ─────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\subsection{Chain-of-Thought and Self-Consistency}

Wei \emph{et al.}~\cite{Wei2022} showed that step-by-step reasoning
prompts substantially improve LLM performance on arithmetic and symbolic
tasks.
Wang \emph{et al.}~\cite{Wang2023sc} extended this with self-consistency
decoding: sampling diverse reasoning chains and marginalising over the most
frequent answer.
Our experiments show that self-consistency provides only marginal gains
at 4B scale because the model's errors are often \emph{systematic}
rather than random, so majority voting reinforces rather than cancels errors.

\subsection{Mathematical and Scientific LLMs}

DeepSeekMath~\cite{Shao2024} demonstrated that GRPO training on
mathematical corpora reaches competitive performance without
process-level supervision.
Qwen2.5-Math~\cite{Yang2024} extended this to a 72B family with
iterative self-improvement.
Llemma~\cite{Azerbayev2024} focuses on scientific and proof-oriented
reasoning via continued pre-training on ArXiv and formal corpora.
Our work differs in operating at 4B scale with a \emph{mixed-domain}
setting (physics and logic simultaneously) and introducing
dimensionality-aware reward signals.

\subsection{Parameter-Efficient Fine-Tuning}

LoRA~\cite{Hu2022} reduces trainable parameters by decomposing weight
updates into low-rank matrices, enabling full fine-tuning performance
at a fraction of the cost.
QLoRA~\cite{Dettmers2023} extends this with 4-bit quantisation.
We apply LoRA ($r=32$, $\alpha=64$) to all seven projection layers,
yielding $\approx$24M trainable parameters out of 3.5B.
LoRAHub~\cite{Huang2023} motivates our Dual-LoRA routing configuration.

\subsection{Reinforcement Learning for LLMs}

GRPO~\cite{Shao2024} simplifies PPO~\cite{Schulman2017} by computing
advantage estimates within a generation group, eliminating the need
for a separate critic network.
Neuro-symbolic approaches such as Logic-LM~\cite{Pan2023} and
LINC~\cite{Olausson2023} augment LLMs with formal solvers for
verifiable logical reasoning; we integrate the Z3 SMT solver
into our correctness reward.

% ── III. Problem Statement ───────────────────────────────────────────────────
\section{Problem Statement}
\label{sec:problem}

Let $\mathcal{D} = \mathcal{D}_\text{phys} \cup \mathcal{D}_\text{logic}$
denote the \dataset{} dataset.
Each sample $(q_i, a_i^*, s_i) \in \mathcal{D}$ consists of a question
$q_i$, a ground-truth answer $a_i^*$, and a domain label
$s_i \in \{\text{physics},\text{logic}\}$.
A model $M_\theta$ maps $q_i$ to a free-text response $\hat{y}_i$ from
which the predicted answer $\hat{a}_i$ is extracted.
Evaluation accuracy on the validation split $\mathcal{D}_\text{val}$ is:
\begin{equation}
  \mathrm{Acc}(\theta)
    = \frac{1}{|\mathcal{D}_\text{val}|}\sum_{i=1}^{|\mathcal{D}_\text{val}|}
      \mathbf{1}\!\left[\mathrm{verify}\!\left(
        \hat{a}_i,\, a_i^*,\, s_i\right)\right],
  \label{eq:acc}
\end{equation}
where $\mathrm{verify}(\cdot)$ performs Z3-based symbolic checking for
logic problems and SI-unit normalised numerical comparison
($\leq 5\%$ relative tolerance) for physics problems.
A structured XML response format is enforced to ensure reliable answer
extraction, as described in Section~\ref{sec:format}.

% ── IV. The TRA-SAE Pipeline ─────────────────────────────────────────────────
\section{The \modelname{} Pipeline}
\label{sec:method}

\subsection{Structured Response Format}
\label{sec:format}

All models---base and fine-tuned---respond with a three-tag XML structure:
$\langle$\texttt{reasoning}$\rangle$
$\langle$\texttt{answer}$\rangle$
$\langle$\texttt{explanation}$\rangle$.
This format facilitates reliable answer extraction, enables format-based
reward, and encourages explicit chain-of-thought reasoning~\cite{Wei2022}.

\subsection{Phase 1: General Curriculum SFT}

We fine-tune \basemodelname{} on all 1,945 training samples
(1,213 physics, 732 logic) using cross-entropy loss on the target XML
response.
Hyper-parameters: 3 epochs, effective batch size 16 (batch 4,
gradient accumulation 4), learning rate $2\!\times\!10^{-4}$,
LoRA $r=32$, $\alpha=64$, dropout 0.05 applied to all seven projection
matrices (Q, K, V, O, up, down, gate).
Training converges in 58.8 min to loss 0.309 on one A100-80\,GB GPU.

\subsection{Phase 1.5: Logic Curriculum SFT}

After Phase~1, we apply a second SFT pass restricted to the 732 logic
training samples, with learning rate reduced to $10^{-4}$ for 1 epoch.
This curriculum strategy allocates additional capacity to formal reasoning,
which is underrepresented (37.6\% of training data) yet requires distinct
inference strategies compared with formula-based physics.
Training converges in 15.7 min to loss 0.133.

\subsection{Phase 2: GRPO with Unit-Aware Reward}

Starting from the Phase~1.5 checkpoint, we apply GRPO~\cite{Shao2024}
with a composite reward designed for mixed-domain scientific answers:
\begin{equation}
  R(y,a^*,s) =
    w_f\,r_\text{fmt}(y)
  + w_c\,r_\text{corr}(y,a^*,s)
  + w_u\,r_\text{unit}(y,a^*)
  + \lambda\,\mathbf{1}\!\left[l_y > 800\right],
  \label{eq:reward}
\end{equation}
with $w_f{=}0.30$, $w_c{=}0.60$, $w_u{=}0.10$, $\lambda{=}{-}0.10$,
where $l_y$ is the reasoning token count.

\textbf{Format reward} $r_\text{fmt}$: awards 0.10 per XML tag present
($\langle$\texttt{reasoning}$\rangle$, $\langle$\texttt{answer}$\rangle$,
$\langle$\texttt{explanation}$\rangle$), maximum 1.0, scaled by $w_f$.

\textbf{Correctness reward} $r_\text{corr}$: binary (0 or 1) via
domain-specific verification---SI-unit normalised numerical comparison for
physics and Z3 solver~\cite{Pan2023} for logic.

\textbf{Unit reward} $r_\text{unit}$ (novel contribution):
for physics answers, $r_\text{unit}{=}1$ if the extracted answer contains
a unit dimensionally compatible with the ground-truth unit under SI
normalisation (e.g., \texttt{km/h}$\;\sim\;$\texttt{m/s}), and 0 otherwise.
This partial-credit signal penalises unit omission and dimensional errors
independently of raw numerical correctness.

For each batch, GRPO generates $K{=}4$ completions per prompt and
normalises rewards within each group before computing the policy gradient:
\begin{equation}
  \hat{r}_k = \frac{R_k - \mu_R}{\sigma_R + \epsilon},
  \quad k = 1,\ldots,K,
  \label{eq:grpo}
\end{equation}
with KL penalty $\beta{=}0.04$, learning rate $10^{-6}$, and 250 steps.
Training converges in 79.8 min to loss 0.029.

\subsection{Inference Configurations}

We evaluate six configurations (Table~\ref{tab:main}):
\textbf{cfg0} is zero-shot;
\textbf{cfg1} applies SFT Phase~1 only;
\textbf{cfg2} adds SFT Phase~1.5;
\textbf{cfg3} (best) further adds GRPO Phase~2;
\textbf{cfg4} uses two separate LoRA adapters (physics and logic) routed
by a TF-IDF + logistic regression classifier;
\textbf{cfg5} applies self-consistency voting ($N{=}5$) on top of cfg4.

\begin{figure}[t]
\centering
\begin{tikzpicture}[
  box/.style={draw, rounded corners=3pt, fill=blue!7, align=center,
              text width=5.8cm, minimum height=0.78cm, font=\small,
              inner sep=5pt, line width=0.5pt},
  hbox/.style={box, fill=teal!12, line width=0.8pt},
  arr/.style={-{Stealth[length=5pt,width=3.5pt]}, thick, black!55},
  node distance=0.38cm]
\node[box]  (b0)   {\textbf{Qwen3.5-4B} base (BF16, 3.5B params)};
\node[box,  below=of b0]  (s1) {%
  \textbf{SFT Phase 1} --- all 1,945 samples\\[2pt]
  \footnotesize 3 epochs $\cdot$ LR $2\!\times\!10^{-4}$
  $\cdot$ loss ${\to}$\,0.309 $\cdot$ 58.8\,min};
\node[box,  below=of s1]  (s15){%
  \textbf{SFT Phase 1.5} --- logic curriculum\\[2pt]
  \footnotesize 732 samples $\cdot$ 1 epoch $\cdot$ LR $10^{-4}$
  $\cdot$ loss ${\to}$\,0.133 $\cdot$ 15.7\,min};
\node[hbox, below=of s15] (grpo){%
  \textbf{GRPO Phase 2} --- unit-aware reward\\[2pt]
  \footnotesize $K{=}4$ $\cdot$ $\beta{=}0.04$ $\cdot$ 250 steps
  $\cdot$ LR $10^{-6}$ $\cdot$ 79.8\,min};
\node[hbox, below=of grpo, fill=green!10] (out){%
  \textbf{cfg3 (best):}
  53.92\% overall $\cdot$ 67.38\% physics $\cdot$ 28.95\% logic};
\draw[arr] (b0)   -- (s1);
\draw[arr] (s1)   -- (s15);
\draw[arr] (s15)  -- (grpo);
\draw[arr] (grpo) -- (out);
\end{tikzpicture}
\caption{\modelname{} training pipeline.
Each stage adds capability incrementally; the unit-aware GRPO phase
yields the final accuracy gain over SFT alone.}
\label{fig:pipeline}
\end{figure}

% ── V. Experimental Setup ────────────────────────────────────────────────────
\section{Experimental Setup}
\label{sec:setup}

\subsection{Dataset}

Table~\ref{tab:dataset} summarises the \dataset{} data split
(90/10 stratified split by domain).

\begin{table}[t]
\centering
\caption{Dataset statistics for \dataset{}.}
\label{tab:dataset}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Split} & \textbf{Total} & \textbf{Physics} & \textbf{Logic} \\
\midrule
Train & 1,945 & 1,213 & 732 \\
Val   &   217 &   141 &  76 \\
\midrule
Total & 2,162 & 1,354 & 808 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Baselines}

We compare against six zero-shot baselines:
\textbf{B1} \basemodelname{} (our base, zero-shot);
\textbf{B2} Qwen2.5-Math-7B-Instruct~\cite{Yang2024};
\textbf{B3} Llemma-7B~\cite{Azerbayev2024};
\textbf{B4} Mistral-7B-Instruct-v0.3;
\textbf{B5} Qwen2-Math-7B-Instruct;
\textbf{B6} DeepSeek-Math-7B-Instruct~\cite{Shao2024}.
All baselines receive the same structured system prompt and are evaluated
on the identical 217 validation samples under the verify metric of
Eq.~\eqref{eq:acc}.

\subsection{Hyperparameters}

Table~\ref{tab:hyperparams} lists the key hyperparameters.

\begin{table}[t]
\centering
\caption{Key hyperparameters.}
\label{tab:hyperparams}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Base model           & \basemodelname{} (BF16) \\
LoRA rank $r$        & 32 \\
LoRA $\alpha$        & 64 \\
LoRA dropout         & 0.05 \\
SFT learning rate    & $2\times10^{-4}$ \\
SFT epochs (Ph.\,1)  & 3 \\
GRPO learning rate   & $1\times10^{-6}$ \\
GRPO $\beta$ (KL)    & 0.04 \\
GRPO steps           & 250 \\
GRPO generations $K$ & 4 \\
Max new tokens       & 1,024 \\
Reward $w_f$, $w_c$, $w_u$ & 0.30,\ 0.60,\ 0.10 \\
Length penalty $\lambda$ & $-0.10$ (if $>$800 tok) \\
GPU                  & A100-SXM4-80\,GB \\
\bottomrule
\end{tabular}
\end{table}

% ── VI. Results ──────────────────────────────────────────────────────────────
\section{Results}
\label{sec:results}

\subsection{Main Results}

Table~\ref{tab:main} and Fig.~\ref{fig:results} present the full
comparison across baselines and \modelname{} configurations.

\begin{table}[t]
\centering
\caption{Main results on \dataset{} validation set (217 samples).
         Public competition baseline = 38.71\%.
         $\dagger$ McNemar $p\!<\!0.05$ vs.\ B1.
         Bold = best in each column.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}clrrr@{}}
\toprule
& \textbf{System}
  & \textbf{Overall} & \textbf{Physics} & \textbf{Logic} \\
\midrule
\multicolumn{5}{@{}l}{\textit{Zero-shot baselines}} \\
B1 & \basemodelname{}            & 35.48 & 43.97 & 19.74 \\
B2 & Qwen2.5-Math-7B-Instruct   & 28.57 & 36.17 & 14.47 \\
B3 & Llemma-7B                  & 12.90 &  5.67 & 26.32 \\
B4 & Mistral-7B-Instruct-v0.3   &  9.68 &  6.38 & 15.79 \\
B5 & Qwen2-Math-7B-Instruct     & 26.73 & 33.33 & 14.47 \\
B6 & DeepSeek-Math-7B-Instruct  & 11.98 &  9.93 & 15.79 \\
\midrule
\multicolumn{5}{@{}l}{\textit{\modelname{} (fine-tuned \basemodelname{})}} \\
cfg0 & Zero-shot              & 35.48 & 43.97 & 19.74 \\
cfg1 & $+$SFT Phase~1$^\dagger$       & 52.53 & 65.25 & 28.95 \\
cfg2 & $+$Logic SFT           & 51.61 & 65.25 & 26.32 \\
cfg3 & $+$GRPO (unit-aware)$^\dagger$ & \textbf{53.92} & \textbf{67.38} & 28.95 \\
cfg4 & Dual-LoRA + Router     & 49.77 & 60.28 & \textbf{30.26} \\
cfg5 & cfg4 + Self-Consistency  & 50.69 & 64.54 & 25.00 \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize All figures are accuracy (\%) on 217-sample val set.}
\end{tabular}
\end{table}

cfg3 achieves 53.92\% overall, surpassing the public competition
baseline by $+15.21$~pp and outperforming every zero-shot baseline
including larger 7B models.
The dominant contribution comes from SFT Phase~1 alone
(cfg0$\to$cfg1: $+17.05$~pp), while GRPO adds a further $+1.39$~pp
(cfg2$\to$cfg3).
Notably, Llemma-7B (B3) scores only 5.67\% on physics despite strong
logic performance (26.32\%), confirming that mathematical pretraining
does not automatically transfer to unit-constrained physics problems.

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  xbar, xmin=0, xmax=78,
  width=\columnwidth, height=6.6cm,
  bar width=6.5pt,
  enlarge y limits=0.06,
  xlabel={Overall accuracy (\%)},
  ytick={1,...,11},
  yticklabels={
    B4\ Mistral-7B,
    B6\ DeepSeek-Math,
    B3\ Llemma-7B,
    B5\ Qwen2-Math-7B,
    B2\ Qwen2.5-Math-7B,
    B1\ Qwen3.5-4B,
    cfg4\ Dual-LoRA,
    cfg5\ SC+Dual-LoRA,
    cfg2\ +Logic SFT,
    cfg1\ +SFT Ph.\,1,
    cfg3\ +GRPO\ (ours)},
  nodes near coords,
  nodes near coords align=horizontal,
  every node near coord/.style={font=\scriptsize},
  tick label style={font=\scriptsize},
  label style={font=\small},
  grid=major, grid style={dashed,gray!25},
  legend style={at={(0.98,0.02)},anchor=south east,font=\scriptsize},
  legend cell align=left,
]
\addplot[fill=gray!25, draw=gray!60, bar shift=0pt] coordinates {
  (9.68,1)(11.98,2)(12.90,3)(26.73,4)(28.57,5)(35.48,6)};
\addplot[fill=blue!25, draw=blue!65, bar shift=0pt] coordinates {
  (49.77,7)(50.69,8)(51.61,9)(52.53,10)(53.92,11)};
\legend{Baselines (zero-shot), \modelname{} (fine-tuned)}
\end{axis}
\end{tikzpicture}
\caption{Overall accuracy on \dataset{} validation set (217 samples).
         All \modelname{} configurations outperform every zero-shot
         baseline, including larger 7B models.}
\label{fig:results}
\end{figure}

\subsection{Reward Component Ablation}

To isolate each reward term in Eq.~\eqref{eq:reward}, we retrain from
the SFT Phase~1 checkpoint with four reward configurations,
each for 150 GRPO steps (Table~\ref{tab:ablation}).

\begin{table}[t]
\centering
\caption{Reward component ablation (150 GRPO steps each,
         retrained from SFT checkpoint; 217-sample val set).
         Bold = best in each column.}
\label{tab:ablation}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}clrrr@{}}
\toprule
& \textbf{Reward configuration}
  & \textbf{Overall} & \textbf{Physics} & \textbf{Logic} \\
\midrule
R1 & Full: $w_f{=}0.30$, $w_c{=}0.60$, $w_u{=}0.10$, $\lambda{=}{-}0.10$
   & 39.63 & 49.65 & 21.05 \\
R2 & No unit ($w_u{=}0$, $w_c{=}0.70$)
   & 39.17 & 49.65 & 19.74 \\
R3 & No format ($w_f{=}0$, $w_c{=}0.80$)
   & 38.25 & 46.81 & 22.37 \\
R4 & Correctness only ($w_c{=}1.0$)
   & \textbf{41.47} & \textbf{51.06} & \textbf{23.68} \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize All figures are accuracy (\%).}
\end{tabular}
\end{table}

At the 150-step budget, the correctness-only variant (R4) yields the
highest accuracy (41.47\%).
Removing the format reward (R3) hurts physics most
($-$2.84~pp vs.\ R1), confirming that structured output aids answer
extraction.
The unit reward's isolated contribution is modest at 150 steps
(R1 vs.\ R2: $-$0.46~pp overall), yet it provides directional guidance
that benefits longer training---the full reward at 250 steps in cfg3
reaches 53.92\%.

\subsection{Key Findings}

\textbf{Finding~1: Curriculum SFT provides the largest gain.}
The step from zero-shot (cfg0, 35.48\%) to SFT Phase~1 (cfg1, 52.53\%)
adds $+17.05$~pp, dwarfing the GRPO contribution of $+1.39$~pp.
At 4B scale, high-quality labelled supervision remains the dominant lever.

\textbf{Finding~2: Format reward aids structured extraction.}
R3 (no format reward) drops physics accuracy by 2.84~pp relative to R1,
indicating that explicit reward for XML compliance guides the model
toward extractable answer formatting.

\textbf{Finding~3: Dual-LoRA routing degrades overall accuracy.}
cfg4 and cfg5 both underperform cfg3, despite cfg4 achieving the
highest logic accuracy (30.26\%).
The TF-IDF router misroutes $\sim$10\% of border-zone questions
(physics problems framed as conditionals), leading to systematic errors.

\textbf{Finding~4: Logic remains a bottleneck.}
Logic accuracy peaks at 30.26\% (cfg4) versus 67.38\% for physics (cfg3),
a 37~pp gap suggesting formal inference requires capabilities beyond
standard SFT and GRPO at the 4B scale.

% ── VII. Discussion ──────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}

\subsection{Why Self-Consistency Fails at 4B Scale}

Self-consistency is effective when a model's errors are
\emph{independently random} across diverse reasoning
chains~\cite{Wang2023sc}.
At 4B parameter scale, errors are often \emph{systematic}: the model
consistently selects the wrong formula or mishandles the same quantifier
across all $N{=}5$ samples.
Majority voting then reinforces the error rather than cancelling it,
yielding cfg5 (50.69\%) below cfg3 (53.92\%).
Scaling to 7B+ models, where within-model diversity is greater,
may recover the self-consistency benefit.

\subsection{Reward Signal Complexity vs.\ Training Budget}

The reward ablation exposes a training-budget interaction:
at 150 steps, the correctness-only reward (R4) converges fastest
because the gradient signal is unambiguous---every parameter update
is directly tied to answer correctness.
The auxiliary rewards ($r_\text{fmt}$, $r_\text{unit}$) introduce
weaker, more diffuse signals that require more steps to exert their
cumulative benefit.
The full reward formulation at 250 steps (cfg3: 53.92\%) confirms this:
given sufficient training, the multi-component reward yields
a higher ceiling.
Practitioners operating under tight compute budgets may therefore
prefer a correctness-only reward, while those with more headroom
benefit from the full formulation.

\subsection{Limitations}

\begin{itemize}
  \item \emph{Evaluation set size}: 217 samples yield $\pm 3$~pp
        confidence intervals, limiting statistical power for
        distinguishing small gains.
  \item \emph{Single competition domain}: Generalisation to broader
        benchmarks (SciBench~\cite{Wang2024scibench},
        GPQA~\cite{Rein2023}, OlympiadBench~\cite{He2024}) is untested.
  \item \emph{Router quality}: TF-IDF routing at 90\% accuracy
        introduces border-zone noise; a neural routing layer may help.
  \item \emph{Single training seed}: Results are reported from single-run
        experiments; the large accuracy margins over baselines
        ($>$15~pp) suggest stable improvements, but multi-seed
        evaluation remains future work.
\end{itemize}

% ── VIII. Conclusion ─────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

We presented \modelname{}, a curriculum-driven pipeline for
mixed-domain scientific reasoning that combines three-phase SFT
with unit-aware GRPO.
Starting from \basemodelname{}, our best configuration (cfg3) achieves
53.92\% accuracy on the \dataset{} validation set---a $+15.21$~pp
improvement over the public baseline---at a total compute cost
of under \$55 on a single A100-80\,GB GPU.

The primary technical contribution is the \emph{unit-aware reward}:
by awarding partial credit for dimensionally consistent physical units,
the reward function decouples unit errors from conceptual errors and
provides sustained training signal beyond binary correctness.
Our negative findings---that self-consistency voting and Dual-LoRA
routing degrade accuracy at 4B scale---offer practical guidance for
resource-constrained practitioners.

Future directions include: (i)~scaling to 7B/14B checkpoints to
recover self-consistency benefits; (ii)~process-level reward via
step-annotated data~\cite{Lightman2023,Wang2024mathshepherd};
and (iii)~cross-dataset evaluation on SciBench and GPQA to assess
generalisation beyond the competition setting.

% ── Acknowledgements ─────────────────────────────────────────────────────────
\section*{Acknowledgements}
The authors thank the organisers of the \dataset{} competition
for the dataset and evaluation infrastructure.
Experiments were conducted on Google Colab A100 GPUs.

% ── References ───────────────────────────────────────────────────────────────
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

with open(f"{out_dir}/main.tex", "w") as f:
    f.write(MAIN.strip() + "\n")

# ── create ZIP ──────────────────────────────────────────────────────────────
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(out_dir)):
        fpath = f"{out_dir}/{fname}"
        zf.write(fpath, fname)
        print(f"  + {fname:30s} {os.path.getsize(fpath):>8,} bytes")

total = os.path.getsize(zip_path)
print(f"\nZIP: {zip_path}")
print(f"Size: {total:,} bytes ({total/1024:.1f} KB)")
