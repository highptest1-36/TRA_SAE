#!/usr/bin/env python3
import re, sys

BASE = "/content/drive/MyDrive/TRA-SAE/paper/format/WileyDesign/Optimal-Design-layout"
TEX = f"{BASE}/TRA_SAE_Wiley.tex"
BIB = f"{BASE}/references.bib"
CLS = f"{BASE}/USG.cls"
PAPER_BIB = "/content/drive/MyDrive/TRA-SAE/paper/references.bib"

t = open(TEX).read()

def rep(old, new, label, n=1):
    c = t.count(old)
    assert c == n, f"[{label}] expected {n} match(es), found {c}"
    return t.replace(old, new, n)

# ---- 1. documentclass: ASNA -> APA ----
t = rep(r"\documentclass[ASNA,twocolumn]{USG}",
        r"\documentclass[APA,twocolumn]{USG}", "documentclass")

# ---- 2. history dates ----
t = rep(
r"""\received{00 Month 2026}
\revised{00 Month 2026}
\accepted{00 Month 2026}""",
r"""%% History dates are inserted by the publisher at production.
%\received{}
%\revised{}
%\accepted{}""", "history-dates")

# ---- 3. preamble: alias citep/citet + place before \begin{document} ----
t = rep(
r"""\articledoi{}

\begin{document}""",
r"""\articledoi{}

%% APA (apacite) does not provide natbib-style \citep/\citet; alias them when
%% absent so the same source compiles under the APA style and the Chicago fallback.
\makeatletter
\@ifundefined{citep}{\let\citep\cite}{}
\@ifundefined{citet}{\let\citet\citeA}{}
\makeatother

\begin{document}""", "preamble-alias")

# ---- 4. title ----
t = rep(
r"""\title{\modelname{}: A Reproducible Low-Cost Curriculum and Reward-Shaping
       Study for Mixed-Domain Scientific Reasoning with a 4B Language Model}""",
r"""\title{\modelname{}: Low-Cost Retrieval-Augmented Fine-Tuning for
       Mixed-Domain Scientific Reasoning with a 4B Language Model}""", "title")

# ---- 5. running title (<40 chars) ----
t = rep(r"\titlemark{Low-Cost Curriculum and Reward Shaping for Scientific Reasoning}",
        r"\titlemark{Retrieval-Augmented 4B Reasoning}", "titlemark")

# ---- 6. keywords ----
t = rep(
r"""\keywords{large language models | scientific question answering |
group relative policy optimization | curriculum fine-tuning |
parameter-efficient fine-tuning | reproducibility}""",
r"""\keywords{large language models | scientific question answering |
retrieval-augmented generation | group relative policy optimization |
curriculum fine-tuning | parameter-efficient fine-tuning |
explainable educational question answering}""", "keywords")

# ---- 7. abstract ----
abs_old = r"""\abstract[ABSTRACT]{Scientific question answering couples numerical physical
reasoning with symbolic logical inference, and resource-constrained settings
frequently preclude large proprietary models or extensive reinforcement
learning. This paper reports \modelname{}, a low-cost curriculum training
pipeline for the \dataset{} mixed-domain scientific question answering task
built on a 4B-parameter open model. The pipeline combines general supervised
fine-tuning (SFT), a logic-focused curriculum adaptation stage, and Group
Relative Policy Optimization (GRPO) with lightweight format, correctness, and
unit-consistency reward terms; total training cost is US\$18 and the full
pipeline including all evaluations is below US\$64 on a single
A100-SXM4-80GB. All systems are evaluated under a single greedy decoding pass.
On a 217-sample validation split, the strongest configuration raises overall
accuracy from 29.95\% (zero-shot) to 47.47\% (63.12\% physics, 18.42\% logic);
a controlled decomposition attributes $+9.22$ percentage points (pp) of the
gain to retrieval-augmented exemplars ($p<10^{-3}$) and a further, borderline
$+5.07$~pp to supervised fine-tuning ($p=0.054$). The overall zero-shot-to-cfg3
improvement is statistically significant ($p<10^{-7}$), whereas the
GRPO-over-logic-SFT increment is not ($p=0.61$); a three-seed evaluation yields $46.70\pm0.70\%$. A fair
multi-prompt re-evaluation of five 7B-class baselines returns 11.5--29.0\%,
which suggests that prompt-format mismatch alone does not explain the baseline
gap. On the external MMLU physics benchmark the fine-tuned model improves over
the zero-shot model (48.62\% versus 42.29\%), and a tool-augmented variant
(Python evaluation, Z3) yields a marginal $+0.46$~pp change. A structured error
analysis indicates that a substantial fraction of residual errors are
answer-extraction failures, with logical-fallacy errors forming the dominant
substantive-reasoning category, which identifies formal logic as the principal
bottleneck at this scale. All code, checkpoints, and evaluation scripts will be
released upon acceptance.}"""
abs_new = r"""\abstract[ABSTRACT]{Scientific and educational question answering couples
numerical physical reasoning with symbolic logical inference, yet
resource-constrained settings often preclude large proprietary models or
extensive reinforcement learning. This paper reports \modelname{}, a low-cost
pipeline for the \dataset{} mixed-domain (physics and logic) reasoning challenge,
built on a 4B-parameter open model. \modelname{} couples retrieval-augmented
few-shot inference with a three-stage curriculum: general supervised fine-tuning
(SFT), a logic-focused curriculum stage, and Group Relative Policy Optimization
(GRPO) with lightweight format, correctness, and unit-consistency rewards. Total
training cost is US\$18, and the full pipeline including all evaluations is below
US\$64 on a single A100-80GB. Every system is evaluated under a single greedy
decoding pass. On a 217-sample validation split, the strongest configuration
raises overall accuracy from 29.95\% to 47.47\% (63.12\% physics, 18.42\% logic).
A controlled decomposition attributes $+9.22$ percentage points (pp) of the gain
to retrieval-augmented exemplars and a further, borderline $+5.07$~pp to
supervised fine-tuning, whereas the GRPO increment over the curriculum stage is
not statistically significant. A fair multi-prompt re-evaluation places five
7B-class baselines at 11.5--29.0\%, in-domain fine-tuning transfers to MMLU
physics ($+6.33$~pp), and a tool-augmented variant yields only a marginal change.
A structured error analysis attributes most residual errors to answer-extraction
failures and identifies formal logic as the principal reasoning bottleneck at
this scale.}"""
t = rep(abs_old, abs_new, "abstract")

# ---- 8. introduction paragraph 3 (lineage) ----
intro_old = r"""The \dataset{} task provides a controlled testbed with 1{,}945 training
examples (1{,}213 physics, 732 logic) and a held-out evaluation set, with a
reported public baseline of 38.71\%. This study treats the task as a proxy for
mixed-domain scientific question answering and quantifies how far a
\basemodelname{} model can be advanced using publicly available fine-tuning
techniques under a fixed compute budget. To support an unambiguous
interpretation, every system in this study, fine-tuned and baseline, is
evaluated under a single greedy decoding pass without any answer-key-dependent
regeneration."""
intro_new = r"""\dataset{} is the 2nd International XAI Challenge for Transparent Educational
Question-Answering, organised as an IEEE~IJCNN~2026 competition
\citep{EXACT2026}. It succeeds the XAI Challenge 2025 on explainable educational
question answering over academic regulations \citep{Nguyen2025XAIChallenge} and
broadens the scope from academic-policy logic to STEM physics, so that a system
must address both logic-based educational queries and physics problems on
electric circuits and electrostatics. The task provides a controlled testbed with
1{,}945 training examples (1{,}213 physics, 732 logic) and a held-out evaluation
set, with a reported public baseline of 38.71\%. This study treats the task as a
proxy for mixed-domain scientific and educational reasoning and quantifies how
far a \basemodelname{} model can be advanced using publicly available
fine-tuning techniques under a fixed compute budget. To support an unambiguous
interpretation, every system in this study, fine-tuned and baseline, is evaluated
under a single greedy decoding pass without any answer-key-dependent
regeneration."""
t = rep(intro_old, intro_new, "intro-lineage")

# ---- 9. Related Work: new subsection 2.5 + positioning table ----
rw_anchor = r"""learning \citep{Bengio2009} motivates the staged data ordering used here.

%% ============================================================
\section{Problem Formulation}\label{sec3}"""
rw_new = r"""learning \citep{Bengio2009} motivates the staged data ordering used here.

\subsection{Explainable educational QA challenges}
The closest predecessor to \dataset{} is the XAI Challenge 2025, an explainable
educational question-answering competition organised by the URA Research Group at
Ho Chi Minh City University of Technology together with the TRNS-AI workshop at
IJCNN~2025 \citep{Nguyen2025XAIChallenge}. That challenge required systems to
produce not only final answers but also logic-grounded natural-language
explanations for questions about university policies; its dataset (481 training
and 50 test records) was constructed from first-order-logic templates with Z3
validation and expert-student review, with premises supplied in both natural
language and first-order logic. \dataset{} continues this line of work and
broadens it from academic-regulation logic to STEM physics, evaluating
submissions along answer correctness, explanation quality, and reasoning depth
while encouraging---but not requiring---symbolic tools such as Z3
\citep{EXACT2026}. In contrast to the 2025 challenge report, which presents the
competition, dataset, and evaluation protocol, the present paper is a
system-level empirical study: it fixes a single 4B open model and reports
controlled ablations of retrieval, supervised fine-tuning, logic-curriculum
adaptation, GRPO reward shaping, tool augmentation, and self-consistency.
Table~\ref{tabrelated} situates \modelname{} relative to representative reasoning
systems and to these two challenge artefacts.

\begin{table*}[t]
\centering
\caption{Positioning of \modelname{} relative to representative LLM reasoning
systems and to the EXACT challenge artefacts. ``Tuning'' indicates whether the
approach trains model weights.\label{tabrelated}}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}p{0.21\textwidth}p{0.18\textwidth}p{0.06\textwidth}p{0.40\textwidth}@{}}
\toprule
\textbf{Work} & \textbf{Focus / domain} & \textbf{Tuning} & \textbf{Relation to \modelname{}} \\
\midrule
XAI Challenge 2025 report \citep{Nguyen2025XAIChallenge} & Academic-regulation QA & --- & Predecessor; defines the challenge and dataset, not a system study \\
\dataset{} task \citep{EXACT2026} & Logic $+$ STEM physics & --- & Official task, rules, and evaluation; not a method \\
CoT / self-consistency \citep{Wei2022,Wang2023sc} & Prompt-time reasoning & No & Evaluated after 4B adaptation; no net gain observed \\
PAL \citep{Gao2022} & Program-aided arithmetic & No & Python assistance tested; only marginal gain \\
Logic-LM / LINC \citep{Pan2023,Olausson2023} & Logic with solvers & No & Z3 used, but the NL-logic parser limits coverage \\
DeepSeekMath / Qwen-Math \citep{Shao2024,Yang2024} & Maths-specialised LLMs & Yes & Motivation and baselines for low-cost GRPO \\
LoRA / QLoRA \citep{Hu2022,Dettmers2023} & Parameter-efficient tuning & Yes & Applied to a 4B model under a fixed budget \\
\modelname{} (this work) & Logic $+$ physics & Yes & Retrieval $+$ SFT $+$ GRPO with reproducible ablations \\
\bottomrule
\end{tabular*}
\end{table*}

%% ============================================================
\section{Problem Formulation}\label{sec3}"""
t = rep(rw_anchor, rw_new, "relatedwork-subsection")

# ---- 10. Problem Formulation: cite EXACT ----
pf_old = r"""Let $\mathcal{D}=\mathcal{D}_{\text{phys}}\cup\mathcal{D}_{\text{logic}}$ denote
the \dataset{} dataset. Each sample $(q_i,a_i^{*},s_i)\in\mathcal{D}$ comprises a"""
pf_new = r"""Let $\mathcal{D}=\mathcal{D}_{\text{phys}}\cup\mathcal{D}_{\text{logic}}$ denote
the \dataset{} dataset \citep{EXACT2026}. Each sample
$(q_i,a_i^{*},s_i)\in\mathcal{D}$ comprises a"""
t = rep(pf_old, pf_new, "pf-cite")

# ---- 11. Problem Formulation: dimensions sentence ----
pf2_old = r"""decoding pass; no component of the evaluation uses the ground-truth label to
decide whether to regenerate a response."""
pf2_new = r"""decoding pass; no component of the evaluation uses the ground-truth label to
decide whether to regenerate a response. The official challenge scores
submissions along three dimensions---answer correctness, explanation quality, and
reasoning depth---and encourages optional structured evidence such as
first-order-logic derivations and chain-of-thought steps \citep{EXACT2026}; the
present study targets the answer-correctness dimension and additionally examines
answer formatting and reasoning errors in Section~\ref{sec6}."""
t = rep(pf2_old, pf2_new, "pf-dimensions")

# ---- 12. Baselines: Gemini disclaimer ----
gem_old = r"""model (Gemini-2.5-flash-lite, accessed through an API) are evaluated on the 217
validation samples. To separate format compliance from reasoning capability,"""
gem_new = r"""model (Gemini-2.5-flash-lite, accessed through an API) are evaluated on the 217
validation samples. The challenge rules prohibit closed-source models in
submitted systems \citep{EXACT2026}; the Gemini baseline is therefore reported
only as a non-submission diagnostic reference and is not part of the
competition-compliant \modelname{} system, all components of which use
open-weight models. To separate format compliance from reasoning capability,"""
t = rep(gem_old, gem_new, "gemini-disclaimer")

# ---- 13. Main results: baseline-low explanation ----
mr_old = r"""over cfg2 is driven by physics: logic accuracy ($18.42\%$) does not improve over
cfg2 ($21.05\%$), which indicates that the GRPO reward is more effective for
answer formatting and numerical reasoning than for formal logical inference."""
mr_new = r"""over cfg2 is driven by physics: logic accuracy ($18.42\%$) does not improve over
cfg2 ($21.05\%$), which indicates that the GRPO reward is more effective for
answer formatting and numerical reasoning than for formal logical inference. The
low absolute scores of the maths-specialised 7B baselines should be read in light
of the mixed physics--logic format and the strict answer-extraction protocol
(Appendix~\ref{appA}) rather than as a general ranking of those models: even
under the fair best-of-three protocol, the heterogeneous answer space of
\dataset{} penalises systems that are not adapted to it."""
t = rep(mr_old, mr_new, "baseline-low")

# ---- 14. Error analysis: qualitative example table ----
err_anchor = r"""\end{table}

%% ============================================================
\section{Discussion}\label{sec7}"""
err_new = r"""\end{table}

To make these categories concrete, Table~\ref{tab11} lists representative cfg3
failures drawn from the single-pass run. The dominant pattern is an
answer-extraction mismatch in which the model reasons soundly but emits an answer
whose form the verifier cannot reconcile with a categorical ground truth,
together with truncated generations that stop before a final answer is emitted;
the substantive logical failures are typically faulty quantifier or existential
reasoning.

\begin{table*}[t]
\centering
\caption{Representative cfg3 errors (single-pass run) illustrating the dominant
categories of Table~\ref{tab10}.\label{tab11}}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}p{0.16\textwidth}p{0.44\textwidth}p{0.30\textwidth}@{}}
\toprule
\textbf{Category} & \textbf{Symptom (abridged)} & \textbf{Interpretation} \\
\midrule
E6: extraction mismatch & Capacitor charge after disconnection: the model answers ``$100~\mu$C'' (the correct value), but the ground truth is the categorical label ``unchanged'' & Sound reasoning scored wrong; numerical versus categorical answer space \\
E6: incomplete output & Electric-field problem: the generation terminates mid-derivation (``\ldots\ A is at $x=1$~cm. B is'') with no final answer & Generation-length limit; constrained decoding could recover the answer \\
E4: logical fallacy & Quantifier item: the model declares the query a tautology and answers ``Yes'' where the entailment is ``No'' & Faulty existential/quantifier reasoning, not an extraction issue \\
\bottomrule
\end{tabular*}
\end{table*}

%% ============================================================
\section{Discussion}\label{sec7}"""
t = rep(err_anchor, err_new, "error-examples")

# ---- 15. Discussion: empirical-study framing on logic gap ----
lg_old = r"""\citep{Lightman2023,Wang2024mathshepherd} or solver-guided training are the most
direct avenues for improvement."""
lg_new = r"""\citep{Lightman2023,Wang2024mathshepherd} or solver-guided training are the most
direct avenues for improvement. Because the measured gains are driven primarily
by the physics domain while logic accuracy remains below 23\%, \modelname{}
should be read as a low-cost empirical study of what curriculum and retrieval
achieve at this scale rather than as a complete solution to mixed-domain
reasoning."""
t = rep(lg_old, lg_new, "logic-gap-framing")

# ---- 16. Conclusion: SciBench + GPQA future work ----
cc_old = r"""removes the retrieval asymmetry, scaling to 7B and 14B checkpoints, and tighter
integration of solver-guided training signals for the logic domain."""
cc_new = r"""removes the retrieval asymmetry, scaling to 7B and 14B checkpoints, cross-dataset
evaluation on SciBench \citep{Wang2024scibench} and GPQA \citep{Rein2023}, and
tighter integration of solver-guided training signals for the logic domain."""
t = rep(cc_old, cc_new, "conclusion-futurework")

# ---- 17. Back matter ----
bm_old = r"""\bmsubsection*{Author Contributions}
The author(s) designed the study, implemented the pipeline, conducted the
experiments, and wrote the manuscript.

\bmsubsection*{Acknowledgments}
Withheld for review.

\bmsubsection*{Financial Disclosure}
None reported.

\bmsubsection*{Conflicts of Interest}
The authors declare no conflicts of interest.

\bmsubsection*{Data Availability Statement}
The code, evaluation scripts, trained adapters, and result artifacts that
support the findings of this study will be made available in a public repository
upon acceptance. The MMLU benchmark used for external evaluation is publicly
available from its original source. Any restrictions on the \dataset{}
competition data will be described in the repository documentation."""
bm_new = r"""\bmsubsection*{Author Contributions}
The author designed the study, implemented the pipeline, conducted the
experiments, analysed the results, and wrote the manuscript.

\bmsubsection*{Acknowledgments}
Withheld for review.

\bmsubsection*{Funding}
The author received no specific funding for this work.

\bmsubsection*{Conflicts of Interest}
The author declares no conflicts of interest.

\bmsubsection*{Data Availability Statement}
The source code, evaluation scripts, trained LoRA adapters, and result artefacts
that support the findings of this study are available at
\url{https://github.com/ANONYMISED-FOR-REVIEW} (a permanent archive will be
deposited on acceptance). The \dataset{} competition data are governed by the
challenge access policy and are not redistributed here; the MMLU benchmark used
for external evaluation is publicly available from its original source."""
t = rep(bm_old, bm_new, "back-matter")

# ---- 18. bibliography: manual thebibliography -> BibTeX ----
m = re.search(r"\\begin\{thebibliography\}\{99\}.*?\\end\{thebibliography\}", t, re.S)
assert m, "thebibliography block not found"
t = t[:m.start()] + "\\bibliographystyle{apacite}\n\\bibliography{references}" + t[m.end():]

open(TEX, "w").write(t)
print("TEX written OK.")

# ---- references.bib: append two new entries ----
new_entries = r"""

% ── EXACT challenge sources (added for journal revision) ─────────────────────

@inproceedings{Nguyen2025XAIChallenge,
  author    = {Long S. T. Nguyen and Khang H. N. Vo and Thu H. A. Nguyen
               and Tuan C. Bui and Duc Q. Nguyen and Thanh-Tung Tran
               and Anh D. Nguyen and Minh L. Nguyen and Fabien Baldacci
               and Thang H. Bui and Emanuel Di Nardo and Angelo Ciaramella
               and Son H. Le and Ihsan Ullah and Lorenzo Di Rocco
               and Tho T. Quan},
  title     = {Bridging {LLMs} and Symbolic Reasoning in Educational {QA}
               Systems: Insights from the {XAI} Challenge at {IJCNN} 2025},
  booktitle = {Proceedings of the 4th Italian Conference on Big Data and
               Data Science (ITADATA 2025)},
  series    = {CEUR Workshop Proceedings},
  volume    = {4152},
  publisher = {CEUR-WS.org},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4152/paper98.pdf},
}

@misc{EXACT2026,
  author       = {{URA Research Group}},
  title        = {{EXACT} 2026: The 2nd International {XAI} Challenge for
                  Transparent Educational Question-Answering},
  howpublished = {IEEE IJCNN 2026 Competition},
  year         = {2026},
  note         = {Accessed: June 2026},
  url          = {https://ura.hcmut.edu.vn/exact},
}
"""
for bibpath in (BIB, PAPER_BIB):
    b = open(bibpath).read()
    if "Nguyen2025XAIChallenge" in b:
        print(f"  {bibpath}: entries already present, skipping")
        continue
    open(bibpath, "w").write(b.rstrip() + "\n" + new_entries)
    print(f"  {bibpath}: appended 2 entries")

# ---- USG.cls: suppress bare https://doi.org/ when DOI empty ----
c = open(CLS).read()
doi_old = r"\href{https://doi.org/\thearticledoi}{https://doi.org/\thearticledoi}"
if doi_old in c and r"\ifx\thearticledoi\@empty\else\href{https://doi.org" not in c:
    doi_new = r"\ifx\thearticledoi\@empty\else" + doi_old + r"\fi"
    c = c.replace(doi_old, doi_new, 1)
    open(CLS, "w").write(c)
    print("USG.cls: DOI footer guarded.")
else:
    print("USG.cls: DOI already guarded or pattern not found.")

print("\nALL EDITS APPLIED.")
