"""
TRA-SAE Agent Graph v2 — LangGraph StateGraph
===============================================
Dual-LoRA specialist architecture with subject router and self-consistency.

Graph topology (v2):
    router
      │
    retrieve_context
      │
    generate_answer  ◄──────────────────┐
      │                                 │ (retry_count < 2 AND not verified)
    verify_answer                       │
      │                                 │
      ├─── verified=True ──► generate_explanation
      │
      └─── verified=False ─► generate_answer (attempt 1 = self-consistency)
                  │
                  └─── retry_count >= 2 ──► generate_explanation (best-of-N)
                            │
                       format_output
                            │
                           END

Changes vs v1:
  - _MAX_RETRIES = 2  (was 3)
  - router_node added as entry point (classifies physics/logic)
  - adapter_name added to AgentState
  - _router added to runtime resources
  - topology: router → retrieve → generate → verify → [generate|explain] → format
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


# ──────────────────────────────────────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Input fields (set before graph.invoke()) ──────────────────────────────
    question:     str           # The question text
    subject:      str           # "logic" | "physics"  (set by router or dataset)
    ground_truth: str           # Ground-truth answer (eval mode; empty in prod)

    # ── Runtime resources (injected at invoke time) ───────────────────────────
    _model:     Any             # Loaded PEFT multi-adapter model
    _tokenizer: Any             # Tokenizer
    _retriever: Any             # src.retriever.Retriever instance
    _router:    Any             # src.router.SubjectRouter instance (or None)
    _device:    str             # "cuda" | "cpu"

    # ── Router output ─────────────────────────────────────────────────────────
    adapter_name: str           # LoRA adapter to activate ("physics" | "logic")

    # ── Intermediate state ────────────────────────────────────────────────────
    retrieved_examples: list    # Top-k similar training examples
    raw_output:         str     # Latest raw model generation
    all_attempts:       list    # All raw generations across retries
    generated_answer:   str     # Extracted answer text
    verified:           bool    # Whether symbolic verifier accepted the answer
    retry_count:        int     # Number of generate→verify attempts so far

    # ── Output fields ─────────────────────────────────────────────────────────
    explanation:  str           # Explanation text
    confidence:   float         # Confidence score (0–1); vote_frac for SC
    final_output: str           # Final XML-formatted answer+explanation


_MAX_RETRIES = 2   # v2: 2 attempts (attempt 0 = greedy, attempt 1 = self-consistency)


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge
# ──────────────────────────────────────────────────────────────────────────────

def _route_after_verify(state: AgentState) -> str:
    """Conditional edge after verify_answer_node."""
    if state.get("verified", False):
        return "generate_explanation"
    if state.get("retry_count", 0) < _MAX_RETRIES:
        return "generate_answer"
    return "generate_explanation"   # best-of-N fallback


# ──────────────────────────────────────────────────────────────────────────────
# Graph factory
# ──────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    """Build and compile the Agent v2 StateGraph (dual-LoRA + self-consistency).

    Returns:
        Compiled LangGraph app callable with `.invoke(state)`.
    """
    from src.agent_nodes import (
        router_node,
        retrieve_context_node,
        generate_answer_node,
        verify_answer_node,
        generate_explanation_node,
        format_output_node,
    )

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router",               router_node)
    graph.add_node("retrieve_context",     retrieve_context_node)
    graph.add_node("generate_answer",      generate_answer_node)
    graph.add_node("verify_answer",        verify_answer_node)
    graph.add_node("generate_explanation", generate_explanation_node)
    graph.add_node("format_output",        format_output_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("router")
    graph.add_edge("router",               "retrieve_context")
    graph.add_edge("retrieve_context",     "generate_answer")
    graph.add_edge("generate_answer",      "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        _route_after_verify,
        {
            "generate_answer":       "generate_answer",
            "generate_explanation":  "generate_explanation",
        },
    )
    graph.add_edge("generate_explanation", "format_output")
    graph.add_edge("format_output",        END)

    return graph.compile()
