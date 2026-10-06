"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from typing import Any, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """FIN-C2-104 agent state — FinancialMarketSentimentQAAgent.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    Domain-specific fields below follow the Cat-2 nested pipeline:
      PreProcessNode → (inner workflow) → PostProcessNode
    """

    # ── Outer pre_process output ──────────────────────────────────────────
    # Sanitized user query after trust gate and basic validation.
    validated_input: Optional[str]

    # Channel and source metadata for FSA MRM audit trail.
    enriched_context: Optional[dict[str, Any]]

    # ── Inner domain workflow outputs (MarketSentimentWorkflowGraph) ──────
    # QueryParseNode: structured decomposition of the financial query.
    # Fields: intent (str), keywords (list[str]), asset_class (str), company (str)
    parsed_query: Optional[dict[str, Any]]

    # InsiderPatternFilterNode: True if FSA-regulated insider information
    # pattern was detected in the query; triggers ERROR short-circuit.
    insider_risk_flag: Optional[bool]

    # KronosRAGRetrieveNode: list of retrieved Kronos financial KB documents.
    # Each item: {doc_id: str, title: str, content: str, relevance_score: float}
    retrieval_results: Optional[list[dict[str, Any]]]

    # MarketSentimentAnalyzeNode: sentiment analysis result.
    # Fields: overall (str), score (float), bullish_signals (list), bearish_signals (list)
    sentiment_scores: Optional[dict[str, Any]]

    # CitationGateNode: verified source citations (unverifiable refs stripped).
    # Each item: {source: str, title: str, verified: bool}
    validated_citations: Optional[list[dict[str, Any]]]

    # ResponseFormatNode: final formatted financial Q&A response with citations.
    formatted_response: Optional[str]

    # ── Outer graph merge_output (from MarketQAGraphNode) ─────────────────
    # Assembled final answer surfaced to PostProcessNode and FinalizeNode.
    result: Optional[str]
