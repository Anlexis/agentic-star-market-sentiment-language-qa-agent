"""PB-6: Backbone invoke-order verification — FIN-C2-104.

Verifies that FinancialMarketSentimentQAAgent().invoke() with a
VERIFIED_EXTERNAL caller on a SUCCESS-yielding payload produces:

  node_history == [
      "InitializeNode", "PreProcessNode", "MarketQAGraphNode",
      "PostProcessNode", "FinalizeNode"
  ]

Caller trust: VERIFIED_EXTERNAL (not INTERNAL — an INTERNAL context
masks the inner-node trust trap: INTERNAL(2) >= ANONYMOUS(0) always
passes, so the test would go green even if inner nodes were mis-declared
INTERNAL, hiding the real STG failure path for VERIFIED_EXTERNAL callers).

Payload: must yield AgentStatus.SUCCESS.value so the backbone runs all 5 slots
(a non-SUCCESS main slot short-circuits main→finalize, skipping
post_process, and the truncated node_history causes a false pass).
"""

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

# ── Template-specific constants ────────────────────────────────────────────────
# The class name of the `main`-slot GraphNode in src/graph/graph.py.
_MAIN_SLOT_NODE = "MarketQAGraphNode"

# A SUCCESS-yielding domain payload: public-information financial query with
# no MNPI patterns so InsiderPatternFilterNode passes cleanly.
_VALID_PAYLOAD = (
    "What is the current market sentiment for Japanese equities based on recent earnings from AEON and Fast Retailing?"
)

# Expected backbone node_history (class name strings).
_EXPECTED_BACKBONE = [
    "InitializeNode",
    "PreProcessNode",
    _MAIN_SLOT_NODE,
    "PostProcessNode",
    "FinalizeNode",
]
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_domain_emit(monkeypatch):
    """Mute domain-node emit_trace_event so tests run without an audit backend.

    Patched at each node MODULE (not via sys.modules) so the real shared
    package loads normally — the framework's own base_node.py imports
    shared.security.class_jwt_detector at load time and must not be stubbed.
    """
    for mod in (
        "src.nodes.pre_process_node",
        "src.nodes.post_process_node",
        "src.nodes.query_parse_node",
        "src.nodes.insider_pattern_filter_node",
        "src.nodes.kronos_rag_retrieve_node",
        "src.nodes.market_sentiment_analyze_node",
        "src.nodes.citation_gate_node",
        "src.nodes.response_format_node",
    ):
        monkeypatch.setattr(f"{mod}.emit_trace_event", lambda *a, **k: None)


class TestBackboneInvokeOrder:
    """PB-6: fixed-pipeline backbone must run in the correct slot order."""

    def test_backbone_order_success(self):
        """Full invoke with VERIFIED_EXTERNAL caller asserting node_history order."""
        from src.graph.graph import FinancialMarketSentimentQAAgent

        agent = FinancialMarketSentimentQAAgent()
        agent.compile()

        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(user_input=_VALID_PAYLOAD, ctx=ctx)

        assert result["status"] == AgentStatus.SUCCESS.value, (
            f"Expected AgentStatus.SUCCESS.value but got {result['status']}.\nerror_log: {result.get('error_log', [])}"
        )

        node_history = result.get("node_history", [])

        # Normalise to class-name strings (node_history may contain strings or types)
        history_names = []
        for entry in node_history:
            if isinstance(entry, str):
                history_names.append(entry)
            elif isinstance(entry, type):
                history_names.append(entry.__name__)
            else:
                history_names.append(type(entry).__name__)

        assert history_names == _EXPECTED_BACKBONE, (
            f"Backbone order violation.\nExpected: {_EXPECTED_BACKBONE}\nActual:   {history_names}"
        )

    def test_output_key_present(self):
        """invoke() result must expose 'output' key (not 'formatted_output')."""
        from src.graph.graph import FinancialMarketSentimentQAAgent

        agent = FinancialMarketSentimentQAAgent()
        agent.compile()

        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(user_input=_VALID_PAYLOAD, ctx=ctx)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "output" in result, f"invoke() result must expose 'output' key; got keys: {list(result.keys())}"
        assert result["output"] is not None, "output must not be None on SUCCESS"
