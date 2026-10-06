"""PoB-S1: Security Boundary Verification — FIN-C2-104.

Verifies the S-1 (Trust + Input) security gates:
  1. PreProcessNode requires VERIFIED_EXTERNAL trust (not ANONYMOUS)
  2. InsiderPatternFilterNode detects and blocks MNPI-pattern queries
  3. FSA MRM high-severity audit event is emitted with correct payload
     (payload arg, not the full call repr — raw query must NOT be in the payload)

Security model:
  - Outer backbone gate: pre_process required_trust_level = VERIFIED_EXTERNAL
  - Inner S-1 gate: InsiderPatternFilterNode → ERROR on MNPI patterns
"""

from unittest.mock import MagicMock, patch

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


@pytest.fixture(autouse=True)
def patch_emit(monkeypatch):
    """Mute emit_trace_event in all domain node modules."""
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


class TestTrustGate:
    """S-1: pre_process trust level must be VERIFIED_EXTERNAL."""

    def test_pre_process_requires_verified_external(self):
        """PreProcessNode must declare VERIFIED_EXTERNAL as its minimum trust level."""
        from src.nodes.pre_process_node import PreProcessNode

        assert PreProcessNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL, (
            f"PreProcessNode.required_trust_level must be VERIFIED_EXTERNAL "
            f"(FSA MRM requirement), got: {PreProcessNode.required_trust_level}"
        )

    def test_inner_nodes_are_anonymous(self):
        """All inner domain nodes must declare ANONYMOUS trust (not INTERNAL)."""
        from src.nodes.query_parse_node import QueryParseNode
        from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode
        from src.nodes.kronos_rag_retrieve_node import KronosRAGRetrieveNode
        from src.nodes.market_sentiment_analyze_node import MarketSentimentAnalyzeNode
        from src.nodes.citation_gate_node import CitationGateNode
        from src.nodes.response_format_node import ResponseFormatNode

        inner_nodes = [
            QueryParseNode,
            InsiderPatternFilterNode,
            KronosRAGRetrieveNode,
            MarketSentimentAnalyzeNode,
            CitationGateNode,
            ResponseFormatNode,
        ]
        for node_cls in inner_nodes:
            assert node_cls.required_trust_level == TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__}.required_trust_level must be ANONYMOUS "
                f"(inner nodes receive outer InvocationContext unchanged; "
                f"INTERNAL would deny VERIFIED_EXTERNAL callers). "
                f"Got: {node_cls.required_trust_level}"
            )


class TestInsiderPatternFilter:
    """S-1: InsiderPatternFilterNode must block MNPI-pattern queries."""

    _BASE_STATE = {
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "test-pob-s1",
        "session_id": "test-session",
        "node_history": [],
        "error_log": [],
    }

    def _state(self, **kwargs):
        return {**self._BASE_STATE, **kwargs}

    def test_clean_query_passes(self):
        """A public-information financial query must pass the MNPI filter."""
        from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode

        node = InsiderPatternFilterNode()
        state = self._state(
            validated_input="What is the market sentiment for AEON earnings?",
            parsed_query={
                "intent": "sentiment_query",
                "keywords": ["market", "sentiment", "AEON", "earnings"],
                "asset_class": "equity",
                "company": "aeon",
                "raw_query": "What is the market sentiment for AEON earnings?",
            },
        )
        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result.get("insider_risk_flag") is False

    def test_mnpi_query_blocked(self):
        """A query with MNPI patterns must be blocked with ERROR status."""
        from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode

        node = InsiderPatternFilterNode()
        state = self._state(
            validated_input="insider trading opportunity before announcement",
            parsed_query={
                "intent": "factual_query",
                "keywords": ["insider", "trading", "announcement"],
                "asset_class": "equity",
                "company": "",
                "raw_query": "insider trading opportunity before announcement",
            },
        )
        result = node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value, (
            "InsiderPatternFilterNode must return ERROR for MNPI-pattern queries"
        )
        assert result.get("insider_risk_flag") is True
        assert result.get("error_log"), "MNPI block must populate error_log"

    def test_mnpi_audit_event_payload(self):
        """MNPI detection must emit a high-severity audit event.

        Assert against the PAYLOAD (2nd positional arg call.args[1]),
        not repr(call_args_list) — the 3rd arg is the full state which
        legitimately carries the raw query and would cause a false fail.
        """
        import src.nodes.insider_pattern_filter_node as mod

        spy = MagicMock()

        with patch.object(mod, "emit_trace_event", spy):
            from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode

            node = InsiderPatternFilterNode()
            mnpi_query = "material non-public information before earnings release"
            state = {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "correlation_id": "test-pob-s1-spy",
                "session_id": "spy-session",
                "validated_input": mnpi_query,
                "parsed_query": {"raw_query": mnpi_query, "intent": "factual_query"},
                "node_history": [],
                "error_log": [],
            }
            node.execute(state)

        assert spy.called, "emit_trace_event must be called on MNPI detection"

        # Find the MNPI detection event (severity=HIGH)
        mnpi_event_calls = [c for c in spy.call_args_list if len(c.args) > 1 and c.args[1].get("severity") == "HIGH"]
        assert mnpi_event_calls, (
            "Must emit a HIGH-severity audit event for MNPI detection; "
            f"payloads: {[c.args[1] for c in spy.call_args_list if len(c.args) > 1]}"
        )

        # Verify the payload does not expose the raw query text
        payload = mnpi_event_calls[0].args[1]
        assert "raw_query" not in payload, "MNPI audit payload must not expose the raw query text (PII/MNPI leak risk)"
        assert "pattern" in payload, "MNPI audit payload must include the matched pattern"
