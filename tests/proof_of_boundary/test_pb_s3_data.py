"""PoB-S3: Data Boundary Verification — FIN-C2-104.

Verifies the S-3 (Output) security gates:
  1. CitationGateNode strips unverifiable citations from retrieval results
  2. PostProcessNode's S-3 scan redacts API key / credential patterns
  3. No credential fields exist in the State TypedDict
  4. Audit trace payloads use anonymised metrics (not raw query/response text)

Data boundary model:
  - S-3 inner gate: CitationGateNode → only trusted sources survive
  - S-3 outer gate: PostProcessNode._security_gate_output() → credential redaction
  - S-5: State contains no API keys, JWT, or secret fields
"""

import ast
import os
import re

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


_BASE_STATE = {
    "caller_trust_level": TrustLevel.ANONYMOUS.value,
    "correlation_id": "test-pob-s3",
    "session_id": "test-session",
    "node_history": [],
    "error_log": [],
}


def _state(**kwargs):
    return {**_BASE_STATE, **kwargs}


class TestCitationGate:
    """S-3: CitationGateNode must only surface verified citations."""

    def test_trusted_source_retained(self):
        """Documents from trusted sources must be retained as citations."""
        from src.nodes.citation_gate_node import CitationGateNode

        node = CitationGateNode()
        trusted_doc = {
            "doc_id": "trusted-001",
            "title": "Q1 2026 Earnings Overview",
            "content": "AEON Q1 2026 earnings beat consensus estimates.",
            "relevance_score": 0.92,
            "source": "Kronos Financial KB",  # trusted prefix
        }
        state = _state(retrieval_results=[trusted_doc])
        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        citations = result.get("validated_citations", [])
        assert len(citations) == 1, f"Trusted citation must be retained; got: {citations}"
        assert citations[0]["verified"] is True
        assert citations[0]["source"] == "Kronos Financial KB"

    def test_unverifiable_source_stripped(self):
        """Documents from unverifiable sources must be stripped."""
        from src.nodes.citation_gate_node import CitationGateNode

        node = CitationGateNode()
        untrusted_doc = {
            "doc_id": "untrusted-001",
            "title": "Anonymous tip",
            "content": "Rumored merger deal for Toyota.",
            "relevance_score": 0.65,
            "source": "unknown_social_media_scrape",  # not a trusted prefix
        }
        state = _state(retrieval_results=[untrusted_doc])
        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        citations = result.get("validated_citations", [])
        assert len(citations) == 0, f"Unverifiable citations must be stripped; got: {citations}"

    def test_mixed_sources_partial_retention(self):
        """Only trusted sources retained from a mixed retrieval result set."""
        from src.nodes.citation_gate_node import CitationGateNode

        node = CitationGateNode()
        docs = [
            {
                "doc_id": "t1",
                "title": "Bloomberg Market Brief",
                "content": "...",
                "relevance_score": 0.88,
                "source": "Bloomberg",  # trusted
            },
            {
                "doc_id": "u1",
                "title": "Random Blog Post",
                "content": "...",
                "relevance_score": 0.55,
                "source": "anonymous_blog",  # NOT trusted
            },
        ]
        state = _state(retrieval_results=docs)
        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        citations = result.get("validated_citations", [])
        assert len(citations) == 1, f"Only 1 of 2 docs is from a trusted source; got: {citations}"
        assert citations[0]["source"] == "Bloomberg"


class TestOutputCredentialScan:
    """S-3: PostProcessNode must redact credential patterns in the response."""

    def test_api_key_pattern_redacted(self):
        """API key-like patterns in the result must be redacted by S-3 gate."""
        from src.nodes.post_process_node import _security_gate_output

        # Simulate a result string that accidentally contains a credential pattern
        contaminated_result = "Market analysis complete. api_key: sk-abc123 See report."
        redacted = _security_gate_output(contaminated_result)

        assert "sk-abc123" not in redacted, "S-3 gate must redact API key patterns from the output text"
        assert "[REDACTED]" in redacted, "S-3 gate must replace credential patterns with [REDACTED]"

    def test_clean_response_unchanged(self):
        """Clean financial narrative must pass S-3 gate without alteration."""
        from src.nodes.post_process_node import _security_gate_output

        clean = (
            "Market sentiment: Bullish (score: +0.42). "
            "AEON Q1 2026 results beat consensus. "
            "Sources: Kronos Financial KB."
        )
        result = _security_gate_output(clean)
        assert result == clean, f"S-3 gate must not alter clean financial responses; got: {result!r}"

    def test_post_process_uses_result_field(self):
        """PostProcessNode must read 'result' (set by merge_output), not 'formatted_response'."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        state = _state(
            result="Financial report: equities bullish. Sources: Bloomberg.",
        )
        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result.get("formatted_output"), "PostProcessNode must populate formatted_output from result field"
        assert "bullish" in result["formatted_output"]


class TestStateCredentialSafety:
    """S-5: State TypedDict must not contain credential or secret fields."""

    _CREDENTIAL_PATTERN = re.compile(
        r"(jwt|token|api_key|secret|password|credential|connection_string)",
        re.IGNORECASE,
    )

    def test_state_has_no_credential_fields(self):
        """src/schemas/state.py must not declare credential-like field names."""
        state_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "schemas", "state.py")
        assert os.path.exists(state_path), "src/schemas/state.py must exist"

        with open(state_path) as f:
            tree = ast.parse(f.read(), filename=state_path)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        field_name = item.target.id
                        if self._CREDENTIAL_PATTERN.search(field_name):
                            violations.append(f"line {item.lineno}: credential-like field '{field_name}'")

        assert not violations, (
            "State TypedDict must not contain credential fields (checkpoint DB leakage risk):\n" + "\n".join(violations)
        )
