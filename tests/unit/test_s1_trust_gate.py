# FIN-C2-104 — Unit tests: S-1 trust gate.
#
# Every invocation here goes through node(state) — BaseNode.__call__ — which
# runs S-1 trust -> S-2 input gate -> execute() -> S-3 output gate. Calling
# node.execute(state) directly bypasses __call__ and therefore never exercises
# the gate; the rest of this repo's suite calls execute() directly, so before
# this file the only S-1 coverage was the declaration assertion in
# tests/proof_of_boundary/test_pb_s1_security.py — no test observed the gate
# actually refusing a caller.
#
# A denial RETURNS an error dict (it never raises): status ERROR and
# "trust gate denied" in error_log. execute() does not run on a denial, so
# none of the keys that node writes appear in the returned dict, and none of
# the node's own error text appears either — that, not the generic ERROR
# status, is what proves the gate fired rather than some later failure.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.citation_gate_node import CitationGateNode
from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode
from src.nodes.kronos_rag_retrieve_node import KronosRAGRetrieveNode
from src.nodes.market_sentiment_analyze_node import MarketSentimentAnalyzeNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.query_parse_node import QueryParseNode
from src.nodes.response_format_node import ResponseFormatNode

# Keys only PreProcessNode.execute() writes on its success path.
_PRE_PROCESS_OUTPUT_KEYS = ("validated_input", "enriched_context")

_VALID_QUERY = "What is the market sentiment for Japanese equities this quarter?"


def _state(trust_value: str, user_input: str = _VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "correlation_id": "unit-test",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestS1TrustGateDenial:
    """An ANONYMOUS caller must be denied at the VERIFIED_EXTERNAL boundary."""

    def test_anonymous_caller_denied_on_pre_process(self):
        """ANONYMOUS caller on PreProcessNode: __call__ returns an error dict."""
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL

        result = node(_state(TrustLevel.ANONYMOUS.value))

        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any("trust gate denied" in str(entry) for entry in error_log), (
            f"expected 'trust gate denied' in error_log, got: {error_log}"
        )

    def test_denial_returns_none_of_the_node_output_keys(self):
        """execute() must not run on denial — none of its output keys appear."""
        node = PreProcessNode()

        result = node(_state(TrustLevel.ANONYMOUS.value))

        for key in _PRE_PROCESS_OUTPUT_KEYS:
            assert key not in result, f"{key} leaked from a denied invocation — execute() ran despite the S-1 gate"

    def test_denial_error_is_not_the_nodes_own_error(self):
        """The denial must not carry any of PreProcessNode's own error text.

        Every error PreProcessNode itself produces names the node and its
        rejection reason; a denial is produced above the node, so none of that
        text may appear.
        """
        node = PreProcessNode()

        result = node(_state(TrustLevel.ANONYMOUS.value))

        assert result.get("status") == AgentStatus.ERROR.value, (
            "an ANONYMOUS caller must be denied here — a non-ERROR status means the gate did not fire"
        )
        joined = " ".join(str(entry) for entry in result.get("error_log", []))
        for own_error_fragment in (
            "must be a string",
            "is empty or missing",
            "injection marker",
            "potential PII detected",
        ):
            assert own_error_fragment not in joined, (
                f"denial error_log carries the node's own error ({own_error_fragment!r}) — execute() ran"
            )

    def test_denial_holds_for_input_the_node_would_reject_anyway(self):
        """Empty input from an ANONYMOUS caller is still a GATE denial.

        Without the gate this input reaches execute() and produces the node's
        own 'user_input is empty or missing' error, so this case separates the
        two failure modes.
        """
        node = PreProcessNode()

        result = node(_state(TrustLevel.ANONYMOUS.value, user_input=""))

        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(entry) for entry in result.get("error_log", []))
        assert not any("is empty or missing" in str(entry) for entry in result.get("error_log", [])), (
            "execute() ran and produced its own empty-input error despite the S-1 gate"
        )


class TestS1TrustGateAdmission:
    """A VERIFIED_EXTERNAL caller must clear the gate and reach execute()."""

    def test_verified_external_caller_passes_pre_process(self):
        node = PreProcessNode()

        result = node(_state(TrustLevel.VERIFIED_EXTERNAL.value))

        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input") == _VALID_QUERY
        assert result.get("enriched_context")

    def test_node_own_validation_still_applies_after_the_gate(self):
        """Past the gate, the node's own S-2 checks produce its own errors."""
        node = PreProcessNode()

        result = node(_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input="   "))

        assert result.get("status") == AgentStatus.ERROR.value
        assert any("is empty or missing" in str(entry) for entry in result.get("error_log", [])), (
            "post-gate rejection must carry the node's own empty-input error"
        )
        assert not any("trust gate denied" in str(entry) for entry in result.get("error_log", []))

    def test_anonymous_caller_admitted_by_an_inner_node(self):
        """Inner domain nodes are ANONYMOUS and must admit an ANONYMOUS caller."""
        node = QueryParseNode()

        result = node(_state(TrustLevel.ANONYMOUS.value, validated_input=_VALID_QUERY))

        assert not any("trust gate denied" in str(entry) for entry in result.get("error_log", [])), (
            "an ANONYMOUS inner node must not deny an ANONYMOUS caller"
        )
        assert result.get("parsed_query"), "execute() must have run and parsed the query"


class TestTrustLevelMatrix:
    """The declared trust matrix: outer S-1 boundary vs inner domain nodes."""

    def test_pre_process_is_the_verified_external_boundary(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_and_post_process_nodes_admit_anonymous(self):
        for node_cls in (
            QueryParseNode,
            InsiderPatternFilterNode,
            KronosRAGRetrieveNode,
            MarketSentimentAnalyzeNode,
            CitationGateNode,
            ResponseFormatNode,
            PostProcessNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS — the outer PreProcessNode is the S-1 boundary"
            )
