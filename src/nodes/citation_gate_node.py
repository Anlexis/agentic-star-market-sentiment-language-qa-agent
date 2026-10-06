"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Trusted source prefixes — production validates against Kronos source registry.
# Only documents with a verified source are eligible for citation.
_TRUSTED_SOURCE_PREFIXES = (
    "Kronos Financial KB",
    "Bloomberg",
    "Reuters",
    "FSA",
    "TSE",
    "MUFG Research",
    "Nomura Research",
)


# S-3: module-level citation validation helper.
# Removes unverifiable references before they reach the response.
def _verify_citation(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a single retrieval result as a citable source.

    Returns a citation dict if verifiable, None otherwise.
    A citation is verifiable if its source starts with a trusted prefix.
    """
    if not isinstance(doc, dict):
        return None
    source = doc.get("source", "")
    title = doc.get("title", "stub document")
    doc_id = doc.get("doc_id", "")

    verified = any(source.startswith(prefix) for prefix in _TRUSTED_SOURCE_PREFIXES)
    if not verified:
        return None

    return {
        "source": source,
        "title": title,
        "doc_id": doc_id,
        "verified": True,
    }


class CitationGateNode(FunctionNode):
    """S-3: Validate source citations and strip unverifiable references.

    Iterates over retrieval_results and applies the citation verification
    gate. Only documents from trusted sources (Kronos KB, Bloomberg, Reuters,
    FSA, TSE, etc.) are retained as citable references.

    Unverifiable citations are dropped before the response is assembled,
    preventing hallucinated or untrustworthy source attribution in the
    financial Q&A output.

    Security: ANONYMOUS — inner domain node; trust validated by outer pre_process.
    S-3 output gate responsibility: validate and strip unverifiable citations.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieval_results = state.get("retrieval_results", [])

        validated_citations: list[dict[str, Any]] = []
        dropped_count = 0

        for doc in retrieval_results:
            citation = _verify_citation(doc)
            if citation is not None:
                validated_citations.append(citation)
            else:
                dropped_count += 1

        emit_trace_event(
            "citation_gate_applied",
            {
                "total_docs": len(retrieval_results),
                "verified_citations": len(validated_citations),
                "dropped_unverifiable": dropped_count,
            },
            state,
        )

        return {
            "validated_citations": validated_citations,
            "status": AgentStatus.SUCCESS.value,
        }
