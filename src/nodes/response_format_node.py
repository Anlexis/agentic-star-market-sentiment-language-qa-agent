"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings

import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Longest caller query echoed back into the report. Without a cap the echo is
# unbounded: a 35,000-character question produced a 35,447-character "report",
# nearly all of it the caller's own text.
_MAX_ECHOED_QUERY = 200

# Any run of whitespace — newlines included — collapses to a single space.
_WHITESPACE_RUN_RE = re.compile(r"\s+")
# Markdown structural characters, stripped so echoed text cannot manufacture a
# heading, list item, table row, bold run or blockquote inside the report.
_MARKDOWN_STRUCTURE_RE = re.compile(r"[#*_`>|\[\]]")

# The platform's redaction sentinel (framework.security.pii_masking._MASK).
# It arrives inside the caller text, and its brackets are markdown structure —
# so a naive strip renders it as the bare word "MASKED", which a reader takes
# for something they typed. The redaction has to stay legible AS a redaction,
# so it is lifted out before the strip and restored in a bracket-free form.
_MASK_SENTINEL = "[MASKED]"
_MASK_PLACEHOLDER = "\x00MASK\x00"
_MASK_RENDERED = "‹redacted›"


def _inert_echo(text: str) -> str:
    """Render caller text safe to interpolate into the cited report.

    The report is a structured document with a **Sources:** block that a reader
    is entitled to trust. Echoed verbatim, a caller query carrying newlines and
    markdown produced a SECOND, forged sources block above the real one —
    `"outlook\\n\\n**Sources:**\\n1. <official-sounding body> (FSA)"` rendered as
    an authentic-looking citation the agent never retrieved.

    Structure is therefore removed rather than escaped: whitespace runs collapse
    to single spaces (so no line break can start a new block), markdown
    structural characters are dropped, and the result is length-capped.
    """
    flattened = _WHITESPACE_RUN_RE.sub(" ", text).strip()
    # Protect the redaction sentinel across the markdown strip.
    protected = flattened.replace(_MASK_SENTINEL, _MASK_PLACEHOLDER)
    inert = _MARKDOWN_STRUCTURE_RE.sub("", protected).strip()
    inert = inert.replace(_MASK_PLACEHOLDER, _MASK_RENDERED)
    if len(inert) > _MAX_ECHOED_QUERY:
        inert = inert[:_MAX_ECHOED_QUERY].rstrip() + "…"
    return inert


def _format_sentiment_summary(sentiment_scores: dict[str, Any]) -> str:
    """Render a human-readable sentiment summary."""
    if not sentiment_scores:
        return ""
    overall = sentiment_scores.get("overall", "neutral")
    score = sentiment_scores.get("score", 0.0)
    bullish = sentiment_scores.get("bullish_signals", [])
    bearish = sentiment_scores.get("bearish_signals", [])

    parts = [f"Market sentiment: **{overall.capitalize()}** (score: {score:+.2f})"]
    if bullish:
        parts.append(f"Bullish signals: {', '.join(bullish[:5])}")
    if bearish:
        parts.append(f"Bearish signals: {', '.join(bearish[:5])}")
    return " | ".join(parts)


def _format_citations(citations: list[dict[str, Any]]) -> str:
    """Render verified citations as a footnote block."""
    if not citations:
        return ""
    lines = ["\n\n**Sources:**"]
    for i, c in enumerate(citations, 1):
        title = c.get("title", "")
        source = c.get("source", "")
        lines.append(f"{i}. {title} ({source})")
    return "\n".join(lines)


class ResponseFormatNode(FunctionNode):
    """Assemble the final financial Q&A response with sentiment analysis and citations.

    Combines:
    - The user's original query intent (from parsed_query)
    - The Kronos knowledge base answer summary
    - Market sentiment scores
    - Verified citations (from CitationGateNode)

    into a structured, human-readable financial intelligence response.

    Security: ANONYMOUS — inner domain node; trust validated by outer pre_process.
    Output is the final assembled response before being surfaced via get_output().
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        parsed_query = state.get("parsed_query", {})
        sentiment_scores = state.get("sentiment_scores", {})
        validated_citations = state.get("validated_citations", [])
        retrieval_results = state.get("retrieval_results", [])

        intent = parsed_query.get("intent", "factual_query")
        asset_class = parsed_query.get("asset_class", "general")
        company = parsed_query.get("company", "")
        raw_query = parsed_query.get("raw_query", "")

        # Build the primary answer from top retrieval result
        top_doc_content = ""
        if retrieval_results:
            top_doc = retrieval_results[0] if isinstance(retrieval_results[0], dict) else {}
            top_doc_content = top_doc.get("content", "")

        # Compose response sections.
        # `company` and `asset_class` come from closed lists in QueryParseNode,
        # so they are inert by construction; `raw_query` is caller text and is
        # the only field that has to be neutralised before it is interpolated.
        scope = company if company else asset_class.replace("_", " ").title()

        header = f"**Financial Intelligence Report — {scope}**\n"
        echoed_query = _inert_echo(raw_query)
        if echoed_query:
            header += f"Query: {echoed_query}\n"

        answer_section = (
            f"\n{top_doc_content}"
            if top_doc_content
            else "\nNo relevant documents were found in the Kronos knowledge base for this query."
        )

        sentiment_section = ""
        if intent in ("sentiment_query", "factual_query") and sentiment_scores:
            sentiment_summary = _format_sentiment_summary(sentiment_scores)
            if sentiment_summary:
                sentiment_section = f"\n\n{sentiment_summary}"

        citation_section = _format_citations(validated_citations)

        formatted_response = header + answer_section + sentiment_section + citation_section

        emit_trace_event(
            "response_formatted",
            {
                "intent": intent,
                "asset_class": asset_class,
                "company": company,
                "citation_count": len(validated_citations),
                "response_length": len(formatted_response),
            },
            state,
        )

        return {
            "formatted_response": formatted_response,
            "status": AgentStatus.SUCCESS.value,
        }
