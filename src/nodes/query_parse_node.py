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

# The platform's redaction sentinel (framework.security.pii_masking._MASK).
# The platform S-2 gate masks personal data — and, per an internal ticket, ordinary
# title-case proper nouns — before this node runs, so the sentinel arrives as
# an ordinary substring of the caller's text.
_MASK_SENTINEL = "[MASKED]"

# Recognized asset class keywords (stub — production uses Kronos classifier)
_ASSET_CLASS_KEYWORDS = {
    "equity": ["stock", "equity", "share", "earnings", "ipo", "dividend"],
    "fixed_income": ["bond", "yield", "rate", "credit", "spread", "maturity"],
    "fx": ["yen", "dollar", "usd", "jpy", "euro", "eur", "fx", "currency"],
    "macro": ["gdp", "inflation", "monetary", "boj", "fed", "macro", "economy"],
}

# Common company name fragments (stub — production uses NER over Kronos KB)
_KNOWN_COMPANIES = [
    "seven",
    "aeon",
    "fast retailing",
    "mufg",
    "toyota",
    "sony",
    "nintendo",
    "panasonic",
    "hitachi",
    "fujitsu",
    "ntt",
]


def _extract_asset_class(text: str) -> str:
    """Heuristic asset class extraction from query text."""
    lower = text.lower()
    for asset_class, keywords in _ASSET_CLASS_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return asset_class
    return "general"


def _extract_company(text: str) -> str:
    """Heuristic company name extraction from query text."""
    lower = text.lower()
    for company in _KNOWN_COMPANIES:
        if company in lower:
            return company
    return ""


def _extract_keywords(text: str) -> list[str]:
    """Extract domain-relevant keywords (stub — production uses Kronos tokenizer).

    The platform's redaction sentinel is dropped rather than tokenised: a
    masked value is not an extracted value, and letting "masked" through as a
    keyword would make a redaction marker drive retrieval as though it were a
    domain term the caller had asked about.
    """
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "do",
        "does",
        "did",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "not",
    }
    tokens = text.replace(_MASK_SENTINEL, " ").lower().split()
    return [t.strip("?.,!;:\"'") for t in tokens if t.strip("?.,!;:\"'") and t.strip("?.,!;:\"'") not in stop_words][
        :10
    ]


class QueryParseNode(FunctionNode):
    """Parse the incoming financial query into structured intent fields.

    Extracts: intent (str), keywords (list), asset_class (str), company (str).
    Downstream nodes (KronosRAGRetrieveNode, MarketSentimentAnalyzeNode) use
    parsed_query to scope their operations.

    Security: ANONYMOUS — inner domain node; trust already validated by
    outer pre_process (VERIFIED_EXTERNAL).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        validated_input = state.get("validated_input", state.get("user_input", ""))

        if not validated_input:
            emit_trace_event(
                "query_parse_failed",
                {"reason": "empty_validated_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["QueryParseNode: validated_input is empty"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + ("QueryParseNode: validated_input is empty"),
            }

        asset_class = _extract_asset_class(validated_input)
        company = _extract_company(validated_input)
        keywords = _extract_keywords(validated_input)

        # Classify intent: sentiment vs factual vs comparative
        lower = validated_input.lower()
        if any(w in lower for w in ["sentiment", "bullish", "bearish", "mood", "outlook"]):
            intent = "sentiment_query"
        elif any(w in lower for w in ["compare", "vs", "versus", "relative"]):
            intent = "comparative_query"
        else:
            intent = "factual_query"

        parsed_query = {
            "intent": intent,
            "keywords": keywords,
            "asset_class": asset_class,
            "company": company,
            "raw_query": validated_input,
        }

        emit_trace_event(
            "query_parsed",
            {
                "intent": intent,
                "asset_class": asset_class,
                "keyword_count": len(keywords),
            },
            state,
        )

        return {
            "parsed_query": parsed_query,
            "status": AgentStatus.SUCCESS.value,
        }
