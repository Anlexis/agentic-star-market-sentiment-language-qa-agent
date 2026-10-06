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

# Sentiment signal words used by the Kronos domain lexicon (stub).
# Production uses the Kronos model's built-in financial sentiment classifier.
_BULLISH_WORDS = {
    "growth",
    "rally",
    "bullish",
    "positive",
    "strong",
    "record",
    "beat",
    "exceed",
    "outperform",
    "upgrade",
    "buy",
    "optimistic",
    "momentum",
    "surge",
    "rise",
    "inflow",
    "demand",
}
_BEARISH_WORDS = {
    "decline",
    "bearish",
    "negative",
    "weak",
    "miss",
    "underperform",
    "downgrade",
    "sell",
    "pessimistic",
    "slump",
    "fall",
    "outflow",
    "risk",
    "concern",
    "uncertainty",
    "loss",
    "warning",
}


def _score_text(text: str) -> tuple[list[str], list[str]]:
    """Return (bullish_signals, bearish_signals) found in text.

    Tokens are split on non-word characters rather than on whitespace. A plain
    `.split()` leaves punctuation attached, so any signal word that ends a
    sentence or clause never matches the lexicon — measured on the shipped
    corpus, "...policy uncertainty." was silently dropped, which is why the
    score read +1.00 (purely bullish) on a passage carrying both signals.
    """
    words = {w for w in re.split(r"[^a-z]+", text.lower()) if w}
    bullish = sorted(words & _BULLISH_WORDS)
    bearish = sorted(words & _BEARISH_WORDS)
    return bullish, bearish


class MarketSentimentAnalyzeNode(FunctionNode):
    """Analyze market sentiment from Kronos-retrieved financial documents.

    Aggregates sentiment signals across all retrieved documents and computes:
    - overall: 'bullish' | 'bearish' | 'neutral' | 'mixed'
    - score: float in [-1.0, 1.0], positive = bullish
    - bullish_signals: list of detected bullish keyword signals
    - bearish_signals: list of detected bearish keyword signals

    Production implementation uses the Kronos foundation model's built-in
    financial sentiment classification head (trained on Bloomberg/Reuters).

    Security: ANONYMOUS — inner domain node; trust validated by outer pre_process.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieval_results = state.get("retrieval_results", [])

        if not retrieval_results:
            emit_trace_event(
                "sentiment_analysis_skipped",
                {"reason": "no_retrieval_results"},
                state,
            )
            return {
                "sentiment_scores": {
                    "overall": "neutral",
                    "score": 0.0,
                    "bullish_signals": [],
                    "bearish_signals": [],
                },
                "status": AgentStatus.SUCCESS.value,
            }

        all_bullish: list[str] = []
        all_bearish: list[str] = []

        for doc in retrieval_results:
            content = doc.get("content", "") if isinstance(doc, dict) else ""
            b, br = _score_text(content)
            all_bullish.extend(b)
            all_bearish.extend(br)

        # Deduplicate while preserving order
        unique_bullish = list(dict.fromkeys(all_bullish))
        unique_bearish = list(dict.fromkeys(all_bearish))

        bull_count = len(unique_bullish)
        bear_count = len(unique_bearish)
        total = bull_count + bear_count

        if total == 0:
            overall = "neutral"
            score = 0.0
        else:
            raw_score = (bull_count - bear_count) / total
            score = round(raw_score, 3)
            if score > 0.25:
                overall = "bullish"
            elif score < -0.25:
                overall = "bearish"
            elif score != 0.0:
                overall = "mixed"
            else:
                overall = "neutral"

        sentiment_scores = {
            "overall": overall,
            "score": score,
            "bullish_signals": unique_bullish,
            "bearish_signals": unique_bearish,
        }

        emit_trace_event(
            "sentiment_analysis_complete",
            {
                "overall": overall,
                "score": score,
                "bull_signal_count": bull_count,
                "bear_signal_count": bear_count,
                "doc_count": len(retrieval_results),
            },
            state,
        )

        return {
            "sentiment_scores": sentiment_scores,
            "status": AgentStatus.SUCCESS.value,
        }
