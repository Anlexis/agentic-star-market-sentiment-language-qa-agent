"""AgentCore Platform v1.0"""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings [A1]
#  - Read input_context via state.get("input_context", {}) — read-only [C1]
#  - Never import from mediator/, api/, or other agents

import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result
from src.services.source_disclosure import source_label

# Closed-set notice used whenever no answer may be released. Deliberately
# TRUTHY: AgentBaseGraph.get_output() returns `formatted_output or result`, so
# a falsy replacement re-opens the fallback and ships the un-gated `result`.
_NO_OUTPUT_NOTICE = "No response could be produced for this request."
_WITHHELD_NOTICE = "Response withheld: output failed the credential safety check."

# ---------------------------------------------------------------------------
# S-3 LOCAL credential patterns — labelled secrets of the `name: value` shape.
#
# These are the UNION partner of the framework detector, never its replacement.
# The framework's patterns describe credential FORMATS (sk_live_, sk-, eyJ,
# AKIA, Bearer, db URIs) and match none of the shapes below; this set describes
# LABELS and matches none of the framework's. Delegating wholly to the
# framework would make this gate strictly narrower while looking like a
# tightening — wider is safe, narrower is a bypass.
#
# `password` / `passwd` / `pwd` / `credential` / `access_key` / `private_key`
# were absent from the original alternation, so `password=<value>` was rendered
# to the caller verbatim. Measured end-to-end through /invoke before the fix.
# ---------------------------------------------------------------------------
_LOCAL_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key"
    r"|password|passwd|pwd|credential|token|secret|bearer)\b\s*[:=]\s*\S+"
)


# S-3: module-level output credential scan helper.
# Called from execute() to check the assembled response before returning it.
def _security_gate_output(text: str) -> str:
    """Redact labelled-secret material from the response text.

    This is the LOCAL half of the S-3 union and it REDACTS, because a labelled
    secret is usually a fragment of an otherwise useful answer. The framework
    half is not applied here: a credential FORMAT reaching the output is not a
    fragment to tidy away but a signal that the answer cannot be released, so
    execute() withholds on it rather than editing it (see below).
    """
    return _LOCAL_SECRET_RE.sub("[REDACTED]", text)


class PostProcessNode(FunctionNode):
    """Format and finalize the financial Q&A response.

    Responsibilities:
    - Read the assembled `result` from the outer state (set by MarketQAGraphNode.merge_output)
    - Apply the S-3 output scan as a UNION of two layers: local label patterns
      (redacted in place) and the framework credential detector (the floor)
    - Withhold the answer and clear every output-bearing field when the
      framework detector matches
    - Emit audit trail event (S-4)
    - Populate formatted_output for FinalizeNode (always truthy)
    """

    # S-1: ANONYMOUS — called after pre_process trust gate validated the caller.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("result", "")

        if not result:
            emit_trace_event(
                "post_process_empty_result",
                {"session_id": state.get("session_id", "")},
                state,
            )
            return {
                # Truthy on purpose — an empty string re-opens get_output()'s
                # `formatted_output or result` fallback.
                "formatted_output": _NO_OUTPUT_NOTICE,
                "result": "",
                "status": AgentStatus.SUCCESS.value,
            }

        # S-3: domain output credential scan before returning to caller
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=result,
            domain="FIN FinancialMarketSentimentQAAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(result, str) and not detect_credentials(_security_gate_output(str(result + _review))):
            result = result + _review

        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        result = result + source_label(state)
        safe_output = _security_gate_output(str(result))

        # Framework FLOOR — the second half of the union. Any format the
        # platform's own detector blocks means the answer is not releasable, so
        # nothing is released and EVERY output-bearing field is cleared.
        #
        # Neither half subsumes the other: the local patterns describe LABELS
        # (`password=…`) and match none of the framework's formats; the
        # framework's describe FORMATS (sk_live_, sk-, eyJ, AKIA, Bearer, db
        # URIs) and match none of the labels. Replacing one with the other
        # would narrow the gate while looking like a tightening.
        #
        # Clearing, not raising: a raise inside post-process makes the wrapper
        # return a bare ERROR partial that DISCARDS whatever this node returned,
        # and get_output()'s `formatted_output or result` fallback then ships
        # state["result"] — the un-gated inner answer — inside the error
        # envelope. Raising is not containment.
        if detect_credentials(safe_output):
            emit_trace_event(
                "post_process_output_withheld",
                {
                    "session_id": state.get("session_id", ""),
                    "reason": "framework_credential_pattern_in_output",
                },
                state,
            )
            return {
                "formatted_output": _WITHHELD_NOTICE,
                "result": "",
                "status": AgentStatus.ERROR.value,
                "error_log": ["PostProcessNode [S-3]: output withheld by credential gate"],
            }

        # S-4: audit trail — log response delivery
        emit_trace_event(
            "financial_response_delivered",
            {
                "response_length": len(safe_output),
                "session_id": state.get("session_id", ""),
                "has_citations": bool(state.get("validated_citations")),
            },
            state,
        )

        return {
            "formatted_output": safe_output,
            "status": AgentStatus.SUCCESS.value,
        }
