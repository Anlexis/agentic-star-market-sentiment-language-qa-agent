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

# FSA MRM — patterns indicative of material non-public information (MNPI) queries.
# Production implementations consult a continuously updated regulatory pattern library.
_INSIDER_PATTERNS = [
    re.compile(r"\bmnpi\b", re.IGNORECASE),
    re.compile(r"non[\s-]?public\s+information", re.IGNORECASE),
    re.compile(r"material\s+non[\s-]?public", re.IGNORECASE),
    re.compile(r"\binsider\s+trad(?:ing|e)\b", re.IGNORECASE),
    re.compile(r"undisclosed\s+(?:merger|acquisition|takeover)", re.IGNORECASE),
    re.compile(r"pre[\s-]?announcement\s+(?:earnings|results)", re.IGNORECASE),
    re.compile(r"leaked?\s+(?:earnings|results|financials)", re.IGNORECASE),
    re.compile(r"before\s+(?:announcement|disclosure|release)\b", re.IGNORECASE),
]


class InsiderPatternFilterNode(FunctionNode):
    """S-1: Detect FSA MRM-regulated insider information patterns in the query.

    Scans the raw query text for patterns indicative of material non-public
    information (MNPI) requests. If a pattern is detected:
    - Sets insider_risk_flag = True
    - Returns AgentStatus.ERROR.value to short-circuit the pipeline
    - Emits a high-severity FSA audit event (S-4)

    Security: ANONYMOUS — inner domain node; trust already validated by
    outer pre_process (VERIFIED_EXTERNAL). S-1 concern here is content
    compliance, not caller identity.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        validated_input = state.get("validated_input", state.get("user_input", ""))
        parsed_query = state.get("parsed_query", {})
        raw_query = parsed_query.get("raw_query", validated_input)

        # Check for insider information patterns
        matched_pattern = None
        for pattern in _INSIDER_PATTERNS:
            if pattern.search(raw_query):
                matched_pattern = pattern.pattern
                break

        if matched_pattern:
            # S-4: high-severity FSA MRM compliance event
            emit_trace_event(
                "insider_pattern_detected",
                {
                    "pattern": matched_pattern,
                    "query_length": len(raw_query),
                    "session_id": state.get("session_id", ""),
                    "severity": "HIGH",
                },
                state,
            )
            return {
                "insider_risk_flag": True,
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "InsiderPatternFilterNode: query contains patterns indicative of "
                    "material non-public information (MNPI). Request blocked per FSA MRM policy."
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (
                    "InsiderPatternFilterNode: query contains patterns indicative of material non-public information (MNPI). Request blocked per FSA MRM policy."
                ),
            }

        emit_trace_event(
            "insider_pattern_check_passed",
            {
                "query_length": len(raw_query),
                "patterns_checked": len(_INSIDER_PATTERNS),
            },
            state,
        )

        return {
            "insider_risk_flag": False,
            "status": AgentStatus.SUCCESS.value,
        }
