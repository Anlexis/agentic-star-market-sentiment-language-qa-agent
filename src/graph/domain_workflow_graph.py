"""AgentCore Platform v1.0 — FIN-C2-104 inner domain workflow graph."""

# Cat 2 inner graph — MarketSentimentWorkflowGraph
#
# Called by MarketQAGraphNode.get_subgraph() in graph.py.
# Inherits BaseGraph for a fully custom 6-node linear topology.
#
# Pipeline:
#   START → query_parse → insider_filter → kronos_retrieve
#         → sentiment_analyze → citation_gate → response_format → END
#
# All 7 BaseGraph abstract methods are implemented.
# register_nodes() instantiates every domain node with NO constructor args
# (SDK-v1 requirement — FunctionNode subclasses have no __init__, and
# execute(self, state) -> dict takes no config parameter).
#
# The graph itself IS constructed with the declared runtime parameters:
# MarketQAGraphNode._parent_config() (graph.py) loads config/config.yaml and
# passes the declared settings in under config["configurable"], reachable here
# as self.config.
#
# All domain nodes declare:
#   required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS
# (inner nodes must never be INTERNAL — outer InvocationContext is passed
# unchanged by GraphNode.execute(); a trust order mismatch DENIES external callers)

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.citation_gate_node import CitationGateNode
from src.nodes.insider_pattern_filter_node import InsiderPatternFilterNode
from src.nodes.kronos_rag_retrieve_node import KronosRAGRetrieveNode
from src.nodes.market_sentiment_analyze_node import MarketSentimentAnalyzeNode
from src.nodes.query_parse_node import QueryParseNode
from src.nodes.response_format_node import ResponseFormatNode
from src.schemas.state import State


class MarketSentimentWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for FIN-C2-104.

    Orchestrates the 6-step financial market intelligence pipeline:
      1. QueryParseNode       — extract intent, keywords, asset_class, company
      2. InsiderPatternFilterNode — S-1 FSA MRM: block MNPI queries
      3. KronosRAGRetrieveNode   — retrieve from Kronos financial KB
      4. MarketSentimentAnalyzeNode — analyze market sentiment
      5. CitationGateNode        — S-3: validate/strip citations
      6. ResponseFormatNode      — assemble final formatted response

    Called by MarketQAGraphNode.get_subgraph() in graph.py.
    get_output() shapes the sub_result dict consumed by merge_output().
    """

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "market_sentiment_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config keys for this inner graph.

        MarketQAGraphNode._parent_config() forwards the settings declared in
        config/config.yaml under config["configurable"]; all of them are
        optional, so validation is permissive rather than raising ConfigError.
        """
        pass

    # ── Node registration ─────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all domain nodes with NO constructor args (SDK-v1).

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize here (outer backbone concern).
        """
        self._nodes["query_parse"] = QueryParseNode()
        self._nodes["insider_filter"] = InsiderPatternFilterNode()
        self._nodes["kronos_retrieve"] = KronosRAGRetrieveNode()
        self._nodes["sentiment_analyze"] = MarketSentimentAnalyzeNode()
        self._nodes["citation_gate"] = CitationGateNode()
        self._nodes["response_format"] = ResponseFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear 6-step domain workflow topology.

        InsiderPatternFilterNode can return ERROR (MNPI detected); the route()
        method handles the conditional branch to END on error.
        """
        self._sg.add_edge(START, "query_parse")
        self._sg.add_edge("query_parse", "insider_filter")
        self._sg.add_conditional_edges("insider_filter", self.route)
        self._sg.add_edge("kronos_retrieve", "sentiment_analyze")
        self._sg.add_edge("sentiment_analyze", "citation_gate")
        self._sg.add_edge("citation_gate", "response_format")
        self._sg.add_edge("response_format", END)

    # ── Routing ───────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Conditional routing after insider_filter.

        If status is ERROR (MNPI pattern detected), short-circuit to END.
        Otherwise proceed to Kronos retrieval.

        Annotated with this graph's OWN State, not the framework AgentState:
        LangGraph reads a path callable's annotation as its input schema and
        projects away every field the annotation does not declare. `status`
        happens to exist on both, so the branch fires today — but any future
        read of a domain field (insider_risk_flag, parsed_query) would have
        seen it as always-absent while the unit suite stayed green.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "kronos_retrieve"

    # ── Output shape ──────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the sub_result dict returned to MarketQAGraphNode.merge_output().

        Prefers formatted_response (set by ResponseFormatNode) over the
        generic result field. The outer merge_output() maps output → result.
        """
        return {
            "output": state.get("formatted_response") or state.get("result", ""),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
