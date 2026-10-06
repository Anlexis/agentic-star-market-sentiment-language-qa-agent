"""AgentCore Platform v1.0 — FIN-C2-104 outer graph."""

# Cat 2 — AgentBaseGraph (outer) + MarketQAGraphNode (main slot) + inner BaseGraph
#
# Architecture:
#   Outer backbone: initialize → pre_process → main(MarketQAGraphNode)
#                              → post_process → finalize
#   Inner workflow: MarketSentimentWorkflowGraph (6-step linear domain pipeline)
#
# Class name: FinancialMarketSentimentQAAgent
#   Must match config/agent.yaml class: and src/api/server.py import exactly.
#
# Rules:
#   ✅ super().register_nodes() called in outer graph (fills initialize + finalize)
#   ✅ MarketQAGraphNode assigned to self._nodes["main"]
#   ✅ add_edges() is NOT overridden (backbone wiring belongs to the framework)
#   ✅ Inner graph instantiated via get_subgraph() with the manifest config
#      forwarded by _parent_config() (inner domain NODES still take no ctor args)

import os
from typing import Any, ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.graph.base_graph import BaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

# Runtime parameters — config/config.yaml at the repo root (three levels up from
# this file: src/graph/graph.py -> src/graph -> src -> <repo root>).
#
# NOT config/agent.yaml. agent.yaml is the AgentRegistry MANIFEST and its keys sit
# at ROOT level with no `agent:` block, so the old `agent.config` read returned {}
# and every declared runtime value was silently dead. The runtime knobs live in
# config/config.yaml, which AgentRegistry passes to the graph as Graph(config=...).
_RUNTIME_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)

# config/config.yaml key -> the key the inner-graph consumer reads.
# `timeout_s` is the manifest-era rename of `timeout_seconds`; consumers still
# validate the long name, so the mapping is applied on load rather than pushing
# the rename down into every reader.
_RUNTIME_KEY_ALIASES = {"timeout_s": "timeout_seconds"}


def load_runtime_config() -> dict[str, Any]:
    """Return the runtime parameters declared in config/config.yaml, VERBATIM.

    This is the mapping AgentRegistry reads and hands to the graph as
    ``Graph(config=...)``, so it is also what the standalone adapter
    (src/api/server.py) must pass — keys exactly as declared, no renaming.
    `max_retry` is consumed by AgentBaseGraph itself (`_validate_config()` and
    the RETRY branch of `route()`), which reads the declared name.

    Best-effort: a missing, unreadable or unparseable file yields ``{}`` so graph
    construction never breaks — the graph then runs on its framework defaults.
    PyYAML is imported lazily: it is a framework runtime dependency, so importing
    it on demand avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            return {}
        return loaded
    except Exception:
        return {}


def _load_runtime_config() -> dict[str, Any]:
    """Alias-mapped view of the declared config, for the INNER-graph consumer.

    Same values as load_runtime_config(), with `timeout_s` renamed to
    `timeout_seconds` (see _RUNTIME_KEY_ALIASES). The outer graph must NOT be
    built from this view — it would no longer match what the registry supplies.
    """
    return {_RUNTIME_KEY_ALIASES.get(k, k): v for k, v in load_runtime_config().items()}


class MarketQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` backbone slot.

    Wraps MarketSentimentWorkflowGraph (the inner 6-step domain workflow).
    Called after PreProcessNode has validated the query and trust level.

    get_subgraph()   — instantiate and return MarketSentimentWorkflowGraph
    extract_input()  — pull validated_input from outer state
    merge_output()   — map inner graph output → outer state result field
    """

    # ANONYMOUS: outer backbone already gated by pre_process (VERIFIED_EXTERNAL).
    # GraphNode.execute() passes the outer InvocationContext into the subgraph
    # unchanged — inner nodes are ANONYMOUS so they accept any caller level.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> "BaseGraph":
        """Instantiate and return the inner MarketSentimentWorkflowGraph.

        The inner graph is constructed with the declared runtime config from
        _parent_config(); previously it was constructed with no config at all,
        so config/config.yaml's declared values never reached it.
        """
        from src.graph.domain_workflow_graph import MarketSentimentWorkflowGraph

        return MarketSentimentWorkflowGraph(config=self._parent_config())

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared runtime config to the inner graph.

        config/config.yaml declares max_retry and timeout_s (aliased on load to
        timeout_seconds, the name consumers validate). Without this hook the
        inner MarketSentimentWorkflowGraph was built with an empty config, so
        every declared setting was dead on arrival. The non-None declared
        settings are exposed under the LangGraph ``configurable`` key, which is
        where BaseGraph consumers look for them.

        Degrades to ``{"configurable": {}}`` when the file is missing or
        malformed — _load_runtime_config() swallows the error and returns {}
        rather than raising during graph construction.

        Scope note: the current inner nodes are deterministic and read no
        configuration; max_retry / timeout_seconds are runtime knobs consumed
        above the domain nodes. This hook restores the declared-config path so
        the declared values reach the inner graph; no key is invented here to
        create a consumer that does not exist.
        """
        config = _load_runtime_config()
        declared = {
            "max_retry": config.get("max_retry"),
            "timeout_seconds": config.get("timeout_seconds"),
        }
        return {"configurable": {k: v for k, v in declared.items() if v is not None}}

    def extract_input(self, state: AgentState) -> str:
        """Return the validated user query to pass into the inner graph."""
        return str(state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state.

        sub_result is shaped by MarketSentimentWorkflowGraph.get_output().
        Returns only changed keys — never the full state.
        """
        return {
            "result": sub_result.get("output"),
            "status": sub_result.get("status"),
        }


class FinancialMarketSentimentQAAgent(AgentBaseGraph):
    """Cat 2 outer graph — FIN-C2-104 Financial Market Intelligence & Sentiment Q&A.

    Wraps the 6-step Kronos-powered domain workflow inside the standard
    5-node backbone. FSA MRM audit trail enforced at pre_process layer.

    Class name must match:
      config/agent.yaml   class: FinancialMarketSentimentQAAgent
      src/api/server.py   from src.graph.graph import FinancialMarketSentimentQAAgent
    """

    @property
    def name(self) -> str:
        return "FinancialMarketSentimentQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Register backbone + domain main-slot node.

        super() fills: initialize (InitializeNode), finalize (FinalizeNode).
        """
        super().register_nodes()
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MarketQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.
