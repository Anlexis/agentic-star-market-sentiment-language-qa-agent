"""AgentCore Platform v1.0"""

# AgentRegistry entry point — re-export the agent class from this package.
#
# config/agent.yaml declares:
#     module: "src.graph"
#     class:  "FinancialMarketSentimentQAAgent"
#
# AgentRegistry resolves that entry point with
# getattr(import_module("src.graph"), "FinancialMarketSentimentQAAgent").
# This module previously contained only a docstring, so the declared class was
# never bound in the `src.graph` namespace and the registry load raised
# AttributeError at runtime. The test suite imports `src.graph.graph` directly
# and never exercises the manifest path, so CI stayed green while a real
# registry-driven load failed. Re-exporting the class here fixes the load
# without changing the manifest.

from src.graph.graph import FinancialMarketSentimentQAAgent

__all__ = ["FinancialMarketSentimentQAAgent"]
