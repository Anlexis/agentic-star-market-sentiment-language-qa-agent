# Template Design Specification — FIN-C2-104

## Position in AgentCore Architecture

- **Agent Class**: FinancialMarketSentimentQAAgent
- **L1 Base**: AgentBaseGraph (Cat 2 — outer graph direct L1 inheritance)
- **Pattern**: Cat 2 nested — outer AgentBaseGraph + GraphNode in `main` slot + inner BaseGraph
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: L1 inheritance (Template Method: `execute(self, state: dict) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution)

## Architecture Overview

### Node Configuration

| Node | Responsibility | Input State Keys | Output State Keys | Trust Level |
|------|---------------|-----------------|-------------------|-------------|
| initialize | Session / correlation ID injection | — | session_id, schema_version | default InitializeNode |
| pre_process | Query validation, FSA audit trail, trust gate | user_input | validated_input, enriched_context | VERIFIED_EXTERNAL |
| main (MarketQAGraphNode) | Orchestrates inner domain workflow graph | validated_input | result, status | ANONYMOUS |
| post_process | Output formatting, S-3 credential gate | result | formatted_output | ANONYMOUS |
| finalize | Response metadata assembly | formatted_output | — | default FinalizeNode |

### Inner Domain Workflow — MarketSentimentWorkflowGraph

All inner nodes: `required_trust_level = TrustLevel.ANONYMOUS` (declared, never omitted).

| Node | Responsibility | Input State Keys | Output State Keys |
|------|---------------|-----------------|-------------------|
| QueryParseNode | Extract query intent, asset class, company, keywords | validated_input | parsed_query |
| InsiderPatternFilterNode | S-1 FSA MRM: detect insider information patterns; block if found | parsed_query, validated_input | insider_risk_flag |
| KronosRAGRetrieveNode | Retrieve relevant documents from Kronos financial KB | parsed_query | retrieval_results |
| MarketSentimentAnalyzeNode | Analyze market sentiment signals from retrieved docs | retrieval_results, parsed_query | sentiment_scores |
| CitationGateNode | S-3: validate source citations, remove unverifiable refs | retrieval_results, sentiment_scores | validated_citations |
| ResponseFormatNode | Assemble final financial Q&A response with citations | parsed_query, sentiment_scores, validated_citations | formatted_response |

### Data Flow

```
Outer backbone:
  START → initialize → pre_process → main(MarketQAGraphNode) → post_process → finalize → END
                                          ↓ (RETRY, max 3)
                                       pre_process

Inner domain workflow (inside MarketQAGraphNode.get_subgraph()):
  START → query_parse → insider_filter → kronos_retrieve → sentiment_analyze → citation_gate → response_format → END
```

### State Definition

| Field | Type | Purpose | Set By |
|-------|------|---------|--------|
| validated_input | Optional[str] | Sanitized user query | PreProcessNode |
| enriched_context | Optional[dict] | Channel + source metadata | PreProcessNode |
| parsed_query | Optional[dict] | Structured query: intent, keywords, asset_class, company | QueryParseNode |
| insider_risk_flag | Optional[bool] | True if insider information pattern detected | InsiderPatternFilterNode |
| retrieval_results | Optional[list] | Kronos retrieved docs: [{doc_id, title, content, relevance_score}] | KronosRAGRetrieveNode |
| sentiment_scores | Optional[dict] | {overall, score, bullish_signals, bearish_signals} | MarketSentimentAnalyzeNode |
| validated_citations | Optional[list] | Verified source citations: [{source, title, verified}] | CitationGateNode |
| formatted_response | Optional[str] | Final formatted financial Q&A response with citations | ResponseFormatNode |
| result | Optional[str] | Assembled answer, passed from inner graph via merge_output | MarketQAGraphNode |

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types)
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)

## Security Gate Placement

| Gate | Node | Implementation |
|------|------|---------------|
| S-1: Trust gate | PreProcessNode | `required_trust_level = VERIFIED_EXTERNAL` on outer pre_process |
| S-1: Insider pattern | InsiderPatternFilterNode | Detect FSA-regulated insider info patterns; return ERROR if flagged |
| S-2: Input gate | PreProcessNode | Framework `@final _security_gate_input()` auto-runs (PII masking + injection policy); domain scans run inline in `execute()` — type, empty, control tokens, injection phrases (raw and markup-stripped), PII |
| S-3: Output gate | CitationGateNode | Strip unverifiable citations before response assembly |
| S-3: Output gate | PostProcessNode | Domain union gate: local label patterns redacted in place; framework `detect_credentials` as the floor → withhold and clear. Framework `@final _security_gate_output()` also auto-runs |
| S-4: Audit trail | ALL nodes | `emit_trace_event(event, payload, state)` — 3 positional args, every execute() |
| S-5: Credential-free state | State | No API keys / secrets in State TypedDict |

### S-2 input scan — what each layer actually covers

The framework gate runs **before** this template's node, so the two layers see
different text and neither is redundant.

| Layer | Covers | Does not cover |
|---|---|---|
| Framework `@final _security_gate_input()` | Masks email / credit card / JP phone in `user_input` in place, replacing each with `[MASKED]`; blocks high-confidence injection findings | Unspaced Japanese personal-number forms — its word boundaries are computed over `\w`, which includes Kana and Kanji, so `個人番号1234-5678-9012を確認` yields no findings |
| Template `_scan_pii()` | The Japanese form above, via separator-anchored patterns | Anything the framework already masked — by then the raw value is gone, so this layer cannot be what removes it |
| Template `_scan_injection()` | Chat-template control tokens as a class (`<\|…\|>`, `[INST]`, `<<SYS>>`), directive phrases, SQL and script markers, each screened raw **and** markup-stripped | — |

**Consequence, stated rather than papered over:** for email / card / phone the
personal data is removed by the platform, not by this template. A node-level
test that calls `execute()` directly sees a rejection the deployed path never
reaches; the deployed contract is "the raw value never appears in the output",
and that is what the boundary tests assert.

**Route taken around an internal ticket.** The platform detector also masks ordinary
title-case proper nouns as person names — including two listed issuers named in
this template's own STG sign-off payload. Refusing on the presence of the
sentinel would therefore deny the template's own valid traffic, so the sentinel
is accepted and rendered harmless instead: it is dropped from keyword
extraction (a masked value must not drive retrieval as though it were a domain
term the caller asked about) and preserved as a visible redaction marker in the
echoed query rather than being stripped down to the bare word `MASKED`.

### S-3 output gate — a union, deliberately

Neither half of the output gate subsumes the other, and swapping one for the
other narrows it while looking like a tightening:

- **Local label patterns** (`password=`, `api_key:`, `private_key:` …) describe
  *labels*. The framework detector matches none of them.
- **Framework `detect_credentials`** describes credential *formats*
  (`sk_live_`, `sk-`, `eyJ`, `AKIA`, `Bearer`, database URIs). The local
  patterns match none of them.

A labelled secret is redacted in place and the answer still ships. A
framework-detectable format means the answer is not releasable: the node
returns ERROR, replaces `formatted_output` with a fixed notice, and **clears
`result`**. Clearing rather than raising is load-bearing — `get_output()`
returns `formatted_output or result`, so a raise inside post-process makes the
wrapper discard this node's return and ship the un-gated inner answer in the
error envelope. For the same reason the empty-result branch returns a truthy
notice rather than `""`.

### Output invariants

- **Citation invariant** (this template's stated output guarantee): every
  rendered citation comes from a trusted-source prefix; unverifiable sources are
  dropped by `CitationGateNode` before assembly.
- **Echo invariant**: caller text interpolated into the report is flattened to a
  single line, stripped of markdown structural characters and length-capped, so
  it cannot manufacture a heading, list item or a second `**Sources:**` block.
- **Monetary precision grid: not applicable.** This template renders no monetary
  aggregates — the report carries a sentiment score, signal words and citation
  titles, no amounts. There is therefore no rounding grid to enforce, and no
  currency-marker snapping grammar is shipped. A boundary test pins the absence
  so that a future change which starts rendering amounts has to confront the
  rounding contract deliberately rather than inheriting silence.

## Runtime Configuration Contract

The manifest is **flat**: `config/agent.yaml` holds AgentRegistry keys at root
level with no `agent:` block. Runtime parameters live separately in
`config/config.yaml`, which AgentRegistry passes to the graph as
`Graph(config=...)`.

| File | Holds | Read by |
|---|---|---|
| `config/agent.yaml` | id, name, namespace, version, category, industry, base_type, class, `required_trust_level`, `requires.secrets` / `requires.extras` | AgentRegistry |
| `config/config.yaml` | `max_retry` | **outer graph** — `AgentBaseGraph._validate_config()` and the RETRY branch of `AgentBaseGraph.route()`, off `self.config` |
| `config/config.yaml` | `max_retry`, `timeout_s` | inner graph — `MarketQAGraphNode._parent_config()` → `config["configurable"]` |
| `config/config.yaml` | `timeout_s` | the `/invoke` request deadline in `src/api/server.py` |

**Both deployments construct the outer graph with this mapping.** On the
platform path AgentRegistry supplies it; on the standalone path
`src/api/server.py` calls `load_runtime_config()` and passes the result to
`FinancialMarketSentimentQAAgent(config=...)`. The standalone adapter formerly
constructed the agent bare, so `self.config` was `{}` and the outer graph ran on
framework defaults — undetected, because the declared `max_retry` (3) is also
`route()`'s built-in fallback. `load_runtime_config()` returns the file verbatim
so the two paths cannot diverge.

`timeout_s` has **no framework consumer** — the SDK reads `max_retry`,
`memory_enabled` and `hitl` off `self.config` and nothing reads `timeout_s`. The
standalone adapter therefore enforces it itself as an `asyncio.wait_for`
deadline around the invoke, answering `504` when it expires. That bounds the
response, not the worker: a Python thread cannot be cancelled, so an overrunning
invoke completes in the background and its result is discarded.

`timeout_s` is aliased to `timeout_seconds` for the **inner** graph only,
because that is the name that consumer validates; the outer graph and the
adapter both read the declared name. `requires.secrets` is `[]` and `requires.extras` is `[]`:
no node calls `ctx.secrets.require()` and no client is constructed, and
declaring an unprovisioned secret would make the agent fail at compile time.

### Sentiment scoring — scope of the shipped corpus

`MarketSentimentAnalyzeNode` scores the **retrieved documents**, never the
caller's question — a user asking "is it bearish?" must not make the market
bearish. The arithmetic is live and moves with the corpus it is given
(strongly-bullish and strongly-bearish document sets score `+1.00` and `-0.75`
respectively). With the seeded two-document corpus that ships here it settles at
`0.00 / neutral`, because that corpus carries one bullish and one bearish signal.
That constancy is a property of the seeded corpus, not of the metric; replacing
the retrieval back end is what makes the score vary per query.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, caller_trust_level) — outer backbone
- [x] S-2: framework `@final _security_gate_input()` auto-runs on FunctionNode subclasses
- [x] S-3: framework `@final _security_gate_output()` auto-runs credential scan on all output dicts
- [x] S-4: `emit_trace_event(event, payload, state)` — 3 positional args, called in every execute()

### Composition Pattern

- **Pattern**: GraphNode (subgraph) — Cat 2 nested
- **GraphNode class**: MarketQAGraphNode (assigned to `main` slot in outer graph)
- **Inner graph**: MarketSentimentWorkflowGraph (BaseGraph, custom linear topology)
- **Error propagation strategy**: propagate (SubgraphError re-raised)

## Import Isolation Confirmation
- [x] Template does not import agenticstar-platform SDK (Level 0)
- [x] Import targets: framework/ and shared/ only (no agents/base/ required)
- [x] No L2 intermediary (VectorRAGAgent etc.) — direct L1 inheritance from AgentBaseGraph / BaseGraph

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed multi-step pipeline; no autonomous loop needed |
| Inner graph parent | BaseGraph | AgentBaseGraph | BaseGraph | Fully custom 6-node linear topology; no standard backbone |
| Composition pattern | Flat Cat-1 slots | GraphNode (Cat 2 nested) | GraphNode (Cat 2 nested) | FIN domain orchestrates 6 sequential steps; Cat 2 required for non-CMN industries |
| Pre_process trust | ANONYMOUS | VERIFIED_EXTERNAL | VERIFIED_EXTERNAL | FSA MRM requirement: only verified external callers may submit financial queries |
| Inner node trust | INTERNAL | ANONYMOUS | ANONYMOUS | Inner nodes receive outer InvocationContext unchanged; INTERNAL would block VERIFIED_EXTERNAL(1) callers |
