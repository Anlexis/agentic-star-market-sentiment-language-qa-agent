# Test Specification — FIN-C2-104 FinancialMarketSentimentQAAgent

## Test Strategy
- Coverage target: 80%+
- Test types: Unit (domain nodes) / Hardening contract / Proof-of-Boundary
  (security + data + backbone order + end-to-end through the ASGI entry point)
- Framework: pytest 9.0.3
- SDK mode: real (`agenticstar-agentcore` wheel installed by CI from the registry;
  the spec is carried centrally in `AGENTCORE_WHEEL_SPEC`)

> **Node-level ≠ deployed.** Most tables below exercise nodes in isolation. Only
> the end-to-end table drives the FastAPI app, its Bearer auth boundary and the
> declared trust level — which is the only place a "the deployed agent can serve
> a request" claim can come from.

## Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Status |
|-------|------|----------------|--------|
| TC-01 | State contract: flat TypedDict (no Pydantic/dataclass) | Type check pass; test_state_safety.py: 0 violations | PASS (test_pb_s3_data.py::TestStateCredentialSafety) |
| TC-02 | empty/whitespace user_input rejected by PreProcessNode | AgentStatus.ERROR + error_log populated | PASS (test_agent.py::TC-2) |
| TC-03 | No JWT/Credential in State TypedDict | CI gate-credential-scan: 0 violations; state.py field scan: 0 violations | PASS (test_pb_s3_data.py) |
| TC-04 | InvocationContext not in State TypedDict | test_state_safety.py: InvocationContext not in annotations | PASS (test_state_safety.py) |
| TC-05 | S-4: no duplicate lifecycle events in execute() — node_start/node_complete absent from execute() body | 0 duplicates (framework emits them; our execute() emits only domain events) | PASS (coe_preflight S-4 check) |
| TC-06 | _security_gate_input() not overridden on FunctionNode subclasses | No instance-method override; @final enforced by framework | PASS (coe_preflight gate-hook check) |
| TC-07 | _security_gate_output() not overridden on FunctionNode subclasses | No instance-method override; @final enforced by framework | PASS (coe_preflight gate-hook check) |
| TC-08 | required_trust_level enforced: pre_process=VERIFIED_EXTERNAL, inner nodes=ANONYMOUS | Trust gate fires correctly; test_pb_s1_security.py confirms | PASS (test_pb_s1_security.py::TC-8) |
| TC-09 | S-2 extra input gate on InsiderPatternFilterNode (MNPI detection) | MNPI query returns ERROR; audit event emitted with HIGH severity | PASS (test_pb_s1_security.py::TC-9) |
| TC-10 | S-3 domain output gate: PostProcessNode redacts labelled secrets and withholds on framework-detectable formats | Label pattern redacted to [REDACTED]; framework format → ERROR with `result` cleared; clean text unchanged | PASS (test_pb_s3_data.py::TC-10, test_refit_hardening.py::TestS3GateIsAUnion) |
| TC-11 | S-4: emit_trace_event() in every node execute() — at least 1 domain event per node | 9 nodes × ≥1 event each | PASS (coe_preflight S-4 check) |

## Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | File |
|-------|----------|------|----------------|------|
| PB-1 | BaseNode → EventEmitter | emit_trace_event fires on every node invocation path | 9 nodes × emit verified | coe_preflight.py #3 |
| PB-2 | State serialization | post-invoke State is primitives only; no Pydantic/dataclass | TypedDict only; test_state_safety.py | test_state_safety.py |
| PB-3 | Level 2 → External service | Kronos KB retrieval returns results (stub in test; real Kronos in STG) | retrieval_results populated | test_agent.py (via full invoke) |
| PB-4 | Import isolation | No Level 0 (agenticstar-platform) imports in src/ | AST scan: 0 violations | test_import_isolation.py |
| PB-5 | Checkpoint safety | No JWT/Pydantic in State checkpoint | Inspection pass; state.py scan clean | test_state_safety.py |
| PB-6 | Backbone invoke order | FinancialMarketSentimentQAAgent().invoke(VERIFIED_EXTERNAL) → node_history == [InitializeNode, PreProcessNode, MarketQAGraphNode, PostProcessNode, FinalizeNode] | Order verified; output key present | test_pb_invoke_order.py |

## Security Boundary Tests (PoB-S1)

| TC-ID | Gate | Test | Expected Result | File |
|-------|------|------|----------------|------|
| S1-01 | Trust gate | PreProcessNode.required_trust_level == VERIFIED_EXTERNAL | Static assertion | test_pb_s1_security.py |
| S1-02 | Inner node trust | All 6 inner domain nodes declare ANONYMOUS (not INTERNAL) | Static assertion (6 nodes) | test_pb_s1_security.py |
| S1-03 | MNPI detection | Query with "insider trading" pattern → ERROR + insider_risk_flag=True | InsiderPatternFilterNode blocks query | test_pb_s1_security.py |
| S1-04 | Clean query pass | Public-information financial query → insider_risk_flag=False | InsiderPatternFilterNode passes query | test_pb_s1_security.py |
| S1-05 | MNPI audit event | HIGH-severity emit_trace_event with pattern field (no raw query text) | Audit payload masking verified | test_pb_s1_security.py |

## Data Boundary Tests (PoB-S3)

| TC-ID | Gate | Test | Expected Result | File |
|-------|------|------|----------------|------|
| S3-01 | Citation gate | Trusted source (Kronos KB) retained by CitationGateNode | 1 citation retained | test_pb_s3_data.py |
| S3-02 | Citation gate | Unverifiable source stripped by CitationGateNode | 0 citations retained | test_pb_s3_data.py |
| S3-03 | Mixed citations | Only trusted sources survive from mixed set | 1/2 retained | test_pb_s3_data.py |
| S3-04 | S-3 credential scan | API key pattern in result → [REDACTED] | Credential redacted | test_pb_s3_data.py |
| S3-05 | S-3 clean pass | Clean financial narrative passes S-3 gate unchanged | Text unchanged | test_pb_s3_data.py |
| S3-06 | State safety | No credential-like field names in State TypedDict | 0 violations (AST scan) | test_pb_s3_data.py |

## Business Logic Tests

| TC-ID | Test | Input | Expected Result | File |
|-------|------|-------|----------------|------|
| BL-01 | Full pipeline happy path | Financial query for Japanese equities | AgentStatus.SUCCESS; output non-null | test_agent.py::TC-1 |
| BL-02 | Empty input rejected | user_input="" | AgentStatus.ERROR | test_agent.py::TC-2 |
| BL-03 | MNPI query blocked | "insider trading before announcement" | AgentStatus.ERROR; insider_risk_flag=True | test_agent.py::TC-3 |
| BL-04 | Query parsed correctly | "What is the market sentiment for NISA equity funds?" | intent=sentiment_query; asset_class=equity | test_agent.py::TC-4 |
| BL-05 | Unverifiable citation stripped | Unrecognised source document | validated_citations=[] | test_agent.py::TC-5 |
| BL-06 | Response formatted with sentiment and citations | Bullish sentiment + Bloomberg source | Response includes 'bullish'; cites Bloomberg | test_agent.py::TC-6 |

## Hardening Contract Tests (tests/unit/test_refit_hardening.py)

Each pins a property that was **measured to be broken or absent** before, so a
regression fails here rather than in production.

| TC-ID | Property | Was | File |
|-------|----------|-----|------|
| H-01 | `config/config.yaml` values reach the inner graph, `timeout_s` aliased to `timeout_seconds` | Reader looked for an `agent.config` block the flat manifest does not have → `{}`, every declared value dead | test_refit_hardening.py::TestRuntimeConfigReachesTheGraph |
| H-08 | The standalone adapter constructs the OUTER graph with the declared config, and enforces `timeout_s` as the `/invoke` deadline | `src/api/server.py` built the agent bare → `self.config == {}`; `AgentBaseGraph.route()` silently used its own `max_retry` fallback and no request deadline existed | test_refit_hardening.py::TestStandaloneEntryPointUsesDeclaredRuntimeConfig |
| H-02 | Chat-template control tokens refused as a class, raw and markup-spliced | `<<SYS>>` passed the screen and was echoed to the caller | test_refit_hardening.py::TestControlTokenScreen |
| H-03 | S-3 gate is a union: neither the local labels nor the framework detector subsumes the other | Local set missed all 5 framework formats; `password=` matched neither and shipped verbatim | test_refit_hardening.py::TestS3GateIsAUnion |
| H-04 | A framework-detectable credential is withheld and every output-bearing field cleared; replacements are truthy | Empty-string `formatted_output` re-opened `get_output()`'s `formatted_output or result` fallback | test_refit_hardening.py::TestContainmentOnViolation |
| H-05 | Echoed caller text cannot manufacture report structure; length-capped | Caller newlines rendered a second, forged `**Sources:**` block above the real one | test_refit_hardening.py::TestEchoIsInert |
| H-06 | The redaction sentinel is neither a keyword nor rendered as caller text | — (introduced with the echo hardening) | test_refit_hardening.py::TestRedactionSentinelIsNotContent |
| H-07 | The local PII layer covers the framework's Japanese-script gap; ordinary JA and numeric text is not false-flagged | — (pins the upstream gap) | test_refit_hardening.py::TestLocalPiiLayerCoversTheFrameworkGap |
| H-08 | The sentiment metric moves with the corpus; a sentence-final signal is detected | `.split()` left punctuation attached, so `uncertainty.` never matched and the score read `+1.00` on a passage carrying both signals | test_refit_hardening.py::TestSentimentMetricIsLive |
| H-09 | The conditional-route callable is annotated with the graph's own `State`; both branches reachable | Annotated with the framework `AgentState`, which projects away every domain field | test_refit_hardening.py::TestRouteCallableSchema |

## End-to-End Boundary Tests (tests/proof_of_boundary/test_pb_invoke_e2e.py)

Driven through the real ASGI `/invoke` with Bearer auth at the declared trust level.

| E2E-ID | Test | Expected Result |
|--------|------|----------------|
| E2E-01 | `/health` responds | `status: ok` |
| E2E-02 | Unauthenticated and wrong-token callers | HTTP 401, generic body |
| E2E-03 | Authenticated caller | `status: success`, non-empty report with a `**Sources:**` block |
| E2E-04 | Backbone ran end to end | `node_history == [Initialize, PreProcess, MarketQAGraphNode, PostProcess, Finalize]` |
| E2E-05 | Injection refused (control tokens + directive phrases) | `status: error`, no output published — asserted on BEHAVIOUR, never on framework wording |
| E2E-06 | MNPI query blocked before retrieval | `status: error`, no output |
| E2E-07 | No credential reaches the caller (5 formats + one labelled secret) | The value appears nowhere in the envelope |
| E2E-08 | Raw PII never reaches the output | Value absent from the envelope (removed by the platform gate, not by this template) |
| E2E-09 | Unspaced Japanese personal-number form | `status: error` — this template's own scan, which the framework detector misses |
| E2E-10 | Forged source block cannot be manufactured | Exactly one `**Sources:**` block; caller text stays inside the echo line |
| E2E-11 | Report size does not track input size | 35,000-char input does not produce a proportional report |
| E2E-12 | Every rendered citation comes from a trusted source | Citation invariant holds on the deployed path |
| E2E-13 | No monetary aggregate is rendered | Pins the "precision grid n/a" position so a future change confronts it |

## Test Execution Summary

- Execution environment: CI installs the real `agenticstar-agentcore` wheel from the
  registry (spec centrally in `AGENTCORE_WHEEL_SPEC`); local verification runs against
  the vendored wheel via the campaign harness.
- Test files: `tests/unit/test_agent.py`, `tests/unit/test_refit_hardening.py`,
  `tests/unit/test_s1_trust_gate.py`, `tests/unit/test_framework_compliance_tc06_tc07.py`,
  `tests/proof_of_boundary/{test_pb_invoke_order, test_pb_invoke_e2e, test_pb_s1_security,
  test_pb_s3_data, test_import_isolation, test_state_safety, test_server_boot,
  test_pb7_hitl_interrupt_propagation}.py`
- Suite total: **157 tests — 156 passed, 1 skipped, 0 failed**
- STG functional test: live queries via the runbook (docs/07_operation_guide.md)
