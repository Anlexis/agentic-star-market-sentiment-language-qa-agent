# Market Sentiment & Language Q&A Agent

AI agent for answering financial market sentiment and language intelligence questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific, multi-step pipeline)
> **Industry**: Finance
> **Template ID**: FIN-C2-104

## Overview

A question-answering agent for financial document corpora — earnings reports, macro commentary
and credit profiles — aimed at asset managers, securities firms and investor advisory teams.

The input is a natural-language question. The agent parses it into intent, keywords, asset class
and company; screens it for material non-public information (MNPI) patterns *before* any retrieval
runs; retrieves the relevant documents; scores market sentiment across them; validates every
citation against the retrieved sources; and returns a grounded answer with its sources attached,
behind an audit trail sized for financial model-risk-management review.

Answering questions is the whole scope. The agent does not place trades, move positions or write
to any downstream system.

The shipped pipeline is **deterministic and network-free**: no node makes an outbound call, reads
a credential or invokes a language model. Retrieval serves a small seeded corpus held in the
repository, and the parsing, screening, sentiment and citation steps are in-repo implementations
of the same shape. That is the part you are expected to replace — see *Customising* below.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## How it works

The agent is a five-node backbone (`initialize → pre_process → main → post_process → finalize`)
whose `main` slot wraps a six-step domain workflow:

```
query_parse → insider_filter → kronos_retrieve → sentiment_analyze → citation_gate → response_format
```

The ordering carries the compliance property, not just the throughput: screening runs before
retrieval so a prohibited question never reaches the corpus, and the citation gate runs before
formatting so an unverifiable source is dropped before it can be quoted back to the reader. A
screening match short-circuits the rest of the pipeline outright.

Input validation, the output credential gate and audit-event emission are enforced on every
invocation, independently of which node produced the answer.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the seeded corpus in `src/nodes/kronos_rag_retrieve_node.py` with your own retrieval
   back end. `src/services/service.py` is the seam reserved for that integration.
3. Replace the heuristics in `src/nodes/` with the models you actually use — query parsing,
   sentiment scoring and the trusted-source list are all deliberately simple placeholders.
4. Review the screening pattern set in `src/nodes/insider_pattern_filter_node.py` against the
   regulatory regime you operate under. The shipped list is illustrative, not exhaustive.
5. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
