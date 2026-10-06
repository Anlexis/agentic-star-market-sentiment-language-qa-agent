"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.

import asyncio
import os
import secrets
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets import SecretProvider
from framework.secrets.context import bound_secrets
from shared.secrets import InMemoryProvider
from shared.secrets import factory as secrets_factory
from src.graph.graph import FinancialMarketSentimentQAAgent, load_runtime_config

app = FastAPI(title="Agent")

_NAMESPACE = "fin-c2-104"
_AGENT_NAME = "FinancialMarketSentimentQAAgent"

# Runtime configuration — the SAME mapping AgentRegistry hands the graph as
# Graph(config=...) on the platform path (docs/02_design.md "Configuration").
#
# This adapter used to construct the agent bare, so self.config was {} and every
# value declared in config/config.yaml was dead on the standalone path while the
# platform path honoured it. The divergence was invisible because the declared
# max_retry (3) happens to equal AgentBaseGraph.route()'s built-in fallback
# (`self.config.get("max_retry", 3)`) — identical behaviour, for the wrong
# reason. Declaring max_retry: 1 changed nothing here until this wiring existed.
_RUNTIME_CONFIG = load_runtime_config()

agent = FinancialMarketSentimentQAAgent(config=_RUNTIME_CONFIG)
agent.compile()
# Namespace and agent_name match config/agent.yaml id and class.
agent.provision_secrets(secrets_factory(namespace=_NAMESPACE, agent_name=_AGENT_NAME))

# timeout_s is declared runtime config with no framework consumer: the SDK reads
# max_retry / memory_enabled / hitl off self.config, and nothing reads timeout_s
# (AgentStatus.TIMEOUT exists but no config path sets it; RemoteAgentNode carries
# its own unrelated hardcoded _timeout). So the request deadline is the adapter's
# to enforce -- read off the same mapping handed to the graph, not a second file
# read, so the declared value cannot drift between the two.
_REQUEST_TIMEOUT_S = _RUNTIME_CONFIG.get("timeout_s")


def _entrypoint_secrets() -> SecretProvider:
    """Boot-time SecretProvider for the ENTRY-POINT / DEPLOYMENT credential.

    S-5 (secret-provider boundary): the caller-auth token INVOKE_AUTH_TOKEN is a
    *deployment-level* credential — it authenticates the /invoke caller at the
    standalone HTTP boundary, BEFORE any InvocationContext (and thus ctx.secrets)
    exists — so it is a different class of secret from the agent's per-invocation
    secrets, which resolve through ctx.secrets and the namespaced provider.

    The framework SecretProvider is still the sanctioned access boundary: this
    loads the deployment credential from its documented source (the process
    environment set by the deploy job — the staging runbook §5 / deploy/local-stg.yml
    / the pipeline definition deploy-stg exports it, and scripts/stg_invoke_evidence.py
    presents the SAME env value as a Bearer token) INTO an InMemoryProvider once
    at boot — exactly as shared.secrets.factory() loads dotenv values into a
    DotenvProvider — so the request handler reads the token via the provider
    accessor (.get()) rather than a raw os.environ read. Boot-safe: an
    absent/empty token yields a provider whose .get() returns None, preserving
    the "no token set → ANONYMOUS callers" contract with no raise at import.

    PATTERN NOTE (for other templates): entry-point/deployment credentials that
    arrive via the process environment are loaded into an InMemoryProvider here;
    agent secrets that arrive via env/*.env files stay on secrets_factory() and
    are read node-side through ctx.secrets.require().
    """
    values = {}
    token = os.environ.get("INVOKE_AUTH_TOKEN")
    if token:
        values["INVOKE_AUTH_TOKEN"] = token
    return InMemoryProvider(values, namespace=_NAMESPACE, agent_name=_AGENT_NAME)


_ENTRYPOINT_SECRETS = _entrypoint_secrets()


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


def _invoke_agent(user_input: str, ctx: InvocationContext) -> dict[str, Any]:
    """Run the agent with its SecretProvider bound for the calling scope.

    Kept a plain sync function so it is identical whether called inline (no
    declared deadline) or via asyncio.to_thread (deadline declared) — the
    secrets binding then lives in whichever context actually runs the graph.
    """
    with bound_secrets(agent._secrets_provider):
        result: dict[str, Any] = agent.invoke(user_input, ctx=ctx)
        return result


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone/STG caller auth ((internal issue reference removed), the staging runbook §5): when
    # INVOKE_AUTH_TOKEN is set on the server environment, callers that no upstream
    # middleware vouched for (still ANONYMOUS) must present it as a Bearer token
    # and run at VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary (standalone equivalent of
    # platform AuthMiddleware) — a deployment-level caller credential, not an
    # agent secret, so ctx.secrets does not apply (no InvocationContext exists
    # before auth); see the framework rules §3 "Entry-point exception".
    # Without it PreProcessNode's required_trust_level=VERIFIED_EXTERNAL denies
    # every standalone invoke at the S-1 gate (agent status=error).
    #
    # S-5: the token is read through the framework SecretProvider accessor
    # (_ENTRYPOINT_SECRETS.get(), built once at boot from the deployment env),
    # never via a raw os.environ read in the handler.
    expected = _ENTRYPOINT_SECRETS.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    ctx = InvocationContext(
        session_id=req.session_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )
    if _REQUEST_TIMEOUT_S is None:
        # No deadline declared — preserve the unbounded synchronous call.
        return _invoke_agent(req.input, ctx)
    try:
        # to_thread so the deadline can actually fire: agent.invoke() is sync, and
        # awaiting it inline would block the event loop with nothing left to time it
        # out. bound_secrets is entered INSIDE the worker (framework.secrets.context
        # documents this as the sanctioned cross-to_thread shape, as used by the
        # platform's own AgentGateway._dispatch).
        return await asyncio.wait_for(
            asyncio.to_thread(_invoke_agent, req.input, ctx),
            timeout=_REQUEST_TIMEOUT_S,
        )
    except TimeoutError:
        # Bounds the RESPONSE, not the worker: a Python thread cannot be cancelled,
        # so the orphaned invoke runs to completion and its result is discarded.
        # Generic body — same non-disclosure rule as the 401 above.
        raise HTTPException(status_code=504, detail="Request timed out.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "FinancialMarketSentimentQAAgent"}
