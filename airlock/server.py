"""The HTTP face of the gateway.

The completions endpoint speaks the OpenAI request shape on purpose: an
existing application switches to the gateway by changing a base URL, with no
code change. That is the only migration path anyone will actually take.

FastAPI is an optional dependency — the library, and all of its tests, run
without it. Only this module needs it.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .errors import AirlockError, AllProvidersFailed, BudgetExceeded, GuardrailTripped
from .gateway import Gateway
from .types import CompletionRequest, Message, Role


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    """The subset of the OpenAI chat schema this gateway acts on."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=1024, ge=1, le=32_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Not OpenAI's, but the point of the gateway: who is being billed.
    api_key_id: str = "default"
    tags: list[str] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None

    def to_domain(self) -> CompletionRequest:
        return CompletionRequest(
            model=self.model,
            messages=tuple(
                Message(_role(m.role), m.content) for m in self.messages
            ),
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            response_schema=self.response_schema,
            api_key_id=self.api_key_id,
            tags=tuple(self.tags),
        )


def _role(value: str) -> Role:
    try:
        return Role(value.lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown role {value!r}") from exc


def create_app(gateway: Gateway | None = None) -> FastAPI:
    """Build the app. Pass a gateway, or get the offline demo one."""
    if gateway is None:
        from .cli import build_demo_gateway

        gateway, _, _ = build_demo_gateway()

    app = FastAPI(
        title="airlock",
        description="A gateway for LLM traffic: retries, fallback, budgets, guardrails, traces.",
        version="0.1.0",
    )
    app.state.gateway = gateway

    @app.post("/v1/chat/completions")
    async def completions(body: ChatRequest) -> JSONResponse:
        try:
            response = await app.state.gateway.complete(body.to_domain())
        except BudgetExceeded as error:
            # 402 is the honest code: the request was well formed and refused
            # on payment grounds.
            raise HTTPException(status_code=402, detail=str(error)) from error
        except GuardrailTripped as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except AllProvidersFailed as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except AirlockError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        trace = app.state.gateway.traces.recent(1)[0]
        return JSONResponse(
            {
                "id": f"airlock-{trace.id}",
                "object": "chat.completion",
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": response.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                # Everything below is airlock's, not OpenAI's.
                "airlock": {
                    "trace_id": trace.id,
                    "provider": response.provider,
                    "cached": response.cached,
                    "cost_usd": round(response.cost_usd, 6),
                    "latency_ms": round(response.latency_ms, 2),
                    "retries": trace.retries,
                    "guardrails": trace.guardrails,
                    "parsed": response.parsed,
                },
            }
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return app.state.gateway.stats()

    @app.get("/traces")
    async def traces(limit: int = 20) -> dict[str, Any]:
        return {
            "traces": [t.to_dict() for t in app.state.gateway.traces.recent(limit)],
        }

    @app.get("/traces/{trace_id}")
    async def one_trace(trace_id: int) -> dict[str, Any]:
        trace = app.state.gateway.traces.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"no trace {trace_id}")
        return {**trace.to_dict(), "explain": trace.explain()}

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return DASHBOARD

    return app


DASHBOARD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>airlock</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0d1117; --card: #161b22; --line: #30363d;
    --text: #e6edf3; --dim: #8b949e; --ok: #3fb950; --bad: #f85149; --warn: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 24px; }
  h1 { font-size: 18px; margin: 0 0 2px; letter-spacing: .5px; }
  p.sub { color: var(--dim); margin: 0 0 24px; }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          margin-bottom: 24px; }
  .tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
          padding: 14px; }
  .tile .label { color: var(--dim); font-size: 11px; text-transform: uppercase;
                 letter-spacing: .08em; }
  .tile .value { font-size: 22px; margin-top: 6px; }
  h2 { font-size: 13px; color: var(--dim); text-transform: uppercase; letter-spacing: .08em;
       margin: 24px 0 8px; }
  table { width: 100%; border-collapse: collapse; background: var(--card);
          border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line);
           white-space: nowrap; }
  th { color: var(--dim); font-weight: 500; font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  .ok { color: var(--ok); } .bad { color: var(--bad); } .warn { color: var(--warn); }
  .wrap { overflow-x: auto; }
</style>
</head>
<body>
  <h1>airlock</h1>
  <p class="sub">every request, what it cost, and what went wrong</p>
  <div class="grid" id="tiles"></div>
  <h2>providers</h2>
  <div class="wrap"><table id="breakers"></table></div>
  <h2>recent requests</h2>
  <div class="wrap"><table id="traces"></table></div>
<script>
const fmt = (n, d = 2) => Number(n).toFixed(d);
const stateClass = s => s === "closed" ? "ok" : s === "open" ? "bad" : "warn";

function tile(label, value, cls = "") {
  return `<div class="tile"><div class="label">${label}</div>
          <div class="value ${cls}">${value}</div></div>`;
}

function rows(head, body) {
  return `<tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` + body;
}

async function refresh() {
  const [stats, traces] = await Promise.all([
    fetch("stats").then(r => r.json()),
    fetch("traces?limit=15").then(r => r.json()),
  ]);
  const t = stats.traffic, l = stats.latency_ms;
  document.getElementById("tiles").innerHTML = [
    tile("requests", t.requests),
    tile("success", fmt(t.success_rate * 100, 0) + "%", t.success_rate < 0.99 ? "warn" : "ok"),
    tile("cache hits", fmt(t.cache_hit_rate * 100, 0) + "%"),
    tile("spend", "$" + fmt(t.total_cost_usd, 4)),
    tile("p50", fmt(l.p50, 0) + "ms"),
    tile("p95", fmt(l.p95, 0) + "ms"),
    tile("p99", fmt(l.p99, 0) + "ms"),
  ].join("");

  document.getElementById("breakers").innerHTML = rows(
    ["provider", "state", "calls", "failures", "rate"],
    (stats.breakers || []).map(b => `<tr>
      <td>${b.provider}</td>
      <td class="${stateClass(b.state)}">${b.state}</td>
      <td>${b.calls_in_window}</td><td>${b.failures_in_window}</td>
      <td>${fmt(b.failure_rate * 100, 0)}%</td></tr>`).join("")
  );

  document.getElementById("traces").innerHTML = rows(
    ["id", "outcome", "served by", "ms", "cost", "retries", "notes"],
    traces.traces.map(x => `<tr>
      <td>${x.id}</td>
      <td class="${x.ok ? "ok" : "bad"}">${x.outcome}</td>
      <td>${x.served_by || "—"}</td>
      <td>${fmt(x.duration_ms, 0)}</td>
      <td>$${fmt(x.cost_usd, 6)}</td>
      <td>${x.retries}</td>
      <td>${(x.guardrails || []).join("; ") || (x.skipped || []).join("; ") || ""}</td>
    </tr>`).join("")
  );
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""
