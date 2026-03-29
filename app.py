"""OpenEnv HTTP app — customer-support-triage.

Implements the full OpenEnv server spec:
  POST /reset        reset environment
  POST /step         execute action
  GET  /state        current observation
  GET  /health       {"status": "healthy"}
  GET  /metadata     env name + description
  GET  /schema       action / observation / state schemas
  POST /mcp          JSON-RPC 2.0 MCP endpoint
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

from env import Action, CustomerSupportEnv, Observation

app = FastAPI(title="customer-support-triage", version="0.1.0")

_env: CustomerSupportEnv = CustomerSupportEnv(task="easy")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"


class StepRequest(BaseModel):
    ticket_id: str
    action_type: Literal["assign", "prioritize", "respond", "close"]
    target_agent: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    response_draft: Optional[str] = None


# ---------------------------------------------------------------------------
# Standard OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> Dict[str, str]:
    return {
        "name": "customer-support-triage",
        "description": (
            "RL environment for triaging customer support tickets "
            "across easy / medium / hard difficulty tiers."
        ),
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "elapsed_time": {"type": "integer"},
                "sla_breaches": {"type": "integer"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request) -> Dict[str, Any]:
    """Minimal JSON-RPC 2.0 MCP endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    req_id = body.get("id")
    method = body.get("method", "")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment to initial state.",
                        "inputSchema": ResetRequest.model_json_schema(),
                    },
                    {
                        "name": "step",
                        "description": "Execute a triage action on a ticket.",
                        "inputSchema": StepRequest.model_json_schema(),
                    },
                ]
            },
        }

    if method == "tools/call":
        params = body.get("params", {})
        tool = params.get("name", "")
        args = params.get("arguments", {})
        if tool == "reset":
            result = reset(ResetRequest(**args) if args else None)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if tool == "step":
            result = step(StepRequest(**args))
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


# ---------------------------------------------------------------------------
# Environment endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root_get() -> Dict[str, str]:
    return {"name": "customer-support-triage", "status": "ok"}


@app.post("/")
def root_post(req: Optional[ResetRequest] = None) -> dict:
    return reset(req)


@app.post("/reset")
@app.post("/reset/")
@app.post("/openenv/reset")
def reset(req: Optional[ResetRequest] = None) -> dict:
    global _env
    task = req.task if req else "easy"
    _env = CustomerSupportEnv(task=task)
    obs = _env.reset()
    return {"observation": obs.model_dump(), "info": {"task": task}}


@app.get("/state")
@app.get("/state/")
@app.get("/openenv/state")
def state() -> dict:
    obs = _env.state()
    return {"observation": obs.model_dump()}


@app.post("/step")
@app.post("/step/")
@app.post("/openenv/step")
def step(req: StepRequest) -> dict:
    action = Action(**req.model_dump())
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import os
    import uvicorn
    uvicorn.run(app, host=host, port=int(os.getenv("PORT", str(port))))


if __name__ == "__main__":
    main()
