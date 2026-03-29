"""OpenEnv HTTP app.

Implements POST reset/step and GET state for automated checks.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import Action, CustomerSupportEnv

app = FastAPI(title="customer-support-triage")

# Runtime env instance
_env: CustomerSupportEnv = CustomerSupportEnv(task="easy")


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"


class StepRequest(BaseModel):
    ticket_id: str
    action_type: Literal["assign", "prioritize", "respond", "close"]
    target_agent: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    response_draft: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# Compatibility root for platforms that probe POST /
@app.post("/")
def root_post(req: Optional[ResetRequest] = None) -> dict:
    return reset(req)


@app.get("/")
def root_get() -> dict:
    return {"name": "customer-support-triage", "status": "ok"}


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



if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
