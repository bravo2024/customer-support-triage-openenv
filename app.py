"""OpenEnv HTTP app — customer-support-triage. Implements the full OpenEnv server spec:
POST /reset        reset environment
POST /step         execute action
GET /state         current observation
GET /health        {"status": "healthy"}
GET /metadata      env name + description
GET /schema        action / observation / state schemas
POST /mcp          JSON-RPC 2.0 MCP endpoint
"""
from __futureing import annotations
from typing import Any, Dict, Literal, Optional
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from env import Action, CustomerSupportEnv, Observation, StepRequest
import gradio as gr

# FastAPI subapp strictly for OpenEnv spec compliance
openenv_subapp = FastAPI(title="customer-support-triage", version="0.1.0")
_env = None

class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"

@openenv_subapp.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}

@openenv_subapp.get("/metadata")
def metadata() -> Dict[str, str]:
    return {
        "name": "customer-support-triage",
        "description": (
            "RL environment for triaging customer support tickets "
            "across easy / medium / hard difficulty tiers."
        ),
    }

@openenv_subapp.get("/schema")
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

@openenv_subapp.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> dict:
    global _env
    task = req.task if req else "easy"
    _env = CustomerSupportEnv(task=task)
    obs = _env.reset()
    return {"observation": obs.model_dump(), "info": {"task": task}}

@openenv_subapp.post("/step")
def step(req: StepRequest) -> dict:
    if _env is None:
        raise ValueError("Environment not initialized. Call /reset first.")
    action = Action(**req.model_dump())
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }

@openenv_subapp.get("/state")
def state() -> dict:
    if _env is None:
        raise ValueError("Environment not initialized. Call /reset first.")
    obs = _env.state()
    return {"observation": obs.model_dump()}

# Gradio mount at /ui
ui_app = FastAPI()
ui_app.mount("/openenv", openenv_subapp)

@ui_app.post("/reset")
def reset_forward(req: Optional[ResetRequest] = None) -> dict:
    return openenv_subapp.reset(req)

@ui_app.post("/step")
def step_forward(req: StepRequest) -> dict:
    return openenv_subapp.step(req)

@ui_app.get("/state")
def state_forward() -> dict:
    return openenv_subapp.state()

@ui_app.post("/reset/")
def reset_slash_forward(req: Optional[ResetRequest] = None) -> dict:
    return reset_forward(req)

@ui_app.post("/step/")
def step_slash_forward(req: StepRequest) -> dict:
    return step_forward(req)

@ui_app.get("/state/")
def state_slash_forward() -> dict:
    return state_forward()

@ui_app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

# Mount Gradio explicitly at /ui
gradio_ui = gr.Interface(
    fn=lambda task, verbose: (_env.reset() if _env else {}),
    inputs=[
        gr.Dropdown(label="Task", choices=["easy", "medium", "hard"]),
        gr.Checkbox(label="Verbose", value=False),
    ],
    outputs=gr.JSON(label="Observation"),
    title="Customer Support Triage",
    allow_flagging="never",
)

gradio_app = gr.mount_gradio_app(ui_app, gradio_ui, path="/ui")

app = gradio_app