"""OpenEnv HTTP app + optional Gradio UI.

Implements POST reset/step and GET state for automated checks.
"""

from __future__ import annotations

from typing import Literal, Optional

import gradio as gr
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


# Optional Gradio UI for manual testing

def run_episode(task: Literal["easy", "medium", "hard"], max_steps: int = 10) -> str:
    env = CustomerSupportEnv(task=task)
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while steps < max_steps:
        oldest_ticket = min(obs.tickets, key=lambda t: t.sla_remaining)
        action = Action(ticket_id=oldest_ticket.id, action_type="close")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return f"Task: {task} | Steps: {steps} | Score: {total_reward:.2f}"


def show_observation(task: Literal["easy", "medium", "hard"]) -> str:
    env = CustomerSupportEnv(task=task)
    obs = env.reset()
    tickets_str = "\n".join(
        f"- {t.id}: {t.subject} (Priority: {t.priority}, SLA: {t.sla_remaining}m)"
        for t in obs.tickets
    )
    team_str = "\n".join(
        f"- {agent}: {status.workload} tickets assigned"
        for agent, status in obs.team_status.items()
    )
    return (
        f"Tickets:\n{tickets_str}\n\n"
        f"Team Status:\n{team_str}\n\n"
        f"Elapsed Time: {obs.elapsed_time}m | SLA Breaches: {obs.sla_breaches}"
    )


with gr.Blocks(title="Customer Support Triage") as demo:
    gr.Markdown("# Customer Support Ticket Triage (OpenEnv)")
    with gr.Tab("Run Episode"):
        task_dropdown = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task")
        max_steps = gr.Slider(1, 20, value=10, step=1, label="Max Steps")
        run_btn = gr.Button("Run")
        out_score = gr.Textbox(label="Score")
    with gr.Tab("Observation"):
        obs_task = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task")
        obs_btn = gr.Button("Show")
        out_obs = gr.Textbox(label="Observation", lines=10)

    run_btn.click(run_episode, inputs=[task_dropdown, max_steps], outputs=out_score)
    obs_btn.click(show_observation, inputs=obs_task, outputs=out_obs)


app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
