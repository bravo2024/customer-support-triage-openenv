"""
FastAPI + Gradio UI for Customer Support Ticket Triage (OpenEnv).
Combines FastAPI for HF Spaces and Gradio for interactive task execution.
"""

from fastapi import FastAPI
import gradio as gr
from env import CustomerSupportEnv, Action, Observation
from typing import Literal


# FastAPI App (for HF Spaces)
app = FastAPI()


@app.get("/")
def greet():
    return {"status": "Customer Support Triage Environment"}


# Gradio UI (for interacting with the environment)
def run_episode(task: Literal["easy", "medium", "hard"], max_steps: int = 10) -> str:
    """Run a full episode and return the final score."""
    env = CustomerSupportEnv(task=task)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    
    while steps < max_steps:
        # Rule-based policy: close the oldest ticket first
        oldest_ticket = min(obs.tickets, key=lambda t: t.sla_remaining)
        action = Action(ticket_id=oldest_ticket.id, action_type="close")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    
    return f"Task: {task} | Steps: {steps} | Score: {total_reward:.2f}"


def show_observation(task: Literal["easy", "medium", "hard"]) -> str:
    """Return the initial observation for the task."""
    env = CustomerSupportEnv(task=task)
    obs = env.reset()
    
    # Format tickets
    tickets_str = "\n".join(
        f"- {t.id}: {t.subject} (Priority: {t.priority}, SLA: {t.sla_remaining}m)"
        for t in obs.tickets
    )
    
    # Format team status
    team_str = "\n".join(
        f"- {agent}: {status.workload} tickets assigned"
        for agent, status in obs.team_status.items()
    )
    
    return (
        f"**Tickets:**\n{tickets_str}\n\n"
        f"**Team Status:**\n{team_str}\n\n"
        f"**Elapsed Time:** {obs.elapsed_time}m | **SLA Breaches:** {obs.sla_breaches}"
    )


# Gradio UI
with gr.Blocks(title="Customer Support Triage") as demo:
    gr.Markdown("# Customer Support Ticket Triage (OpenEnv)")
    
    with gr.Tab("Run Episode"):
        task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"], 
            value="easy", 
            label="Select Task"
        )
        max_steps_slider = gr.Slider(
            minimum=1, maximum=20, value=10, label="Max Steps"
        )
        run_button = gr.Button("Run Episode")
        score_output = gr.Textbox(label="Score")
    
    with gr.Tab("Initial Observation"):
        obs_task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"], 
            value="easy", 
            label="Select Task"
        )
        obs_button = gr.Button("Show Observation")
        obs_output = gr.Textbox(label="Observation", lines=10)
    
    # Event handlers
    run_button.click(
        fn=run_episode,
        inputs=[task_dropdown, max_steps_slider],
        outputs=score_output
    )
    obs_button.click(
        fn=show_observation,
        inputs=obs_task_dropdown,
        outputs=obs_output
    )


# Mount Gradio UI at `/gradio`
app = gr.mount_gradio_app(app, demo, path="/gradio")