"""
OpenEnv-compliant Customer Support Ticket Triage Environment.

Tasks:
- Easy: Label tickets by sentiment.
- Medium: Draft responses to simple queries.
- Hard: Full triage (assign/prioritize/respond) under SLAs.

Reward Signals:
- +0.2 per ticket closed.
- -0.1 per SLA breach.
- +0.5 per happy customer (resolved positive sentiment).
"""

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel


# --- Typed Models ---
class Ticket(BaseModel):
    id: str
    subject: str
    customer: str
    sentiment: Literal["positive", "neutral", "negative"]
    priority: Literal["low", "medium", "high"]
    content: str
    sla_remaining: int  # Minutes until SLA breach
    resolved: bool = False


class AgentStatus(BaseModel):
    agent_id: str
    workload: int  # Tickets assigned
    specialties: List[str]  # e.g., ["billing", "technical"]


class Observation(BaseModel):
    tickets: List[Ticket]
    team_status: Dict[str, AgentStatus]
    elapsed_time: int  # Minutes since reset
    sla_breaches: int
    last_action_error: bool = False


class Action(BaseModel):
    ticket_id: str
    action_type: Literal["assign", "prioritize", "respond", "close"]
    target_agent: Optional[str] = None  # For assign/escalate
    priority: Optional[Literal["low", "medium", "high"]] = None  # For prioritize
    response_draft: Optional[str] = None  # For respond


# --- Environment ---
class CustomerSupportEnv:
    def __init__(self, task: Literal["easy", "medium", "hard"] = "easy"):
        self.task = task
        self.elapsed_time = 0
        self.sla_breaches = 0
        self.history = []
        self._load_data()

    def _load_data(self):
        """Load task-specific datasets (mock for now)."""
        if self.task == "easy":
            self.tickets = [
                Ticket(id="T1", subject="Thanks!", customer="Alice", sentiment="positive", priority="low", content="Love the product!", sla_remaining=120),
                Ticket(id="T2", subject="Broken link", customer="Bob", sentiment="negative", priority="medium", content="Can't access my account.", sla_remaining=60),
            ]
        elif self.task == "medium":
            self.tickets = [
                Ticket(id="T3", subject="Where's my order?", customer="Charlie", sentiment="neutral", priority="medium", content="Order #12345 hasn't shipped.", sla_remaining=240),
            ]
        else:  # hard
            self.tickets = [
                Ticket(id="T4", subject="Refund request", customer="Dana", sentiment="negative", priority="high", content="Want refund for order #67890.", sla_remaining=30),
                Ticket(id="T5", subject="Feature suggestion", customer="Eve", sentiment="positive", priority="low", content="Add dark mode.", sla_remaining=480),
            ]
        self.team_status = {
            "agent_1": AgentStatus(agent_id="agent_1", workload=0, specialties=["billing"]),
            "agent_2": AgentStatus(agent_id="agent_2", workload=0, specialties=["technical"]),
        }

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self.elapsed_time = 0
        self.sla_breaches = 0
        self.history = []
        self._load_data()
        return self.state()

    def state(self) -> Observation:
        """Return current environment state."""
        return Observation(
            tickets=self.tickets,
            team_status=self.team_status,
            elapsed_time=self.elapsed_time,
            sla_breaches=self.sla_breaches,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Execute action and return (observation, reward, done, info)."""
        reward = 0.0
        done = False
        info = {}

        # Execute action
        ticket = next(t for t in self.tickets if t.id == action.ticket_id)
        if action.action_type == "assign":
            if action.target_agent:
                self.team_status[action.target_agent].workload += 1
        elif action.action_type == "prioritize":
            if action.priority:
                ticket.priority = action.priority
        elif action.action_type == "respond":
            if action.response_draft:
                ticket.resolved = "refund" in action.response_draft.lower() if "refund" in ticket.content.lower() else True
        elif action.action_type == "close":
            ticket.resolved = True
            reward += 0.2  # Partial reward for closing

        # Update SLAs
        self.elapsed_time += 10  # Simulate 10 minutes passing
        for t in self.tickets:
            if not t.resolved:
                t.sla_remaining -= 10
                if t.sla_remaining <= 0 and not t.resolved:
                    self.sla_breaches += 1
                    reward -= 0.1

        # Check for episode end
        if all(t.resolved for t in self.tickets) or self.elapsed_time >= 120:
            done = True

        return self.state(), reward, done, info


# --- Validation ---
if __name__ == "__main__":
    env = CustomerSupportEnv(task="hard")
    obs = env.reset()
    action = Action(ticket_id="T4", action_type="close")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")