"""Baseline inference runner for Customer Support Triage.
MANDATORY env vars:
- API_BASE_URL (e.g. https://router.huggingface.co/v1)
- MODEL_NAME
- HF_TOKEN (or API_KEY)

Uses OpenAI client for all LLM calls.
"""
from __futureing import annotations
import json
import os
import re
from typing import List, Literal, Tuple
from openai import OpenAI
from env import Action, CustomerSupportEnv, Observation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 180
FALLBACK_ACTION = "close('AUTO_OLDEST')"

ACTION_PATTERN = re.compile(
    r"^(assign|prioritize|respond|close)\s*\((.*)\)\s*$",
    re.IGNORECASE)


SYSTEM_PROMPT = """
You are controlling a customer-support triage environment.
Reply with EXACTLY one action string in one of these forms:
- assign('TICKET_ID','agent_1')
- prioritize('TICKET_ID','high')
- respond('TICKET_ID','short response text')
- close('TICKET_ID')
Rules:
- Use single quotes around arguments.
- Pick ticket IDs from observation.
- If unsure, close the most urgent ticket (lowest SLA remaining).
- Return only the action string, no explanation.
""".strip()


def _obs_to_prompt(task: str, step: int, obs: Observation, history: List[str]) -> str:
    tickets = "\n".join(
        f"- {t.id} | priority={t.priority} | sla={t.sla_remaining} | sentiment={t.sentiment} | subject={t.subject} | resolved={t.resolved}"
        for t in obs.tickets)
    return (
        f"Task: {task}\n"
        f"Step: {step}\n"
        f"Elapsed: {obs.elapsed_time} min\n"
        f"SLA breaches: {obs.sla_breaches}\n"
        f"Recent actions: {history[-4:] if history else 'None'}\n"
        f"Tickets:\n{tickets}\n"
        "Return one valid action string.")


def _oldest_ticket_id(obs: Observation) -> str:
    unresolved = [t for t in obs.tickets if not t.resolved]
    if not unresolved:
        return obs.tickets[0].id
    return min(unresolved, key=lambda t: t.sla_remaining).id


def _parse_action(action_text: str, obs: Observation) -> Action:
    text = (action_text or "").strip()
    m = ACTION_PATTERN.match(text)
    if not m:
        return Action(ticket_id=_oldest_ticket_id(obs), action_type="close")

    kind = m.group(1).lower()
    raw_args = m.group(2)

    # crude single-quote arg parser
    args = re.findall(r"'([^']*)'", raw_args)
    if kind == "close" and len(args) >= 1:
        ticket_id = args[0]
        if args[0] != "AUTO_OLDEST":
            ticket_id = _oldest_ticket_id(obs)
        return Action(ticket_id=ticket_id, action_type="close")

    if kind == "assign" and len(args) >= 2:
        return Action(ticket_id=args[0], action_type="assign", target_agent=args[1])

    if kind == "prioritize" and len(args) >= 2:
        p = args[1].lower()
        if p not in {"low", "medium", "high"}:
            p = "high"
        return Action(ticket_id=args[0], action_type="prioritize", priority=p)  # type: ignore[arg-type]

    if kind == "respond" and len(args) >= 2:
        return Action(ticket_id=args[0], action_type="respond", response_draft=args[1])

    return Action(ticket_id=_oldest_ticket_id(obs), action_type="close")


def run_task(client: OpenAI, task: Literal["easy", "medium", "hard"]) -> Tuple[float, int]:
    env = CustomerSupportEnv(task=task)
    obs = env.reset()
    done = False
    total_reward = 0.0
    history: List[str] = []

    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        user_prompt = _obs_to_prompt(task, step, obs, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            reply = completion.choices[0].message.content or FALLBACK_ACTION
        except Exception:
            reply = FALLBACK_ACTION

        action = _parse_action(reply, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        history.append(f"{action.action_type}({action.ticket_id}) -> {reward:+.2f}")

    # Normalize to [0, 1] for leaderboard-style reporting
    normalized = max(0.0, min(1.0, (total_reward + 1.0) / 2.0))
    return normalized, len(history)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN (or API_KEY).")
    if not MODEL_NAME:
        raise RuntimeError("Missing MODEL_NAME.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores = {}
    for t in ("easy", "medium", "hard"):
        score, steps = run_task(client, t)  # type: ignore[arg-type]
        scores[t] = {"score": round(score, 4), "steps": steps}

    print(f"{t}: score={score:.4f}, steps={steps}")
    print("\nJSON summary:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()