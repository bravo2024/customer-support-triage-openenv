---
title: Customer Support Triage
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support Ticket Triage

An RL environment where an AI agent manages a queue of customer support tickets. The agent reads incoming tickets and decides how to handle each one — assign it to a team member, change its priority, draft a response, or close it out. The goal is to resolve tickets quickly without breaching SLAs.

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).

## How it works

The environment exposes a simple HTTP API. You reset it to get a fresh set of tickets, then step through it by sending actions. Each action affects the queue and returns an updated observation plus a reward signal.

There are three difficulty levels:
- **Easy** — small queue, straightforward tickets, plenty of SLA time
- **Medium** — more tickets, mixed urgency, some require a proper response
- **Hard** — tight SLAs, negative-sentiment tickets, agent workload matters

## Actions

Each step you pick one ticket and one of these actions:

| Action      | When to use                          |
|-------------|--------------------------------------|
| `assign`    | Route ticket to the right agent      |
| `prioritize`| Bump or lower urgency                |
| `respond`   | Send a reply to the customer         |
| `close`     | Mark resolved                        |

```json
{
  "ticket_id": "T2",
  "action_type": "respond",
  "response_draft": "Sorry to hear that — we're looking into this now."
}
```

## Rewards
- `+0.2` for closing a ticket
- `+0.5` bonus for resolving a negative-sentiment ticket with a real response
- `-0.1` penalty per SLA breach

## Running the agent

Set your API credentials and run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

The script runs all three task difficulties and prints a score summary.

## Running locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

API will be at `http://localhost:7860`. You can hit `/reset`, `/step`, `/state` directly or check `/docs` for the full OpenAPI spec.