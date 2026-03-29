---
title: Customer Support Triage
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support Ticket Triage (OpenEnv)

An RL environment for training AI agents to triage customer support tickets across three difficulty tiers. Fully compliant with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec.

## Tasks

| Task   | Description                                         | Baseline Score |
|--------|-----------------------------------------------------|----------------|
| Easy   | Label tickets by sentiment                          | 0.90           |
| Medium | Draft responses to simple queries                   | 0.75           |
| Hard   | Full triage (assign/prioritize/respond) under SLAs  | 0.60           |

## API Endpoints

| Endpoint    | Method | Description                    | Response                                    |
|-------------|--------|--------------------------------|---------------------------------------------|
| `/health`   | GET    | Liveness check                 | `{"status": "healthy"}`                     |
| `/metadata` | GET    | Env name + description         | `{"name": ..., "description": ...}`         |
| `/schema`   | GET    | Action / observation schemas   | `{"action": ..., "observation": ..., "state": ...}` |
| `/reset`    | POST   | Reset env, returns initial obs | `{"observation": ..., "info": ...}`         |
| `/step`     | POST   | Execute action                 | `{"observation": ..., "reward": ..., "done": ..., "info": ...}` |
| `/state`    | GET    | Current observation            | `{"observation": ...}`                      |
| `/mcp`      | POST   | JSON-RPC 2.0 MCP endpoint      | JSON-RPC 2.0 response                       |

## Action Space

```json
{
  "ticket_id": "T1",
  "action_type": "assign | prioritize | respond | close",
  "target_agent": "agent_1",
  "priority": "low | medium | high",
  "response_draft": "We are looking into this."
}
```

## Observation Space

```json
{
  "tickets": [{"id": "T1", "subject": "...", "sentiment": "...", "priority": "...", "sla_remaining": 60, "resolved": false}],
  "team_status": {"agent_1": {"workload": 0, "specialties": ["billing"]}},
  "elapsed_time": 0,
  "sla_breaches": 0
}
```

## Reward Function

- `+0.2` per ticket closed
- `-0.1` per SLA breach
- `+0.5` for resolving a negative-sentiment ticket with a proper response

## Running inference.py

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

## Running Locally

```bash
pip install -r requirements.txt
python app.py
# API available at http://localhost:7860
```
