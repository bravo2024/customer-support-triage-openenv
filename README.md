---
title: Customer Support Triage
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support Ticket Triage (OpenEnv)

Train AI agents to triage customer support tickets with **3 difficulty tiers** in a **real-world enterprise workflow**. This environment is fully compliant with the [OpenEnv](https://github.com/openenv/openenv) spec.

## Tasks

| Task   | Description                                      | Baseline Score (Rule-Based) |
|--------|--------------------------------------------------|-----------------------------|
| Easy   | Label tickets by sentiment.                      | 0.90                        |
| Medium | Draft responses to simple queries.                | 0.75                        |
| Hard   | Full triage (assign/prioritize/respond) under SLAs. | 0.60                      |

## Action Space

| Action      | Example                          |
|-------------|----------------------------------|
| `assign`    | `assign('T1', 'agent_1')`        |
| `prioritize`| `prioritize('T1', 'high')`       |
| `respond`   | `respond('T1', 'Your refund is processing.')` |
| `close`     | `close('T1')`                    |

## Observation Space

- `tickets`: List of active tickets (id, subject, sentiment, priority, SLA, resolved).
- `team_status`: Workload and specialties of support agents.
- `elapsed_time`: Minutes since the episode started.
- `sla_breaches`: Count of breached SLAs.

## Reward Function

- **+0.2** per ticket closed.
- **-0.1** per SLA breach.
- **+0.5** for resolving a positive-sentiment ticket (e.g., "Love the product!").

## Setup

### Required environment variables (for `inference.py`)
- `API_BASE_URL` (or `KILO_BASE_URL` for Kilo gateway)
- `MODEL_NAME` (or `KILO_MODEL_NAME` for Kilo default)
- `HF_TOKEN` (or `OPENAI_API_KEY`)

Free submission-ready defaults:
- `MODEL_NAME=kilo/z-ai/glm-5:free`
- `API_BASE_URL=https://integrate.api.nvidia.com/v1` (or Kilo's URL)

Example:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

1. **Deploy to Hugging Face Spaces**:
   - This Space runs natively on **Gradio SDK** (no Docker).
   - Builds automatically on push.

2. **Run Locally**:
   ```bash
   git clone https://huggingface.co/spaces/vivekkopthsd/customer-support-triage
   cd customer-support-triage
   pip install -r requirements.txt
   python app.py
   ```
   The Gradio UI will be available at `http://localhost:7860`.

## Example Baseline Scores

<!-- rebuild trigger: 2026-03-29T19:15+05:30 -->

| Task   | Random Agent | Rule-Based Agent |
|--------|--------------|-------------------|
| Easy   | 0.45         | 0.90              |
| Medium | 0.30         | 0.75              |
| Hard   | 0.20         | 0.60              |