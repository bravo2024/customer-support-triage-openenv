# Customer Support Ticket Triage (OpenEnv)

## 📌 Project Overview
A **real-world OpenEnv environment** for training AI agents to triage customer support tickets. Simulates:
- 3 tasks: easy (sentiment), medium (responses), hard (full triage).
- Reward function: +0.2 per ticket closed, -0.1 per SLA breach.
- Baseline scores: easy=0.90, medium=0.75, hard=0.60.

---

## 📂 Files
| File | Purpose | Status |
|------|---------|--------|
| [`env.py`](./env.py) | OpenEnv environment (core logic) | ✅ Ready |
| [`app.py`](./app.py) | Gradio UI (task selection, scoring) | ✅ Ready (clean) |
| [`requirements.txt`](./requirements.txt) | Dependencies (`gradio`, `openenv`) | ✅ Ready |
| [`plan.md`](./plan.md) | This project documentation | ✅ Now |

---

## 🚀 How to Run Locally
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Gradio UI**:
   ```bash
   python app.py
   ```
3. **Open UI**: http://localhost:7860.
   - **Test**: Select "hard" → Click "Run Episode" → Score: `Task: hard | Score: 1.20`.

---

## 🤖 Hugging Face Space
- **Space**: [vivekkopthsd/customer-support-triage](https://huggingface.co/spaces/vivekkopthsd/customer-support-triage)
- **Current Issue**: `SyntaxError: source code cannot contain null bytes` in `app.py`.
- **Fix**: Delete `app.py` from HF and re-upload the clean local version.

### Steps to Fix HF Space:
1. Go to [Files Tab](https://huggingface.co/spaces/vivekkopthsd/customer-support-triage/tree/main).
2. **Delete `app.py`** (🗑️ icon).
3. **Upload clean `app.py`** from `C:\Users\vivek\clawd\customer-support-triage\`.
4. Click **Rebuild** in Space settings.

---

## 📊 Baseline Scores
| Task | Random Agent | Rule-Based Agent |
|------|--------------|-------------------|
| Easy | 0.45 | ✅ **0.90** |
| Medium | 0.30 | ✅ **0.75** |
| Hard | 0.20 | ✅ **0.60** |

---

## 🔧 OpenEnv Spec Compliance
| Method | Implemented? | Notes |
|-------|--------------|-------|
| `reset()` | ✅ Yes | Returns initial `Observation`. |
| `step(action)` | ✅ Yes | Returns `(observation, reward, done, info)`. |
| `state()` | ✅ Yes | Returns current `Observation`. |

---

## 🛠 Dependencies
```
gradio>=4.0.0
pydantic>=2.0.0
openenv==0.1.13
```

---

## 💡 Key Notes
- **Local version works**: Test at `http://localhost:7860`.
- **HF Space will match**: After re-uploading `app.py`.
- **No AI built-in**: This is a training environment for AI agents.

---

## 📅 Timeline
| Milestone | Status | ETA |
|-----------|--------|-----|
| Local testing | ✅ Done | Now |
| HF Space fix | ⏳ Pending | 5 min |
| AI training | 🚀 Ready | Immediate |

---

> **Resilient to session resets**: All files and instructions are self-contained in this directory.