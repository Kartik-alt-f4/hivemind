# ⬡ HiveMind

A recursive multi-agent task solver that runs in your terminal. Give it a hard problem — it splits it into independent workstreams, solves them in parallel across a tree of agents, and merges everything into one coherent answer. Works with any OpenAI-compatible API.

---

## Setup

```bash
cd hivemind
cp .env.example .env
nano .env              # add at least one provider key
./hivemind --test      # verify your providers work
./hivemind             # start solving
```

---

## Usage

```bash
# Single task
./hivemind "Write a competitive analysis of three note-taking apps"

# Interactive mode — keeps asking for tasks
./hivemind -i

# Tune the tree depth and agent count
./hivemind "Design a distributed caching system" --max-depth 5 --max-agents 12

# Test all providers before a session
./hivemind --test

# Run a specific provider test
python tests/test_groq.py
```

---

## Provider Guide

| Provider | Free Tier | Best Model | Get Key |
|---|---|---|---|
| **Gemini** | 1,500 req/day · 15/min | `gemini-2.0-flash-lite` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Groq** | 14,400 req/day | `llama-3.1-8b-instant` | [console.groq.com](https://console.groq.com/keys) |
| **OpenRouter** | Varies (`:free` models) | `meta-llama/llama-3.1-8b-instruct:free` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **OpenAI** | No free tier | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Ollama** | Fully local, no limits | `llama3`, `mistral` | [ollama.com](https://ollama.com/download) |

Add multiple keys for any provider (`key1,key2,key3`) — the cluster rotates between them automatically to multiply your effective rate limits.

---

## How It Works

Each agent outputs a plain-text marker, not JSON — small models work reliably:

```
##SPLIT##          → spawn child agents for each bullet point
##SOLVE##          → answer directly, return result up the tree
##CLARIFY##        → (root only) ask the user for missing context
```

Tree structure for a complex task:

```
Task
 └─ Root (depth 0) ──── SPLIT
     ├─ Agent A (depth 1) ── SOLVE ──────────────────────── result A
     ├─ Agent B (depth 1) ── SPLIT
     │   ├─ Agent B1 (depth 2) ── SOLVE ────────────────── result B1
     │   └─ Agent B2 (depth 2) ── SOLVE ────────────────── result B2
     │        └─ [B merges B1 + B2] ──────────────────────── result B
     └─ Agent C (depth 1) ── SOLVE ──────────────────────── result C
          └─ [Root merges A + B + C] ───────────────── final answer
```

Agents with 4+ independent subtasks are dispatched as Ray remote processes (true multi-process parallelism). Smaller fans use `asyncio.gather`.

---

## Tuning

| Setting | Default | Where | Effect |
|---|---|---|---|
| `MAX_TOTAL_AGENTS` | 30 | `agents/node.py` | Hard budget — prevents runaway trees |
| `MAX_DEPTH` | 6 | `.env` / `--max-depth` | Deeper = more splits, more agents |
| `MAX_PARALLEL_AGENTS` | 8 | `.env` / `--max-agents` | Concurrent LLM calls |
| `MIN_COMPLEXITY_TO_SPLIT` | 3 | `.env` / `--min-complexity` | Lower = splits more aggressively |
| `RAY_THRESHOLD` | 4 | `agents/node.py` | Min subtasks to justify Ray processes |

**Recommended starting points:**
- Fast/cheap: `--max-depth 4 --max-agents 6`
- Deep research: `--max-depth 6 --max-agents 12`
- Single provider with rate limits: `--max-agents 3`

---

## Troubleshooting

**Provider not working:**
```bash
./hivemind --test              # run all configured providers
python tests/test_groq.py      # debug a specific one
```

**Agents getting stuck or timing out:**
```bash
tail -f hivemind_debug.log     # live view of agent decisions
```

**Tree exploding (too many agents):**
Lower `MAX_TOTAL_AGENTS` in `agents/node.py` or pass `--max-agents 4`.

**Ray startup is slow:**
Expected — Ray has ~10-15s process startup overhead. It only activates when there are 4+ parallel subtasks (`RAY_THRESHOLD` in `agents/node.py`). Smaller trees fall back to asyncio automatically.

**Adding a new provider:**
No code needed — just add a block to `.env`:
```env
PROVIDER_MYAPI_BASE_URL=https://api.example.com/v1
PROVIDER_MYAPI_MODEL=my-model-name
PROVIDER_MYAPI_KEYS=key1,key2
```
