# ⬡ HiveMind

A recursive multi-agent task solver that lives in your terminal.
Spins up a dynamic tree of agents, splits hard problems into pieces,
solves them in parallel, and merges everything back together.

Works with **any OpenAI-compatible API** — Gemini, Groq, Together AI,
Ollama, OpenAI, and more. Rotates keys automatically across multiple accounts.

---

## Setup

```bash
cd hivemind
pip install -r requirements.txt

# Configure providers
cp .env.example .env
nano .env   # add your API keys
```

## Usage

```bash
# Single task
python main.py "Explain how transformers work in machine learning"

# Interactive mode (keeps asking for tasks)
python main.py -i

# Tune the tree
python main.py "Write a business plan for a coffee shop" \
  --max-depth 5 \
  --max-agents 12 \
  --min-complexity 4
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-depth` | 6 | Hard limit on recursion depth |
| `--max-agents` | 8 | Max agents running in parallel |
| `--min-complexity` | 3 | Score 1-10; below this, agent solves directly |
| `-i / --interactive` | off | Keep running after each task |

## Adding Providers

Edit `.env` — any block of three vars registers a new provider:

```env
PROVIDER_<NAME>_BASE_URL=https://...
PROVIDER_<NAME>_MODEL=model-name
PROVIDER_<NAME>_KEYS=key1,key2,key3
```

Built-in examples in `.env.example`: Gemini, Groq, Together AI, OpenAI, Ollama.

## How it works

```
Task
 └─ Orchestrator (depth 0): "split or solve?"
     ├─ Agent A (depth 1): subtask → solve directly
     ├─ Agent B (depth 1): subtask → split again
     │    ├─ Agent B1 (depth 2): leaf → solve
     │    └─ Agent B2 (depth 2): leaf → solve
     │         └─ [B merges B1+B2]
     └─ Agent C (depth 1): subtask → solve directly
          └─ [Root merges A+B+C → final answer]
```

Each agent decides its own depth. The `--min-complexity` threshold controls
how eagerly agents split vs. solve. Lower = more splits, more agents, deeper trees.
```
