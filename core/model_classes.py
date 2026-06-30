"""
Model Classes — capability-based model selection for HiveMind.

Four classes, ranked by capability/cost:

  ORCHESTRATOR  — strongest available model. Used for:
                  root planning, ambiguity detection, task decomposition,
                  final synthesis/merge when the task is complex.
                  Free-tier examples: gemini-2.5-pro, llama-3.3-70b (groq),
                  deepseek-r1 (openrouter free)

  ANALYST       — mid-weight reasoning model. Used for:
                  subtask planning at depth 1, code architecture decisions,
                  domain analysis that needs solid reasoning but not top-tier.
                  Free-tier examples: gemini-2.0-flash, llama-3.1-70b, qwen-72b

  WORKER        — capable but light. Used for:
                  leaf-level execution: writing code files, drafting prose,
                  running searches, extracting facts from context.
                  Free-tier examples: gemini-2.0-flash-lite, llama-3.1-8b,
                  mistral-7b

  FAST          — smallest/fastest. Used for:
                  merge of simple results, formatting, echo/pass-through tasks,
                  anything where latency matters more than quality.
                  Free-tier examples: gemini-flash-lite, llama-3.1-8b-instant

Each subtask in the decomposition carries a `model_class` tag so the
orchestrator allocates the right model before any agent runs.
"""
from enum import Enum


class ModelClass(str, Enum):
    ORCHESTRATOR = "orchestrator"   # planning, decomposition, final merge
    ANALYST      = "analyst"        # reasoning, architecture, mid-depth planning
    WORKER       = "worker"         # execution, writing, coding
    FAST         = "fast"           # formatting, simple merge, pass-through
