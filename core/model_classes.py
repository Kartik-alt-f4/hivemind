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


# ── Capability tags ────────────────────────────────────────────────────────────
# A set of string tags describing what a model class is good at.
# The orchestrator uses these when allocating subtasks.

CAPABILITIES: dict[ModelClass, set[str]] = {
    ModelClass.ORCHESTRATOR: {
        "planning", "decomposition", "reasoning", "ambiguity",
        "architecture", "synthesis", "complex_coding", "research",
        "multi_step", "long_context",
    },
    ModelClass.ANALYST: {
        "reasoning", "architecture", "analysis", "medium_coding",
        "summarization", "structured_output", "planning",
    },
    ModelClass.WORKER: {
        "coding", "writing", "extraction", "formatting",
        "file_generation", "simple_reasoning",
    },
    ModelClass.FAST: {
        "formatting", "merging", "simple_writing", "echo",
        "short_answer",
    },
}


# ── Task-type → recommended class ─────────────────────────────────────────────
# When the orchestrator decomposes a task it tags each subtask with one of these
# task types. This maps them to the model class that should handle them.

TASK_TYPE_TO_CLASS: dict[str, ModelClass] = {
    # Orchestration-level
    "plan":             ModelClass.ORCHESTRATOR,
    "decompose":        ModelClass.ORCHESTRATOR,
    "synthesize":       ModelClass.ORCHESTRATOR,
    "final_merge":      ModelClass.ORCHESTRATOR,

    # Analysis / architecture
    "analyze":          ModelClass.ANALYST,
    "architect":        ModelClass.ANALYST,
    "research":         ModelClass.ANALYST,
    "review":           ModelClass.ANALYST,
    "debug":            ModelClass.ANALYST,

    # Execution
    "code":             ModelClass.WORKER,
    "write":            ModelClass.WORKER,
    "implement":        ModelClass.WORKER,
    "extract":          ModelClass.WORKER,
    "generate":         ModelClass.WORKER,
    "test":             ModelClass.WORKER,

    # Trivial / formatting
    "format":           ModelClass.FAST,
    "merge_simple":     ModelClass.FAST,
    "summarize_short":  ModelClass.FAST,
}


def class_for_task_type(task_type: str) -> ModelClass:
    """Return the recommended ModelClass for a given task_type string."""
    return TASK_TYPE_TO_CLASS.get(task_type.lower(), ModelClass.WORKER)


def class_label(mc: ModelClass) -> str:
    return mc.value.upper()
