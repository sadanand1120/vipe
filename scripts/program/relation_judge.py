from __future__ import annotations

from pathlib import Path


PROGRAM_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PROGRAM_DIR / "prompts"

DEFAULT_POINT_SELECTION_DIST_M = 0.8
RELATION_LLM_CONCURRENCY = 16
RELATION_JUDGE_INSTRUCTIONS = (PROMPT_DIR / "task_relation_judge_instructions.txt").read_text().strip()
RELATION_JUDGE_PROMPT_TEMPLATE = (PROMPT_DIR / "task_relation_judge_prompt.txt").read_text()
RELATION_JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keep": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 180},
    },
    "required": ["keep", "reason"],
}


def make_relation_judge_prompt(
    *,
    task: str,
    relation: str,
    source_class: str,
    target_class: str,
    target_score: float,
    measurements: dict[str, float],
) -> str:
    values = {
        "task": task or "(task text not provided)",
        "relation": relation,
        "source_class": source_class,
        "source_instance_id": str(int(measurements["source_instance_id"])),
        "target_class": target_class,
        "target_instance_id": str(int(measurements["target_instance_id"])),
        "target_score": f"{target_score:.6f}",
    }
    values.update(
        {
            key: f"{float(value):.4f}"
            for key, value in measurements.items()
            if key not in {"source_instance_id", "target_instance_id"}
        }
    )
    prompt = RELATION_JUDGE_PROMPT_TEMPLATE
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt
