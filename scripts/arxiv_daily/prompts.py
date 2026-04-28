from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptBundle:
    version: str
    system: str
    user_template: str
    schema: dict


def load_prompt_bundle(prompt_dir: Path) -> PromptBundle:
    system = read_text(prompt_dir / "system.md")
    user_template = read_text(prompt_dir / "user.md")
    with (prompt_dir / "schema.json").open("r", encoding="utf-8") as file:
        schema = json.load(file)
    validate_template(user_template, {"paper_json", "output_schema"})
    return PromptBundle(
        version=f"{prompt_dir.name}/v1",
        system=system,
        user_template=user_template,
        schema=schema,
    )


def render_user_prompt(bundle: PromptBundle, paper: dict) -> str:
    paper_json = json.dumps(paper, ensure_ascii=False, sort_keys=True)
    output_schema = json.dumps(bundle.schema, ensure_ascii=False, indent=2)
    return bundle.user_template.format(
        paper_json=paper_json,
        output_schema=output_schema,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def validate_template(template: str, allowed_fields: set[str]) -> None:
    formatter = string.Formatter()
    fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name is not None
    }
    unknown = fields - allowed_fields
    missing = allowed_fields - fields
    if unknown:
        raise ValueError(f"Unknown prompt placeholders: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing prompt placeholders: {sorted(missing)}")
