from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta


def load_skills(root: Path = SKILLS_DIR) -> list[Skill]:
    if not root.exists():
        return []
    skills: list[Skill] = []
    for path in sorted(root.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        meta = _frontmatter(text)
        name = meta.get("name", path.parent.name)
        description = meta.get("description", "")
        if description:
            skills.append(Skill(name=name, description=description, content=text))
    return skills
