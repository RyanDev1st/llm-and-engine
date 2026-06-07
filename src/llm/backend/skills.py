from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
# Extra skill roots (os.pathsep-separated) layered ON TOP of SKILLS_DIR. Lets the
# demo/test catalog (src/llm/skills_demo) be served without bloating the default:
#   CHESS_SKILLS_DIRS=src/llm/skills_demo python -m backend.server
# The live skills/ root is read first, so its names win on collision.
EXTRA_DIRS_ENV = "CHESS_SKILLS_DIRS"


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


def _roots(root: Path) -> list[Path]:
    roots = [root]
    extra = os.environ.get(EXTRA_DIRS_ENV, "")
    roots += [Path(p) for p in extra.split(os.pathsep) if p.strip()]
    return roots


def load_skills(root: Path = SKILLS_DIR) -> list[Skill]:
    skills: list[Skill] = []
    seen: set[str] = set()
    for r in _roots(root):
        if not r.exists():
            continue
        for path in sorted(r.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            meta = _frontmatter(text)
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "")
            if description and name not in seen:
                seen.add(name)
                skills.append(Skill(name=name, description=description, content=text))
    return skills
