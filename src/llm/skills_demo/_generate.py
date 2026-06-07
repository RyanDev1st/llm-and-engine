"""Render the 40 demo chess skills from _specs.SKILLS into <slug>/SKILL.md.

Each file gets frontmatter (name + description) that load_skills() needs, plus a
When-to-use / Steps / Constraint body matching the trained skill style.

Run from anywhere:
    python src/llm/skills_demo/_generate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _specs import SKILLS  # noqa: E402


def render(slug: str, desc: str, when: str, steps: tuple[str, str, str], constraint: str) -> str:
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return (
        f"---\nname: {slug}\ndescription: {desc}\n---\n\n"
        f"# {slug}\nWhen to use: {when}\nSteps:\n{body}\nConstraint: {constraint}\n"
    )


def main() -> None:
    slugs = [s[0] for s in SKILLS]
    if len(set(slugs)) != len(slugs):
        raise SystemExit("duplicate slug in _specs.SKILLS")
    for slug, desc, when, steps, constraint in SKILLS:
        out = ROOT / slug
        out.mkdir(parents=True, exist_ok=True)
        (out / "SKILL.md").write_text(render(slug, desc, when, steps, constraint), encoding="utf-8")
    print(f"wrote {len(SKILLS)} skills to {ROOT}")


if __name__ == "__main__":
    main()
