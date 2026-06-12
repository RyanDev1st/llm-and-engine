"""Runtime SKILL + plugin admin for the web app — paste a SKILL.md and tweak
plugin_context live, to test how the model follows injected skills.

Injected skills are written under a runtime dir added to CHESS_SKILLS_DIRS, so
both the catalog (serving_skills_index) and load_skill execution (ToolExecutor)
pick them up with no extra plumbing. Wiped on each server start.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[3] / "runs" / "runtime_skills"
ENV = "CHESS_SKILLS_DIRS"


def register() -> None:
    """Add RUNTIME_DIR to CHESS_SKILLS_DIRS and start from a clean slate."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for child in RUNTIME_DIR.glob("*"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
    parts = [p for p in os.environ.get(ENV, "").split(os.pathsep) if p.strip()]
    if str(RUNTIME_DIR) not in parts:
        parts.append(str(RUNTIME_DIR))
    os.environ[ENV] = os.pathsep.join(parts)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "skill"


def add_skill(name: str, description: str, body: str) -> str:
    """Write a runtime SKILL.md. name -> slug (load_skill arg can't contain
    spaces); description is forced single-line (frontmatter is line-based)."""
    if not name.strip() or not description.strip():
        raise ValueError("name and description are required")
    slug = _slug(name)
    desc = " ".join(description.split())
    out = RUNTIME_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    text = f"---\nname: {slug}\ndescription: {desc}\n---\n\n{body.strip()}\n"
    (out / "SKILL.md").write_text(text, encoding="utf-8")
    return slug


def delete_skill(name: str) -> bool:
    out = RUNTIME_DIR / _slug(name)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
        return True
    return False


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(",") if p.strip()]


def apply_plugin(plugin_context: dict, body: dict) -> dict:
    """Mutate plugin_context in place from {installed, enabled, marketplace}."""
    for key in ("installed", "enabled", "marketplace"):
        if key in body:
            plugin_context[key] = _as_list(body[key])
    return plugin_context


def catalog_payload(plugin_context: dict) -> dict:
    """Current served catalog (incl. runtime + plugin skills), the installed plugin
    bundles (name + the tools/skills each contributes), and the live plugin_context."""
    from .inference import serving_skills_index
    from . import plugins
    skills = [{"name": s["name"], "description": s["description"],
               "source": s.get("source", ""), "plugin": s.get("plugin", "")}
              for s in serving_skills_index(plugin_context)]
    runtime = {p.parent.name for p in RUNTIME_DIR.glob("*/SKILL.md")}
    for s in skills:
        s["runtime"] = s["name"] in runtime
    return {"skills": skills, "plugins": plugins.catalog(), "plugin_context": plugin_context}
