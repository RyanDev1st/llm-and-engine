"""LLM SFT dataset package.

Active generator lives in ``llm_dataset.v1`` (see ``v1/contracts.py`` for the
spec and ``v1/profiles.py`` for the v1.2 corpus). The pre-v1 pipeline
(contracts/, pipeline/, reports/, runtime/, validation/{admission,redteam,
replay,routing_sanity}, build/) was archived to "legacy [ignore]/"; only
``validation.hygiene`` remains, used by ``v1.dedup``.

This package intentionally exports nothing at the top level — import the
specific submodule you need (e.g. ``from llm_dataset.v1 import generate``).
"""
