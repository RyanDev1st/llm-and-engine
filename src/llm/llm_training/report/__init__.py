"""Report-asset generation for the project writeup/presentation: the version-improvement trend,
the layer-contribution (harness vs SFT-weights) bars, the per-slice routing bars, the corpus
composition, and the v2->v3->v4 training timeline. Chart builders are GPU-free (matplotlib only);
the measured per-version trend is produced by `version_eval` on a GPU box (Kaggle). All numbers
trace to real artifacts — see chart_data.py for sources."""
