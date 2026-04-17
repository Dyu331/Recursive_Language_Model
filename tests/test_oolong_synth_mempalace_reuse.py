"""Palace reuse contract for bench_Oolong_synth/run_rlm_mempalace_benchmark.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BENCH = _REPO / "bench_Oolong_synth"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run_rlm_mempalace_benchmark as synth_mp


def test_ctx_hash_stable_for_identical_ingest_string() -> None:
    """Reuse key must be exactly the string passed to build_ephemeral_palace_tools (labeled)."""
    a = "Date: Jan 01, 2024 || User: 1 || Instance: x || Label: location\n"
    assert synth_mp._ctx_hash(a) == synth_mp._ctx_hash(a)
    b = a + "\nextra preamble line"
    assert synth_mp._ctx_hash(a) != synth_mp._ctx_hash(b)


def test_ctx_hash_differs_from_unlabeled_context() -> None:
    """Palace identity must follow labeled text, not context_window_text alone."""
    labeled = (
        "Date: Jan 01, 2024 || User: 1 || Instance: x || Label: location\n"
    )
    unlabeled = "Date: Jan 01, 2024 || User: 1 || Instance: x\n"
    assert synth_mp._ctx_hash(labeled) != synth_mp._ctx_hash(unlabeled)
