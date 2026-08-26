"""Regression-suite fixtures.

Pins matplotlib to the headless Agg backend for every module in this directory
and closes figures after each test, so a plotting test cannot leak a window, a
renderer, or a "More than 20 figures" warning into an unrelated test.

The pin is an environment variable rather than ``matplotlib.use()`` so this file
imports nothing: matplotlib reads ``MPLBACKEND`` at its own import time, which
keeps the ~1 s matplotlib import off every ``make test-fast`` session that never
plots.
"""

import os
import sys

import pytest

# Must be set before the first `import matplotlib` anywhere in the session.
# setdefault, so an explicit MPLBACKEND from the environment still wins.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test opened, without importing pyplot for tests that
    never plot -- the teardown is a no-op until something else has imported it."""
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture
def mini_detector(monkeypatch):
    """Shrink ``DETECTOR['ccd']`` to a 20x20, overscan-free detector.

    The DATAPR* checks pin exact shapes read from ``DETECTOR`` at call time, so a
    conforming L1 would otherwise be 4 x (4080, 4080) float32 -- 266 MB per
    fixture, on disk as well as in memory. Patching the geometry instead of the
    fixtures keeps the synthetic frames small: ``write_amp_l0``'s default
    4 x (10, 10) amps tile to exactly 20x20, and ``_make_kpf1``'s default
    (20, 20) is the assembled frame. ``DETECTOR['norder']`` is deliberately left
    alone -- ``_KPF2DataDict`` bakes it in at import, and 35/32 rows are cheap.

    Tests that run on truth frames must not use this.
    """
    from kpfpipe import DETECTOR

    for key, value in (
        ("nrow", 20),
        ("ncol", 20),
        ("prescan", 0),
        ("oscan_srl", 0),
        ("oscan_prl", 0),
    ):
        monkeypatch.setitem(DETECTOR["ccd"], key, value)
