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
