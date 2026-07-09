"""Shared helpers for master-frame unit tests (test_master_base, bias, dark).

Not a test module (no ``test_`` prefix), so pytest does not collect it.
"""

import numpy as np

CHIPS = ("GREEN", "RED")
NROW, NCOL = 10, 10  # small arrays for unit tests


def make_l1_arrays(rng=None, chips=CHIPS, shape=(NROW, NCOL)):
    """Return a synthetic ``stack_frames`` output dict.

    Seeded for deterministic shape/dtype/sign fixtures; the values themselves
    are not asserted numerically (real numeric behavior is pinned by the
    regression and stacking unit tests in each module).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    nrow, ncol = shape
    arrays = {}
    for chip in chips:
        arrays[f"{chip}_IMG"] = rng.normal(0.0, 5.0, (nrow, ncol)).astype(np.float32)
        arrays[f"{chip}_SNR"] = np.abs(rng.normal(10.0, 1.0, (nrow, ncol))).astype(
            np.float32
        )
        arrays[f"{chip}_MASK"] = np.ones((nrow, ncol), dtype=bool)
    return arrays
