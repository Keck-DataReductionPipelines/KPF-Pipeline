"""Shared helpers for the master-frame unit tests (not a test module)."""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

CHIPS = ("GREEN", "RED")
NROW, NCOL = 10, 10  # small arrays for unit tests


def make_l1_arrays(rng=None, chips=CHIPS, shape=(NROW, NCOL)):
    """Return a synthetic ``stack_frames`` output dict.

    Seeded for deterministic shape/dtype/sign fixtures; the values themselves are
    never asserted numerically, which the regression tests do instead.
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


# Eight synthetic source frames -- more than every ``min_stack_size`` gate in the
# masters modules (the largest is 5), so a stack succeeds unless a test makes it fail.
FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]

# Convention-conforming master name, for tests that need a write target on disk.
# Never asserted against production naming -- it is a destination, not an oracle.
MASTER_NAME = "KP.20240113.23249.10_master_bias_L1.fits"


@contextmanager
def mocked_stack(module, arrays=None):
    """Patch ``stack_frames`` on ``module`` to return synthetic arrays.

    Stacking real frames needs real data; every unit test instead feeds
    ``make_l1_arrays()`` straight into ``make_master_l1``. Use this directly when
    the test needs the module object afterwards (to call ``info`` or
    ``save_master`` on it); otherwise prefer ``make_mocked_master``.
    """
    value = make_l1_arrays() if arrays is None else arrays

    # Real stacking records its survivors in _stacked_files (the master's
    # INPUT_FILES); the mock stands in for that too.
    def _stack(l0_file_list=None, **kwargs):
        module._stacked_files = list(
            module.l0_file_list if l0_file_list is None else l0_file_list
        )
        return value

    with patch.object(module, "stack_frames", side_effect=_stack):
        yield module


def make_mocked_master(cls, *, files=FILE_LIST, arrays=None, config=None, **kwargs):
    """Build a master of type ``cls`` with ``stack_frames`` mocked out.

    Returns the master L1. ``kwargs`` pass through to ``make_master_l1``, so a
    caller can exercise ``master_path=`` or the per-subclass calibration flags.
    """
    module = cls(files) if config is None else cls(files, config=config)
    with mocked_stack(module, arrays):
        return module.make_master_l1(**kwargs)
