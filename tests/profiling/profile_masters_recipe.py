"""End-to-end profile of the masters recipe (``recipes/kpf_drp_masters``).

Builds the nightly calibration masters for one real datecode (master bias and
dark L1, then the master WLS L2 from ThAr) under cProfile and ranks functions
across all master modules, so the report shows which stage dominates the masters
run. Optimized independently of the science recipe. Per-module line detail is in
``profile_master_bias.py`` / ``profile_master_dark.py`` / ``profile_master_wls.py``,
so the second line-profiler pass is disabled here.

The bundled darks span two default-gap clusters, so the recipe's
``build_l0_file_lists`` is wrapped to widen ``cluster_gap_seconds`` (the same
accommodation the master-dark regression test makes) — otherwise the recipe
raises on a too-small dark cluster.

Run with ``make profile-masters`` or
``python tests/profiling/profile_masters_recipe.py``.
Requires the real frames in ``tests/testdata`` (skips cleanly otherwise).
"""

import argparse
import functools
import importlib.util
from pathlib import Path

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.masters.base as m_base
import kpfpipe.modules.masters.bias as m_bias
import kpfpipe.modules.masters.dark as m_dark
import kpfpipe.modules.masters.wls as m_wls

MASTERS_MODULES = [m_base, m_bias, m_dark, m_wls]


def _load_recipe():
    path = Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_masters.py"
    spec = importlib.util.spec_from_file_location("kpf_drp_masters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Widen the cluster gap so the bundled darks group into one stackable cluster.
    # build_l0_file_lists is a FileHandler staticmethod; wrap the class attribute
    # the recipe resolves through.
    original = module.FileHandler.build_l0_file_lists
    module.FileHandler.build_l0_file_lists = staticmethod(
        functools.partial(original, cluster_gap_seconds=P.MASTERS_CLUSTER_GAP_SECONDS)
    )
    return module


def run():
    recipe = _load_recipe()

    def setup():
        config = P.masters_config()
        args = argparse.Namespace(datecode=P.MASTERS_DATECODE, obs_id=None)
        return (config, args)

    P.run_profile(
        title="Masters recipe (bias, dark, WLS)",
        report_name="masters_recipe",
        setup=setup,
        call=recipe.main,
        candidate_modules=MASTERS_MODULES,
        line_pass=False,
        recipe=True,
        io_compute=True,
    )


if __name__ == "__main__":
    run()
