"""Profile ``WLS.make_master_l2`` (master wavelength solution from ThAr stack).

Processes a ThAr stack through the L0->L2 pipeline (associating the bundled
bias/dark/flat masters from ``tests/testdata/masters``), fits the line list per
orderlet, and solves the per-fiber Legendre WLS — the heaviest master product.
The line-level pass re-runs the whole stack, so this harness is slow.

Run with ``make profile-master_wls`` or
``python tests/profiling/profile_master_wls.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.masters.base as m_base
import kpfpipe.modules.masters.wls as m_wls


def run():
    def setup():
        return m_wls.WLS(P.masters_l0_files("thar"), config=P.MASTERS_CONFIG)

    P.run_profile(
        title="Master WLS (WLS.make_master_l2)",
        report_name="master_wls",
        setup=setup,
        call=lambda mod: mod.make_master_l2(verbose=False),
        candidate_modules=[m_wls, m_base],
    )


if __name__ == "__main__":
    run()
