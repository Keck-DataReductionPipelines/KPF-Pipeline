"""Profile ``CrossCorrelation.perform`` (build the per-order CCFs, L2 -> L4).

Cross-correlates each illuminated orderlet against its mask to build the per-
order CCF cubes and seed the RV table; the fits themselves are done downstream by
``RadialVelocity``. The line-level drill-down pass is kept enabled.

Run with ``make profile-cross_correlation`` or
``python tests/profiling/profile_cross_correlation.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.cross_correlation as m_ccf


def run():
    def setup():
        config = P.science_config()
        return m_ccf.CrossCorrelation(P.bary_l2(config), config)

    P.run_profile(
        title="CrossCorrelation.perform (L2 -> L4)",
        report_name="cross_correlation",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_ccf],
    )


if __name__ == "__main__":
    run()
