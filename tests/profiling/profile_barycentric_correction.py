"""Profile ``BarycentricCorrection.perform`` (per-order barycentric correction).

Note: the first run may include a one-time Gaia DR3 network lookup, which the
report will surface as a (non-compute) cumulative-time entry.

Run with ``make profile-barycentric_correction`` or
``python tests/profiling/profile_barycentric_correction.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.barycentric_correction as m_bary


def run():
    def setup():
        config = P.science_config()
        return m_bary.BarycentricCorrection(P.wls_l2(config), config)

    P.run_profile(
        title="BarycentricCorrection.perform",
        report_name="barycentric_correction",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_bary],
    )


if __name__ == "__main__":
    run()
