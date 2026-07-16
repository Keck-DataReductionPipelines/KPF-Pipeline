"""Profile ``RadialVelocity.perform`` (fit the per-order CCFs to RVs).

The per-order CCFs are built upstream by ``CrossCorrelation``; this module only
fits them, so its cost is the order-by-order two-pass Gaussian CCF-dip fits in
``_compute_rv_1d``. The line-level drill-down pass is kept enabled.

Run with ``make profile-radial_velocity`` or
``python tests/profiling/profile_radial_velocity.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.radial_velocity as m_rv


def run():
    def setup():
        config = P.science_config()
        return m_rv.RadialVelocity(P.ccf_l4(config), config)

    P.run_profile(
        title="RadialVelocity.perform (L2 -> L4)",
        report_name="radial_velocity",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_rv],
    )


if __name__ == "__main__":
    run()
