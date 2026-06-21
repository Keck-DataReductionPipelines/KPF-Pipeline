"""Profile ``RadialVelocity.perform`` (CCF computation and RV fitting, L2 -> L4).

This is the expected tallest tentpole of the science pipeline (per-velocity,
per-line inner loops in ``_compute_ccf_1d``), so the line-level drill-down pass
is kept enabled and this harness is the slowest to run.

Run with ``make profile-radial_velocity`` or
``python tests/profile_radial_velocity.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.radial_velocity as m_rv


def run():
    def setup():
        config = P.science_config()
        return m_rv.RadialVelocity(P.bary_l2(config), config)

    P.run_profile(
        title="RadialVelocity.perform (L2 -> L4)",
        report_name="radial_velocity",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_rv],
    )


if __name__ == "__main__":
    run()
