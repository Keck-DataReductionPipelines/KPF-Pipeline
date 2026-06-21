"""Profile ``Bias.make_master_l1`` (master bias from a stack of raw L0 biases).

Bias applies no calibrations, so the time is the raw stacking work in
``base.stack_frames``; the companion ``profile_master_base.py`` focuses the
drill-down on the engine itself.

Run with ``make profile-master_bias`` or
``python tests/profile_master_bias.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.masters.base as m_base
import kpfpipe.modules.masters.bias as m_bias


def run():
    def setup():
        return m_bias.Bias(P.masters_l0_files("bias"))

    P.run_profile(
        title="Master bias (Bias.make_master_l1)",
        report_name="master_bias",
        setup=setup,
        call=lambda mod: mod.make_master_l1(),
        candidate_modules=[m_bias, m_base],
    )


if __name__ == "__main__":
    run()
