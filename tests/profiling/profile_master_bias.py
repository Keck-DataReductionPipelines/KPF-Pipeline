"""Profile ``Bias.make_master_l1`` (master bias from a stack of raw L0 biases).

Bias applies no calibrations, so the time is the raw stacking work in
``base.stack_frames``; with attribution charging that engine time to its
``base.py`` methods, this doubles as the profile of the shared stacking engine.

Run with ``make profile-master_bias`` or
``python tests/profiling/profile_master_bias.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
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
