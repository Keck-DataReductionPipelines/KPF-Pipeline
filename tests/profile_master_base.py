"""Profile the master stacking engine (``masters/base.py``) via ``Bias``.

Master bias applies no calibrations, so ``Bias.make_master_l1`` is the cleanest
exercise of ``BaseMasterModule.stack_frames`` (sigma-clipped streaming
accumulation) — the code shared by every master type. The drill-down here is
focused on ``base.py`` (the engine), with the bias-specific profile in
``profile_master_bias.py``.

Run with ``make profile-master_base`` or
``python tests/profile_master_base.py``. Requires real frames in
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
        title="Master stacking engine (base.py stack_frames, via Bias)",
        report_name="master_base",
        setup=setup,
        call=lambda mod: mod.make_master_l1(),
        candidate_modules=[m_base],
    )


if __name__ == "__main__":
    run()
