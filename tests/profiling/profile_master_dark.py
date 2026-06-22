"""Profile ``Dark.make_master_l1`` (master dark: bias-subtract, then stack).

Each raw dark is bias-subtracted (master bias associated from the bundled
``tests/testdata/masters``) before stacking, so this exercises ``_process_frame``
(CalibrationAssociation + ImageProcessing) on top of ``base.stack_frames``.

Run with ``make profile-master_dark`` or
``python tests/profiling/profile_master_dark.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.masters.base as m_base
import kpfpipe.modules.masters.dark as m_dark


def run():
    def setup():
        return m_dark.Dark(P.masters_l0_files("dark"), config=P.MASTERS_CONFIG)

    P.run_profile(
        title="Master dark (Dark.make_master_l1)",
        report_name="master_dark",
        setup=setup,
        call=lambda mod: mod.make_master_l1(),
        candidate_modules=[m_dark, m_base],
    )


if __name__ == "__main__":
    run()
