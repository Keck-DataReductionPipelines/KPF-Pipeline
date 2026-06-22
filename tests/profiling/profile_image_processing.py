"""Profile ``ImageProcessing.perform`` (bias/dark subtraction on the L1 FFI).

Run with ``make profile-image_processing`` or
``python tests/profiling/profile_image_processing.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.image_processing as m_proc
from kpfpipe.modules.calibration_association import CalibrationAssociation


def run():
    def setup():
        config = P.science_config()
        l1 = P.assemble_l1(config)
        l1 = CalibrationAssociation(l1, config).perform(
            ["bias", "dark", "flat", "thar"]
        )
        return m_proc.ImageProcessing(l1, config)

    P.run_profile(
        title="ImageProcessing.perform (bias/dark)",
        report_name="image_processing",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_proc],
    )


if __name__ == "__main__":
    run()
