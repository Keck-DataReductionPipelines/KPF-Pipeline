"""Profile ``CalibrationAssociation.perform`` (master selection for an L1 frame).

Run with ``make profile-calibration_association`` or
``python tests/profile_calibration_association.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.calibration_association as m_calib


def run():
    def setup():
        config = P.science_config()
        l1 = P.assemble_l1(config)
        return m_calib.CalibrationAssociation(l1, config)

    P.run_profile(
        title="CalibrationAssociation.perform",
        report_name="calibration_association",
        setup=setup,
        call=lambda mod: mod.perform(["bias", "dark", "flat", "thar"]),
        candidate_modules=[m_calib],
    )


if __name__ == "__main__":
    run()
