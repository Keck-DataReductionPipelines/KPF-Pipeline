"""Profile ``WavelengthCalibration.perform`` (attach the WLS to L2 spectra).

Run with ``make profile-wavelength_calibration`` or
``python tests/profile_wavelength_calibration.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.wavelength_calibration as m_wls


def run():
    def setup():
        config = P.science_config()
        return m_wls.WavelengthCalibration(P.extract_l2(config), config)

    P.run_profile(
        title="WavelengthCalibration.perform",
        report_name="wavelength_calibration",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_wls],
    )


if __name__ == "__main__":
    run()
