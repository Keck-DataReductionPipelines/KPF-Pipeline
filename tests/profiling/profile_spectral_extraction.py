"""Profile ``SpectralExtraction.perform`` (L1 2D FFI -> L2 1D spectra).

Run with ``make profile-spectral_extraction`` or
``python tests/profiling/profile_spectral_extraction.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.spectral_extraction as m_extract


def run():
    def setup():
        config = P.science_config()
        return m_extract.SpectralExtraction(P.process_l1(config), config)

    P.run_profile(
        title="SpectralExtraction.perform (L1 -> L2)",
        report_name="spectral_extraction",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_extract],
    )


if __name__ == "__main__":
    run()
