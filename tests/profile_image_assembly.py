"""Profile ``ImageAssembly.perform`` (L0 -> L1 full-frame assembly).

Run with ``make profile-image_assembly`` or
``python tests/profile_image_assembly.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.image_assembly as m_assembly


def run():
    def setup():
        config = P.science_config()
        return m_assembly.ImageAssembly(P.load_l0(config), config)

    P.run_profile(
        title="ImageAssembly.perform (L0 -> L1)",
        report_name="image_assembly",
        setup=setup,
        call=lambda mod: mod.perform(),
        candidate_modules=[m_assembly],
    )


if __name__ == "__main__":
    run()
