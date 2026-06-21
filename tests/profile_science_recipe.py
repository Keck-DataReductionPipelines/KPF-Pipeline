"""End-to-end profile of the science recipe (``recipes/kpf_drp_science``).

Runs the full L0 -> L4 reduction for one real obs_id under cProfile and ranks
functions **across all modules**, so the report shows which module dominates the
science run — the first thing to know before optimizing anything. Per-module
line-level detail lives in the individual ``profile_<module>.py`` harnesses, so
the (expensive) second line-profiler pass is disabled here.

Run with ``make profile-science`` or ``python tests/profile_science_recipe.py``.
Requires the real frames in ``tests/testdata`` (skips cleanly otherwise).
"""

import argparse
import importlib.util
from pathlib import Path

try:  # works both as `python -m tests.profile_*` and `python tests/profile_*.py`
    from tests import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

# Pipeline modules eligible for the (here disabled) line-level drill-down.
import kpfpipe.modules.barycentric_correction as m_bary
import kpfpipe.modules.calibration_association as m_calib
import kpfpipe.modules.image_assembly as m_assembly
import kpfpipe.modules.image_processing as m_proc
import kpfpipe.modules.radial_velocity as m_rv
import kpfpipe.modules.spectral_extraction as m_extract
import kpfpipe.modules.wavelength_calibration as m_wls

PIPELINE_MODULES = [
    m_assembly,
    m_calib,
    m_proc,
    m_extract,
    m_wls,
    m_bary,
    m_rv,
]


def _load_recipe():
    path = Path(__file__).parent.parent / "recipes" / "kpf_drp_science.py"
    spec = importlib.util.spec_from_file_location("kpf_drp_science", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    recipe = _load_recipe()

    def setup():
        config = P.science_config()
        args = argparse.Namespace(obs_id=P.SCIENCE_OBS_ID)
        return (config, args)

    P.run_profile(
        title="Science recipe (L0 -> L4)",
        report_name="science_recipe",
        setup=setup,
        call=recipe.main,
        candidate_modules=PIPELINE_MODULES,
        line_pass=False,
    )


if __name__ == "__main__":
    run()
