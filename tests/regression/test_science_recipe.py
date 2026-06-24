"""
Tests for the kpf_drp_science recipe.

Integration tests run the full recipe (L0 → L1 → L2) against a real star
observation from tests/testdata/L0/20240405/.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import build_filepath

# ---------------------------------------------------------------------------
# Test data paths and constants
# ---------------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTDATA_L0_DIR = TESTDATA_DIR / "L0" / "20240405"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "kpf_drp_science.toml"

OBS_ID = "KP.20240405.40113.57"

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL = DETECTOR["ccd"]["ncol"]


def _load_recipe():
    spec = importlib.util.spec_from_file_location(
        "kpf_drp_science",
        Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_science.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Science recipe integration (real L0 star data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestScienceRecipe:
    """End-to-end recipe test: KPF0 → ImageAssembly → SpectralExtraction → KPF2."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("science_out")

        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(TESTDATA_DIR),
                    "KPF_MASTERS_OUTPUT": str(TESTDATA_DIR),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=OBS_ID)

        recipe = _load_recipe()
        recipe.main(config, args)

        out_path = build_filepath(OBS_ID, "L2", data_root=str(tmp_path))
        return out_path

    def test_output_file_exists(self, recipe_output):
        assert os.path.isfile(recipe_output), (
            f"Expected output not found: {recipe_output}"
        )

    def test_output_filename_format(self, recipe_output):
        assert os.path.basename(recipe_output) == "kpf_SL2_20240405T110833.fits"

    def test_output_is_valid_kpf2(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert isinstance(l2, KPF2)

    @pytest.mark.parametrize(
        "key, expected_rows",
        [
            ("GREEN_SCI2_FLUX", NORDER_GREEN),
            ("RED_SCI2_FLUX", NORDER_RED),
            ("SCI2_FLUX", NORDER_GREEN + NORDER_RED),
        ],
    )
    def test_sci2_flux_shape(self, recipe_output, key, expected_rows):
        l2 = KPF2.from_fits(recipe_output)
        assert l2.data[key].shape == (expected_rows, NCOL)

    def test_flux_positive(self, recipe_output):
        """Star flux should be positive after extraction."""
        l2 = KPF2.from_fits(recipe_output)
        assert np.nanmedian(l2.data["GREEN_SCI2_FLUX"]) > 0
        assert np.nanmedian(l2.data["RED_SCI2_FLUX"]) > 0

    def test_variance_positive(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert np.nanmin(l2.data["GREEN_SCI2_VAR"]) >= 0
        assert np.nanmin(l2.data["RED_SCI2_VAR"]) >= 0

    def test_receipt_chain(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        modules = l2.receipt["Module_Name"].values
        assert "image_assembly" in modules
        assert "calibration_association" in modules
        assert "spectral_extraction" in modules
        assert "wavelength_calibration" in modules
        assert "barycentric_correction" in modules

    def test_barycorr_extensions_populated(self, recipe_output):
        """BarycentricCorrection should populate the rvdata-standard extensions
        per-order, with finite values."""
        l2 = KPF2.from_fits(recipe_output)
        norder = NORDER_GREEN + NORDER_RED
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            arr = np.asarray(l2.data[ext])
            assert arr.shape == (norder,), f"{ext} shape {arr.shape} != ({norder},)"
            assert np.all(np.isfinite(arr)), f"{ext} has non-finite values"
            assert np.issubdtype(arr.dtype, np.float64), (
                f"{ext} is {arr.dtype}, expected float64"
            )
        # Sanity: BARYCORR_Z is the redshift z = lambda_obs/lambda_rest - 1
        # (compute_redshift), so |z| = |v|/c << 1, not the 1+z factor near 1.
        z = np.asarray(l2.data["BARYCORR_Z"])
        assert np.all(np.abs(z) < 1e-3)

    @staticmethod
    def _hval(header, key):
        v = header[key]
        return v[0] if isinstance(v, tuple) else v

    def test_per_ccd_barycorr_keywords(self, recipe_output):
        """Per-CCD scalar summaries should land on PRIMARY."""
        l2 = KPF2.from_fits(recipe_output)
        prim = l2.headers["PRIMARY"]
        for key in ("CCD1BJD", "CCD1BKMS", "CCD1BZ", "CCD2BJD", "CCD2BKMS", "CCD2BZ"):
            assert key in prim, f"{key} missing from PRIMARY"
            assert np.isfinite(float(self._hval(prim, key))), f"{key} not finite"

    def test_calibration_headers_set(self, recipe_output):
        """CalibrationAssociation's PRIMARY writes (registered KPF-pipeline
        keywords) survive into the L2 PRIMARY."""
        l2 = KPF2.from_fits(recipe_output)
        prim = l2.headers["PRIMARY"]
        # bias/dark use full-path FILE + float AGE (no DIR). Flat association is
        # not part of the basic runnable path until flat processing is
        # implemented.
        for prefix in ("BIAS", "DARK"):
            assert f"{prefix}FILE" in prim
            assert f"{prefix}DIR" not in prim
            assert f"{prefix}AGE" in prim
        assert "FLATFILE" not in prim
        assert "FLATAGE" not in prim
        # thar uses the same convention: WLSFILE = full path (no WLSDIR),
        # WLSAGE = float days
        assert "WLSFILE" in prim
        assert "WLSDIR" not in prim
        assert self._hval(prim, "WLSFILE").endswith("_master_thar_L2.fits")
        assert isinstance(self._hval(prim, "WLSAGE"), float)

    def test_wave_arrays_populated(self, recipe_output):
        """WavelengthCalibration should fill the per-fiber WAVE extensions."""
        l2 = KPF2.from_fits(recipe_output)
        assert l2.data["GREEN_SCI2_WAVE"].shape == (NORDER_GREEN, NCOL)
        assert l2.data["RED_SCI2_WAVE"].shape == (NORDER_RED, NCOL)
        assert np.any(l2.data["GREEN_SCI2_WAVE"] != 0)
        assert np.any(l2.data["RED_SCI2_WAVE"] != 0)
        # Wavelength solutions are stored in float64.
        assert np.issubdtype(l2.data["GREEN_SCI2_WAVE"].dtype, np.float64)
        assert np.issubdtype(l2.data["RED_SCI2_WAVE"].dtype, np.float64)

    def test_qlp_l0_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L0"
        assert (qlp_dir / f"{OBS_ID}_L0_stitched_image_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L0_stitched_image_red_zoomable.png").is_file()

    def test_qlp_l1_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L1"
        assert (qlp_dir / f"{OBS_ID}_L1_image_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L1_image_red_zoomable.png").is_file()

    def test_qlp_l2_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L2"
        assert (qlp_dir / f"{OBS_ID}_L2_snr_per_order_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L2_snr_per_order_red_zoomable.png").is_file()


# ---------------------------------------------------------------------------
# Science recipe error paths
# ---------------------------------------------------------------------------


class TestScienceRecipeErrors:
    def test_missing_l0_file_raises(self, tmp_path):
        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(tmp_path),
                    "KPF_MASTERS_OUTPUT": str(tmp_path),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=OBS_ID)
        recipe = _load_recipe()
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            recipe.main(config, args)

    def test_missing_obs_id_raises(self, tmp_path):
        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(tmp_path),
                    "KPF_MASTERS_OUTPUT": str(tmp_path),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=None)
        recipe = _load_recipe()
        with pytest.raises(SystemExit, match="--obs_id is required"):
            recipe.main(config, args)
