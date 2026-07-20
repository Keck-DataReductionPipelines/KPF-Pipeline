"""Tests for the kpf_drp_science recipe.

Integration tests run the full recipe (L0 -> L1 -> L2) against a real star
observation from tests/testdata/L0/20240405/.
"""

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_filepath
from recipes._logging import science_run_summary

# ---------------------------------------------------------------------------
# Test data paths and constants
# ---------------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTDATA_L0_DIR = TESTDATA_DIR / "L0" / "20240405"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "kpf_drp_science.toml"

OBS_ID = "KP.20240405.40113.57"

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]


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

        out_path = kpf_filepath(OBS_ID, "L2", data_root=str(tmp_path))
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

    def test_flux_positive(self, recipe_output):
        """Star flux should be positive after extraction."""
        l2 = KPF2.from_fits(recipe_output)
        assert np.nanmedian(l2.data["GREEN_SCI2_FLUX"]) > 0
        assert np.nanmedian(l2.data["RED_SCI2_FLUX"]) > 0

    def test_receipt_chain(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        modules = l2.receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "calibration_association" in modules
        assert "spectral_extraction" in modules
        assert "wavelength_calibration" in modules
        assert "barycentric_correction" in modules

    def test_barycorr_extensions_populated(self, recipe_output):
        """BarycentricCorrection should populate the EPRV-standard extensions
        per-order, with finite values."""
        l2 = KPF2.from_fits(recipe_output)
        norder = NORDER_GREEN + NORDER_RED
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            arr = np.asarray(l2.data[ext])
            assert arr.shape == (norder,), f"{ext} shape {arr.shape} != ({norder},)"
            assert np.all(np.isfinite(arr)), f"{ext} has non-finite values"

    def test_per_ccd_barycorr_keywords(self, recipe_output):
        """Per-CCD scalar summaries land on their barycentric extension headers."""
        l2 = KPF2.from_fits(recipe_output)
        homes = {
            "CCD1BJD": "BJD_TDB",
            "CCD2BJD": "BJD_TDB",
            "CCD1BKMS": "BARYCORR_KMS",
            "CCD2BKMS": "BARYCORR_KMS",
            "CCD1BZ": "BARYCORR_Z",
            "CCD2BZ": "BARYCORR_Z",
        }
        for key, ext in homes.items():
            hdr = l2.headers[ext]
            assert key in hdr, f"{key} missing from {ext}"
            assert np.isfinite(float(hdr.get(key))), f"{key} not finite"

    def test_calibration_headers_set(self, recipe_output):
        """CalibrationAssociation's writes survive onto the L2 product: master
        paths on RECEIPT, ages on QUALITY_CONTROL (their registry homes)."""
        l2 = KPF2.from_fits(recipe_output)
        receipt = l2.headers["RECEIPT"]
        qc = l2.headers["QUALITY_CONTROL"]
        # bias/dark use full-path FILE + float AGE (no DIR). Flat association is
        # not part of the basic runnable path until flat processing is
        # implemented.
        for prefix in ("BIAS", "DARK"):
            assert f"{prefix}FILE" in receipt
            assert f"{prefix}DIR" not in receipt
            assert f"{prefix}AGE" in qc
        assert "FLATFILE" not in receipt
        assert "FLATAGE" not in qc
        # thar uses the same convention: WLSFILE = full path (no WLSDIR),
        # WLSAGE = float days
        assert "WLSFILE" in receipt
        assert "WLSDIR" not in receipt
        assert receipt.get("WLSFILE").endswith("_master_thar_L2.fits")
        assert isinstance(qc.get("WLSAGE"), float)

    def test_provenance_keywords_set(self, recipe_output):
        """DRPTAG (EPRV) stays on the L2 PRIMARY; the WMKO DRP-RUN provenance cards
        live on the L2 RECEIPT."""
        l2 = KPF2.from_fits(recipe_output)
        prim = l2.headers["PRIMARY"]
        receipt = l2.headers["RECEIPT"]
        version = importlib.metadata.version("kpfpipe")
        assert prim.get("DRPTAG") == version
        # The four provenance cards moved off PRIMARY onto RECEIPT.
        assert all(k not in prim for k in ("DRPVERNO", "DRPSTATU", "PROGID", "KOAID"))
        assert receipt.get("DRPVERNO") == version
        assert "PROGID" in receipt
        assert "KOAID" in receipt
        # BarycentricCorrection is the last module to run before the L2 write.
        assert receipt.get("DRPSTATU") == "Barycentric Correction module complete"

    def test_wave_arrays_populated(self, recipe_output):
        """WavelengthCalibration wiring: the per-fiber WAVE extensions are
        populated (nonzero) after the real run."""
        l2 = KPF2.from_fits(recipe_output)
        assert np.any(l2.data["GREEN_SCI2_WAVE"] != 0)
        assert np.any(l2.data["RED_SCI2_WAVE"] != 0)

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

    def test_qlp_l4_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L4"
        assert (qlp_dir / f"{OBS_ID}_L4_ccf_grid_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L4_ccf_grid_red_zoomable.png").is_file()


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


class _FakeL4:
    """Minimal stand-in for a finished KPF4: the attributes _summary reads."""

    def __init__(self, obs_id, headers, receipt):
        self.obs_id = obs_id
        self.headers = headers
        self.receipt = receipt


class TestScienceSummary:
    """Unit tests for the science_run_summary() run-verdict formatter."""

    def _l4(self):
        headers = {
            "RECEIPT": {
                "ORIGID": "KP.20240405.40113.57",
                "BIASFILE": "/m/20240405/KP.20240405.03637.74_master_bias_L1.fits",
                "DARKFILE": "/m/20240405/KP.20240405.03637.74_master_dark_L1.fits",
                "WLSFILE": "/m/20240405/KP.20240405.63499.95_master_thar_L2.fits",
            },
            "PRIMARY": {"RV": 11.290158, "RVERR": 0.000156, "BJDTDB": 2460405.968919},
            "QUALITY_CONTROL": {"ISGOOD": 0},
        }
        receipt = pd.DataFrame(
            [
                ("from_fits", "fn=/in/L0/20240405/KP.20240405.40113.57.fits, foo=None"),
                (
                    "to_fits",
                    "out_filepath=/out/L2/20240405/kpf_SL2_20240405T110833.fits",
                ),
                (
                    "to_fits",
                    "out_filepath=/out/L4/20240405/kpf_SL4_20240405T110833.fits",
                ),
            ],
            columns=["FUNCTION", "ARGS"],
        )
        return _FakeL4("KP.20240405.40113.57", headers, receipt)

    def test_all_fields_from_l4(self):
        text = science_run_summary(self._l4(), 92.4)
        assert "run summary: KP.20240405.40113.57" in text
        # Input/output paths come from the RECEIPT table, shown as basenames.
        assert "inputs:   KP.20240405.40113.57.fits" in text
        assert "/in/" not in text
        assert (
            "outputs:  kpf_SL2_20240405T110833.fits  kpf_SL4_20240405T110833.fits"
            in text
        )
        # Masters from the RECEIPT header cards.
        assert "bias=KP.20240405.03637.74_master_bias_L1.fits" in text
        assert "thar=KP.20240405.63499.95_master_thar_L2.fits" in text
        assert "ISGOOD:   0" in text
        # RV km/s, error m/s (0.000156 km/s -> 0.156 m/s).
        assert "+11.29016 km/s  err 0.156 m/s  @ BJD_TDB 2460405.968919" in text
        assert "elapsed:  92.4 s" in text
        # Internal blank-line padding so the block stands out in the log.
        assert text.startswith("\n\n") and text.endswith("\n\n")

    def test_no_science_combine_and_no_receipt(self):
        # RV absent (or FITS UNDEFINED) -> n/a; no receipt table -> paths n/a.
        l4 = _FakeL4(
            "obs",
            {"RECEIPT": {}, "PRIMARY": {"RV": None}, "QUALITY_CONTROL": {}},
            None,
        )
        text = science_run_summary(l4, 1.0)
        assert "RV:       n/a (no science combine)" in text
        assert "inputs:   n/a" in text
        assert "outputs:  n/a" in text
        assert "bias=n/a" in text
