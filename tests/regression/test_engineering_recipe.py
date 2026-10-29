"""Tests for the kpf_drp_engineering recipe.

Integration tests run the recipe (L0 -> L1 -> L2) against a real star
observation from tests/testdata/L0/20240405/, with no catalog query, so the
L0 TARGETOK and L2 DATAPRL2 checkpoints fail and must not stop the run.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_filepath

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "kpf_drp_engineering.toml"
)

OBS_ID = "KP.20240405.40113.57"


def _run_recipe(out_root, masters_root):
    spec = importlib.util.spec_from_file_location(
        "kpf_drp_engineering",
        Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_engineering.py",
    )
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)

    config = ConfigHandler(
        str(CONFIG_PATH),
        overrides={
            "DATA_DIRS": {
                "KPF_DATA_INPUT": str(TESTDATA_DIR),
                "KPF_MASTERS_OUTPUT": str(masters_root),
                "KPF_ENGINEERING_OUTPUT": str(out_root),
            }
        },
    )
    recipe.main(config, argparse.Namespace(obs_id=OBS_ID))
    return kpf_filepath(OBS_ID, "L2", data_root=str(out_root))


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestEngineeringRecipe:
    @pytest.fixture(scope="class")
    def out_root(self, tmp_path_factory):
        out_root = tmp_path_factory.mktemp("engineering_out")
        _run_recipe(out_root, TESTDATA_DIR)
        return out_root

    @pytest.fixture(scope="class")
    def l2(self, out_root):
        return KPF2.from_fits(kpf_filepath(OBS_ID, "L2", data_root=str(out_root)))

    def test_l2_written(self, out_root):
        assert os.path.isfile(kpf_filepath(OBS_ID, "L2", data_root=str(out_root)))

    def test_receipt_chain(self, l2):
        modules = l2.receipt["FUNCTION"].values
        for module in (
            "image_assembly",
            "calibration_association",
            "spectral_extraction",
            "wavelength_calibration",
        ):
            assert module in modules
        assert "image_processing" not in modules
        assert l2.receipt_read_entry("calibration_association").get("wlsfile")

    def test_flux_and_wave_populated(self, l2):
        for chip in ("GREEN", "RED"):
            assert np.nanmedian(l2.data[f"{chip}_SCI2_FLUX"]) > 0
            wave = np.asarray(l2.data[f"{chip}_SCI2_WAVE"])
            assert np.all(np.diff(wave, axis=1) > 0), f"{chip} WAVE not ascending"

    @pytest.mark.parametrize("level", ["L0", "L1", "L2"])
    def test_qlp_pngs_exist(self, out_root, level):
        qlp_dir = out_root / "QLP" / "20240405" / OBS_ID / level
        assert any(qlp_dir.glob("*.png")), f"no {level} quicklook PNGs in {qlp_dir}"


@pytest.mark.slow
@pytest.mark.requires_testdata
def test_completes_without_thar_master(tmp_path):
    # An empty masters root: association finds no ThAr, WLS is skipped, and the
    # L2 is still written.
    masters_root = tmp_path / "no_masters"
    masters_root.mkdir()
    out_path = _run_recipe(tmp_path / "out", masters_root)

    assert os.path.isfile(out_path)
    l2 = KPF2.from_fits(out_path)
    modules = l2.receipt["FUNCTION"].values
    assert "spectral_extraction" in modules
    assert "wavelength_calibration" not in modules
