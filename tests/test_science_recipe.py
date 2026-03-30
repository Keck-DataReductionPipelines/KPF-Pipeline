"""
Tests for the kpf_drp_science recipe.

Integration tests run the full recipe (L0 → L1 → L2) against a real star
observation from tests/testdata/L0/20240405/.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.pipeline import build_filepath

import importlib.util


# ---------------------------------------------------------------------------
# Test data paths and constants
# ---------------------------------------------------------------------------

TESTDATA_DIR    = Path(__file__).parent / 'testdata'
TESTDATA_L0_DIR = TESTDATA_DIR / 'L0' / '20240405'
CONFIG_PATH     = Path(__file__).parent.parent / 'configs' / 'kpf_drp_science.toml'

OBS_ID = 'KP.20240405.40113.57'

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NCOL         = DETECTOR['ccd']['ncol']


def _load_recipe():
    spec = importlib.util.spec_from_file_location(
        'kpf_drp_science',
        Path(__file__).parent.parent / 'recipes' / 'kpf_drp_science.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Science recipe integration (real L0 star data from tests/testdata/)
# ---------------------------------------------------------------------------


class TestScienceRecipe:
    """End-to-end recipe test: KPF0 → ImageAssembly → SpectralExtraction → KPF2."""

    @pytest.fixture(scope='class')
    def recipe_output(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp('science_out')

        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                'DATA_DIRS': {
                    'KPF_DATA_INPUT':  str(TESTDATA_DIR),
                    'KPF_DATA_OUTPUT': str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=OBS_ID)

        recipe = _load_recipe()
        recipe.main(config, args)

        out_path = build_filepath(OBS_ID, 'L2', data_root=str(tmp_path))
        return out_path

    def test_output_file_exists(self, recipe_output):
        assert os.path.isfile(recipe_output), f"Expected output not found: {recipe_output}"

    def test_output_filename_format(self, recipe_output):
        assert os.path.basename(recipe_output) == 'kpf_SL2_20240405T110833.fits'

    def test_output_is_valid_kpf2(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert isinstance(l2, KPF2)

    def test_green_sci2_flux_shape(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert l2.data['GREEN_SCI2_FLUX'].shape == (NORDER_GREEN, NCOL)

    def test_red_sci2_flux_shape(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert l2.data['RED_SCI2_FLUX'].shape == (NORDER_RED, NCOL)

    def test_full_trace_shape(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert l2.data['SCI2_FLUX'].shape == (NORDER_GREEN + NORDER_RED, NCOL)

    def test_flux_positive(self, recipe_output):
        """Star flux should be positive after extraction."""
        l2 = KPF2.from_fits(recipe_output)
        assert np.nanmedian(l2.data['GREEN_SCI2_FLUX']) > 0
        assert np.nanmedian(l2.data['RED_SCI2_FLUX']) > 0

    def test_variance_positive(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        assert np.nanmin(l2.data['GREEN_SCI2_VAR']) >= 0
        assert np.nanmin(l2.data['RED_SCI2_VAR']) >= 0

    def test_receipt_chain(self, recipe_output):
        l2 = KPF2.from_fits(recipe_output)
        modules = l2.receipt['Module_Name'].values
        assert 'image_assembly' in modules
        assert 'spectral_extraction' in modules


# ---------------------------------------------------------------------------
# Science recipe error paths
# ---------------------------------------------------------------------------


class TestScienceRecipeErrors:

    def test_missing_l0_file_raises(self, tmp_path):
        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                'DATA_DIRS': {
                    'KPF_DATA_INPUT':  str(tmp_path),
                    'KPF_DATA_OUTPUT': str(tmp_path),
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
                'DATA_DIRS': {
                    'KPF_DATA_INPUT':  str(tmp_path),
                    'KPF_DATA_OUTPUT': str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=None)
        recipe = _load_recipe()
        with pytest.raises(SystemExit, match="--obs_id is required"):
            recipe.main(config, args)
