"""Tests for the kpf_drp_masters recipe: end-to-end master production and error
paths.

Integration tests use real L0 data from tests/testdata/L0/20240405/. The
FileHandler and path builders these exercise are unit-tested in test_io.py.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import FileHandler, kpf_filepath

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
MASTERS_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "kpf_drp_masters.toml"
)


def _load_masters_recipe():
    spec = importlib.util.spec_from_file_location(
        "kpf_drp_masters",
        Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_masters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _min_stack_size: per-cal-type resolver (config section -> module default)
# ---------------------------------------------------------------------------


class TestMinStackSize:
    """The recipe resolves min_stack_size from each cal_type's config section,
    falling back to BaseMasterModule._DEFAULTS when the section omits it."""

    @pytest.fixture(scope="class")
    def recipe(self):
        return _load_masters_recipe()

    @pytest.fixture(scope="class")
    def config(self):
        return ConfigHandler(str(MASTERS_CONFIG_PATH))

    def test_reads_bias_section(self, recipe, config):
        assert recipe._min_stack_size(config, "bias") == 5

    def test_reads_dark_section(self, recipe, config):
        assert recipe._min_stack_size(config, "dark") == 3

    def test_falls_back_when_section_absent(self, recipe, config):
        # thar has no [THAR] config section -> the masters-module default.
        from kpfpipe.modules.masters.base import BaseMasterModule

        assert (
            recipe._min_stack_size(config, "thar")
            == BaseMasterModule._DEFAULTS["min_stack_size"]
        )


# ---------------------------------------------------------------------------
# Masters recipe integration (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMastersRecipe:
    """End-to-end recipe test: FileHandler → Bias.make_master_l1 → to_fits."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        from kpfpipe.modules.masters.bias import Bias
        from kpfpipe.utils.kpf_utils import get_obs_id

        tmp_path = tmp_path_factory.mktemp("recipe_out")
        data_root_out = str(tmp_path)

        file_handler = FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)})
        file_handler.build_mini_database("20240405")
        output_paths = []
        for files in file_handler.build_calibration_stacks("bias"):
            bias_handler = Bias(files)
            bias_l1 = bias_handler.make_master_l1()
            out_path = kpf_filepath(
                get_obs_id(files[0]), "L1", data_root=data_root_out, master="bias"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            bias_l1.to_fits(out_path)
            output_paths.append(out_path)

        return output_paths

    def test_at_least_one_master_produced(self, recipe_output):
        assert len(recipe_output) >= 1

    def test_output_files_exist(self, recipe_output):
        for path in recipe_output:
            assert os.path.isfile(path), f"Expected output not found: {path}"

    def test_output_filename_format(self, recipe_output):
        for path in recipe_output:
            fname = os.path.basename(path)
            assert "_master_bias_L1.fits" in fname

    def test_output_is_valid_fits(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert ml1.data["GREEN_IMG"] is not None
            assert ml1.data["RED_IMG"] is not None

    def test_input_files_extension_present(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert "INPUT_FILES" in ml1.extensions

    def test_input_files_extension_has_correct_count(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert len(ml1.data["INPUT_FILES"]) == 5

    def test_input_files_all_fits(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            filenames = ml1.data["INPUT_FILES"]["FILENAME"].tolist()
            assert all(f.endswith(".fits") for f in filenames)


# ---------------------------------------------------------------------------
# Masters recipe error paths
# ---------------------------------------------------------------------------


class TestMastersRecipeErrors:
    def _make_config(self, data_input, data_masters):
        return ConfigHandler(
            str(MASTERS_CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(data_input),
                    "KPF_MASTERS_OUTPUT": str(data_masters),
                }
            },
        )

    def test_nonexistent_l0_dir_raises(self, tmp_path):
        import argparse

        config = self._make_config(tmp_path, tmp_path)
        args = argparse.Namespace(datecode="20240405", obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(SystemExit, match="L0 data directory not found"):
            recipe.main(config, args)

    def test_missing_datecode_raises(self, tmp_path):
        import argparse

        config = self._make_config(tmp_path, tmp_path)
        args = argparse.Namespace(datecode=None, obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(SystemExit, match="--datecode is required"):
            recipe.main(config, args)
