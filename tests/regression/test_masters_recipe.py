"""Tests for the kpf_drp_masters recipe: master production and error paths.

Integration tests use real L0 data from tests/testdata/L0/20240405/. The
FileHandler and path builders they exercise are unit-tested in test_io.py.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import FileHandler, kpf_directory, kpf_filepath
from recipes._logging import masters_run_summary

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
# Masters recipe integration (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestMastersRecipe:
    """End-to-end: FileHandler -> Bias.make_master_l1 -> to_fits."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        from kpfpipe.modules.masters.bias import Bias
        from kpfpipe.utils.kpf import get_obs_id

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
# Masters recipe order-trace stage (stacking stubbed -- wiring only)
# ---------------------------------------------------------------------------


@pytest.mark.requires_testdata
class TestMastersRecipeOrderTraceStage:
    """The recipe traces every master flat it just stacked.

    The stacking modules are stubbed out; what is under test is the wiring --
    which flat the tracer is handed, where its CSV is written, and that the trace
    reaches the run summary. OrderTrace's geometry is covered by
    test_master_order_trace.py.
    """

    @pytest.fixture
    def traced(self, tmp_path, monkeypatch):
        import argparse

        recipe = _load_masters_recipe()
        calls = {}

        class StubFileHandler:
            def __init__(self, data_dirs):
                pass

            def build_mini_database(self, datecode, cache=None):
                pass

            def build_calibration_stacks(self, cal_type, **kwargs):
                # Only the flats matter here; the other stages fall through.
                return [["/l0/KP.20240405.00020.86.fits"]] if cal_type == "flat" else []

        class StubFlat:
            def __init__(self, files, config):
                pass

            def make_master_l1(self, master_path=None):
                calls["flat_path"] = master_path

        class StubOrderTrace:
            def __init__(self, flat_path, config):
                calls["traced_flat"] = flat_path
                self.output_path = str(tmp_path / "kpf_20240405_order_trace.csv")

            def make_master(self, output_dir=None):
                calls["output_dir"] = output_dir

        monkeypatch.setattr(recipe, "FileHandler", StubFileHandler)
        monkeypatch.setattr(recipe, "Flat", StubFlat)
        monkeypatch.setattr(recipe, "OrderTrace", StubOrderTrace)

        def capture_summary(datecode, built, elapsed):
            calls["built"] = built
            return ""

        monkeypatch.setattr(recipe, "masters_run_summary", capture_summary)

        config = ConfigHandler(
            str(MASTERS_CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(TESTDATA_DIR),
                    "KPF_MASTERS_OUTPUT": str(tmp_path),
                }
            },
        )
        recipe.main(config, argparse.Namespace(datecode="20240405", obs_id=None))
        return calls

    def test_traces_the_master_flat_it_just_stacked(self, traced):
        assert traced["traced_flat"] == traced["flat_path"]

    def test_writes_the_trace_into_the_masters_directory(self, traced, tmp_path):
        expected = kpf_directory(
            kind="masters", data_root=str(tmp_path), datecode="20240405"
        )
        assert str(traced["output_dir"]) == str(expected)

    def test_reports_the_trace_in_the_run_summary(self, traced):
        traces = [entry for entry in traced["built"] if entry[0] == "order_trace"]
        assert len(traces) == 1
        _, path, n_frames = traces[0]
        assert path.endswith("_order_trace.csv")
        assert n_frames == 1


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

    @pytest.mark.slow
    @pytest.mark.requires_testdata
    def test_missing_min_stack_size_key_raises(self, tmp_path):
        # min_stack_size is a required quality gate: a [BIAS] section lacking it
        # must fail loud, not silently stack with no size gate.
        import argparse

        cfg = tmp_path / "no_min_stack.toml"
        cfg.write_text(
            f'[DATA_DIRS]\nKPF_DATA_INPUT = "{TESTDATA_DIR}"\n'
            f'KPF_MASTERS_OUTPUT = "{tmp_path}"\n[BIAS]\nmax_stack_size = 20\n'
        )
        config = ConfigHandler(str(cfg))
        args = argparse.Namespace(datecode="20240405", obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(KeyError, match="min_stack_size"):
            recipe.main(config, args)


class TestMastersSummary:
    """Unit tests for the masters_run_summary() formatter."""

    def test_built_masters_listed(self):
        text = masters_run_summary(
            "20240405",
            [
                ("bias", "/m/kpf_20240405_bias_L1.fits", 32),
                ("thar", "/m/kpf_20240405_thar_L2.fits", 12),
            ],
            240.5,
        )
        assert "masters run summary: 20240405" in text
        assert "bias   kpf_20240405_bias_L1.fits  (32 frames)" in text
        assert "thar   kpf_20240405_thar_L2.fits  (12 frames)" in text
        assert "elapsed:  240.5 s" in text
        assert text.startswith("\n\n") and text.endswith("\n\n")

    def test_no_masters_built(self):
        text = masters_run_summary("20240405", [], 1.0)
        assert "(no masters built)" in text
