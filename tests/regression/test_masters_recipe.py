"""Tests for the kpf_drp_masters recipe: stage wiring and error paths.

The recipe's real ``main()`` is called with every stage it resolves replaced by a
recorder, so what is under test is the wiring -- stage order, which stack each
stage is handed, where each master is written, and which config section supplied
each stage's stacking limits. The masters themselves are built from real L0 data
in test_master_{bias,dark,flat}.py; the FileHandler and path builders are
unit-tested in test_io.py.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import pytest

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_directory, kpf_filepath
from kpfpipe.utils.kpf import get_obs_id
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
# Stage wiring: the real main() runs with every collaborator replaced.
#
# This is the canonical recipe-wiring harness; test_science_recipe.py mirrors it.
# The rules it encodes: load the recipe and call its real main() rather than
# re-implementing the loop; monkeypatch collaborators on the *loaded module*, so
# what is asserted is the recipe's own wiring; record every stage into one list
# whose order is itself the assertion; and model "nothing to stack" the way
# production does -- by raising, never by returning an empty list.
# ---------------------------------------------------------------------------

# Three frames per stack: above no gate, below every max, and enough that a
# dropped stack is visible in the run summary's frame counts.
STACK_FILES = {
    "bias": [
        "/l0/20240405/KP.20240405.03637.74.fits",
        "/l0/20240405/KP.20240405.03687.64.fits",
        "/l0/20240405/KP.20240405.03737.52.fits",
    ],
    "dark": [
        "/l0/20240405/KP.20240405.04184.73.fits",
        "/l0/20240405/KP.20240405.04484.61.fits",
        "/l0/20240405/KP.20240405.04784.49.fits",
    ],
    "flat": [
        "/l0/20240405/KP.20240405.00020.86.fits",
        "/l0/20240405/KP.20240405.00120.74.fits",
        "/l0/20240405/KP.20240405.00220.62.fits",
    ],
    "thar": [
        "/l0/20240405/KP.20240405.63499.95.fits",
        "/l0/20240405/KP.20240405.63599.83.fits",
        "/l0/20240405/KP.20240405.63699.71.fits",
    ],
}

# (groupby, min_stack_size, max_stack_size) each stage must read from its OWN
# section of configs/kpf_drp_masters.toml. Reading [BIAS]'s limits for the flats,
# or grouping darks by time_of_day, changes which frames are combined into a
# master -- the RV-stability surface -- so the section-to-stage mapping is
# asserted, not the numbers alone.
STACK_LIMITS = {
    "bias": ("time_of_day", 5, 20),
    "dark": ("obs_night", 3, 20),
    "flat": ("time_of_day", 5, 20),
    "thar": ("time_of_day", 5, 20),
}

# Output level per stage, i.e. which kpf_filepath level token the recipe must use.
STACK_LEVELS = {"bias": "L1", "dark": "L1", "flat": "L1", "thar": "L2"}


def _wire_recipe(tmp_path, monkeypatch, stacks):
    """Load the recipe, replace every stage on it, and return it ready to run.

    ``stacks`` maps a cal type to the list of stacks ``build_calibration_stacks``
    returns for it, or to an exception instance to raise instead -- the two
    outcomes production has (``kpfpipe/utils/io.py`` raises when no cluster meets
    ``min_stack_size``).

    Returns ``(recipe, config, args, record)``. ``record["calls"]`` holds one
    ``(stage, files, master_path, stack_kwargs)`` tuple per stage in call order;
    ``stack_kwargs`` are the keywords the stage's stack source was called with.
    """
    recipe = _load_masters_recipe()
    record = {"calls": [], "built": []}
    stack_kwargs = {}

    (tmp_path / "L0" / "20240405").mkdir(parents=True, exist_ok=True)

    class StubFileHandler:
        def __init__(self, data_dirs):
            pass

        def build_mini_database(self, datecode, cache=None):
            pass

        def build_calibration_stacks(self, cal_type, **kwargs):
            stack_kwargs[cal_type] = kwargs
            outcome = stacks[cal_type]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def _stage(name):
        class StubStage:
            def __init__(self, files, config):
                self._files = files

            def _record(self, master_path):
                record["calls"].append(
                    (name, tuple(self._files), master_path, stack_kwargs[name])
                )

            def make_master_l1(self, master_path=None):
                self._record(master_path)

            def make_master_l2(self, master_path=None):
                self._record(master_path)

        return StubStage

    class StubOrderTrace:
        def __init__(self, flat_path, config):
            self._flat_path = flat_path
            self.output_path = None

        def make_master(self, output_dir=None):
            self.output_path = os.path.join(
                str(output_dir), "KP.20240405.00020.86_master_order_trace.csv"
            )
            record["calls"].append(
                ("order_trace", (self._flat_path,), self.output_path, {})
            )
            record["trace_output_dir"] = output_dir

    monkeypatch.setattr(recipe, "FileHandler", StubFileHandler)
    monkeypatch.setattr(recipe, "Bias", _stage("bias"))
    monkeypatch.setattr(recipe, "Dark", _stage("dark"))
    monkeypatch.setattr(recipe, "Flat", _stage("flat"))
    monkeypatch.setattr(recipe, "WLS", _stage("thar"))
    monkeypatch.setattr(recipe, "OrderTrace", StubOrderTrace)

    def capture_summary(datecode, built, elapsed):
        record["built"] = built
        return ""

    monkeypatch.setattr(recipe, "masters_run_summary", capture_summary)

    config = ConfigHandler(
        str(MASTERS_CONFIG_PATH),
        overrides={
            "DATA_DIRS": {
                "KPF_DATA_INPUT": str(tmp_path),
                "KPF_MASTERS_OUTPUT": str(tmp_path),
            }
        },
    )
    args = argparse.Namespace(datecode="20240405", obs_id=None)
    return recipe, config, args, record


class TestMastersRecipeStages:
    """Every stage the recipe drives, in order, with the paths and limits it uses."""

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        recipe, config, args, record = _wire_recipe(
            tmp_path,
            monkeypatch,
            {cal_type: [files] for cal_type, files in STACK_FILES.items()},
        )
        recipe.main(config, args)
        return record

    def test_stages_run_in_dependency_order(self, run):
        # Bias before dark and flat so CalibrationAssociation finds the bias this
        # run just wrote; flat before the trace, whose only input is that flat.
        assert [call[0] for call in run["calls"]] == [
            "bias",
            "dark",
            "flat",
            "order_trace",
            "thar",
        ]

    def test_each_stage_stacks_the_files_the_handler_returned(self, run):
        stacked = {call[0]: call[1] for call in run["calls"]}
        for cal_type, files in STACK_FILES.items():
            assert stacked[cal_type] == tuple(files)

    def test_each_master_is_written_to_its_convention_path(self, run, tmp_path):
        written = {call[0]: call[2] for call in run["calls"]}
        for cal_type, level in STACK_LEVELS.items():
            assert written[cal_type] == kpf_filepath(
                get_obs_id(STACK_FILES[cal_type][0]),
                level,
                data_root=str(tmp_path),
                master=cal_type,
            )

    def test_each_stage_reads_its_own_config_section(self, run):
        limits = {call[0]: call[3] for call in run["calls"]}
        for cal_type, (groupby, min_size, max_size) in STACK_LIMITS.items():
            assert limits[cal_type] == {
                "min_stack_size": min_size,
                "max_stack_size": max_size,
                "groupby": groupby,
            }

    def test_traces_the_master_flat_it_just_stacked(self, run):
        traced = [call for call in run["calls"] if call[0] == "order_trace"][0]
        flat = [call for call in run["calls"] if call[0] == "flat"][0]
        assert traced[1] == (flat[2],)

    def test_writes_the_trace_into_the_masters_directory(self, run, tmp_path):
        expected = kpf_directory(
            kind="masters", data_root=str(tmp_path), datecode="20240405"
        )
        assert str(run["trace_output_dir"]) == str(expected)

    def test_run_summary_lists_every_master_built(self, run):
        assert [entry[0] for entry in run["built"]] == [
            "bias",
            "dark",
            "flat",
            "order_trace",
            "thar",
        ]
        frames = {entry[0]: entry[2] for entry in run["built"]}
        assert frames["order_trace"] == 1
        assert all(frames[cal_type] == 3 for cal_type in STACK_FILES)


class TestMastersRecipeStackFailure:
    """A night with nothing to stack for one cal type.

    ``build_calibration_stacks`` raises when no cluster meets ``min_stack_size``
    (``kpfpipe/utils/io.py``); it never returns an empty list. The recipe has no
    try/except, so the run aborts mid-way -- after the earlier masters are
    already on disk. That partial-night behaviour is pinned here so neither a
    fail-soft ``try/except`` nor a ``min_stack_size`` change alters it silently.
    """

    @pytest.fixture
    def wired(self, tmp_path, monkeypatch):
        stacks = {cal_type: [files] for cal_type, files in STACK_FILES.items()}
        stacks["dark"] = ValueError("no dark cluster meets min_stack_size for 20240405")
        return _wire_recipe(tmp_path, monkeypatch, stacks)

    def test_propagates_the_stack_failure(self, wired):
        recipe, config, args, _ = wired
        with pytest.raises(ValueError, match="min_stack_size"):
            recipe.main(config, args)

    def test_aborts_after_the_stage_that_already_ran(self, wired):
        recipe, config, args, record = wired
        with pytest.raises(ValueError):
            recipe.main(config, args)
        # The bias masters are on disk; flat, order trace and WLS never happen.
        assert [call[0] for call in record["calls"]] == ["bias"]


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
        config = self._make_config(tmp_path, tmp_path)
        args = argparse.Namespace(datecode="20240405", obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(SystemExit, match="L0 data directory not found"):
            recipe.main(config, args)

    def test_missing_datecode_raises(self, tmp_path):
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
