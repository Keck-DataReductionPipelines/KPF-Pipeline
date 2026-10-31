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
import logging
import os
from pathlib import Path

import pytest

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_directory, kpf_filename, kpf_filepath
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
    "dark": ("hst_day", 3, 20),
    "flat": ("time_of_day", 5, 20),
    "thar": ("time_of_day", 5, 20),
}

# Output level per stage, i.e. the level token in each master's standard filename.
STACK_LEVELS = {"bias": "L1", "dark": "L1", "flat": "L1", "thar": "L2"}


def _wire_recipe(tmp_path, monkeypatch, stacks, failing=()):
    """Load the recipe, replace every stage on it, and return it ready to run.

    ``stacks`` maps a cal type to the list of stacks ``build_calibration_stacks``
    returns for it -- empty when the night has none (``kpfpipe/utils/io.py``
    warns and returns ``[]`` when no cluster meets ``min_stack_size``). A stack
    whose first file is in ``failing`` raises ``ValueError`` from its make_master
    call, as a failed stack gate does.

    Returns ``(recipe, config, args, record)``. ``record["calls"]`` holds one
    ``(stage, files, output_path, stack_kwargs)`` tuple per stage in call order;
    ``stack_kwargs`` are the keywords the stage's stack source was called with.
    ``record["output_dirs"]`` holds the directory each stage was handed.
    ``record["on_disk"]`` maps a cal type to the masters ``find_masters`` returns:
    those written this run, plus any a test seeds as written by an earlier run.
    """
    recipe = _load_masters_recipe()
    record = {"calls": [], "built": [], "failed": [], "output_dirs": {}, "on_disk": {}}
    stack_kwargs = {}

    (tmp_path / "L0" / "20240405").mkdir(parents=True, exist_ok=True)

    class StubFileHandler:
        def __init__(self, data_dirs):
            pass

        def build_mini_database(self, datecode, cache=None):
            pass

        def build_calibration_stacks(self, cal_type, **kwargs):
            stack_kwargs[cal_type] = kwargs
            return stacks[cal_type]

        def find_masters(self, cal_type, level, datecode):
            return sorted(record["on_disk"].get(cal_type, []))

    def _stage(name):
        class StubStage:
            def __init__(self, files, config):
                self._files = files
                self.output_path = None

            def _record(self, level, output_dir):
                if self._files[0] in failing:
                    raise ValueError("only 1 of 3 frames processed")
                # Stand in for the module naming its own output: the standard
                # name for the first frame it stacked, in the directory given.
                self.output_path = os.path.join(
                    str(output_dir),
                    kpf_filename(get_obs_id(self._files[0]), level, master=name),
                )
                record["calls"].append(
                    (name, tuple(self._files), self.output_path, stack_kwargs[name])
                )
                record["output_dirs"][name] = output_dir
                record["on_disk"].setdefault(name, []).append(self.output_path)

            def make_master_l1(self, output_dir=None):
                self._record("L1", output_dir)

            def make_master_l2(self, output_dir=None):
                self._record("L2", output_dir)

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
            record["output_dirs"]["order_trace"] = output_dir

    monkeypatch.setattr(recipe, "FileHandler", StubFileHandler)
    monkeypatch.setattr(recipe, "Bias", _stage("bias"))
    monkeypatch.setattr(recipe, "Dark", _stage("dark"))
    monkeypatch.setattr(recipe, "Flat", _stage("flat"))
    monkeypatch.setattr(recipe, "WLS", _stage("thar"))
    monkeypatch.setattr(recipe, "OrderTrace", StubOrderTrace)

    def capture_summary(datecode, stages, built, failed, elapsed):
        record["built"] = built
        record["failed"] = failed
        return "<summary>"

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

    def test_every_stage_writes_into_the_masters_directory(self, run, tmp_path):
        # The recipe hands out a directory; each module names its own file.
        expected = kpf_directory(
            kind="masters", data_root=str(tmp_path), datecode="20240405"
        )
        assert set(run["output_dirs"]) == {*STACK_LEVELS, "order_trace"}
        for output_dir in run["output_dirs"].values():
            assert str(output_dir) == str(expected)

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

    ``build_calibration_stacks`` returns no stacks (with a warning) when no
    cluster meets ``min_stack_size`` (``kpfpipe/utils/io.py``). The recipe moves on,
    so later stages can still use masters from nearby nights, then fails the run
    naming the empty stage.
    """

    @pytest.fixture
    def wired(self, tmp_path, monkeypatch):
        stacks = {cal_type: [files] for cal_type, files in STACK_FILES.items()}
        stacks["dark"] = []
        return _wire_recipe(tmp_path, monkeypatch, stacks)

    def test_runs_the_later_stages_then_fails(self, wired):
        recipe, config, args, record = wired
        with pytest.raises(RuntimeError, match=r"no master built for: dark;"):
            recipe.main(config, args)
        assert [call[0] for call in record["calls"]] == [
            "bias",
            "flat",
            "order_trace",
            "thar",
        ]


class TestMastersRecipeMasterFailure:
    """A failed master is logged and skipped; a step with no master fails the run
    after the later steps have run."""

    @staticmethod
    def _stacks(dark_stacks):
        stacks = {cal_type: [files] for cal_type, files in STACK_FILES.items()}
        stacks["dark"] = dark_stacks
        return stacks

    def test_continues_past_a_failed_master(self, tmp_path, monkeypatch, caplog):
        darks = STACK_FILES["dark"]
        recipe, config, args, record = _wire_recipe(
            tmp_path,
            monkeypatch,
            self._stacks([darks[:1], darks[1:]]),
            failing={darks[0]},
        )
        with caplog.at_level(logging.WARNING):
            recipe.main(config, args)
        assert f"dark master from {darks[0]} failed" in caplog.text
        assert [entry[:2] for entry in record["failed"]] == [("dark", darks[0])]
        # Any failure raises the summary itself to WARNING.
        summary = [r for r in caplog.records if r.getMessage() == "<summary>"]
        assert [r.levelno for r in summary] == [logging.WARNING]
        assert [entry[0] for entry in record["built"]] == [
            "bias",
            "dark",
            "flat",
            "order_trace",
            "thar",
        ]

    def test_fails_after_the_later_steps_when_every_master_of_a_step_fails(
        self, tmp_path, monkeypatch
    ):
        darks = STACK_FILES["dark"]
        recipe, config, args, record = _wire_recipe(
            tmp_path,
            monkeypatch,
            self._stacks([darks[:1], darks[1:]]),
            failing={darks[0], darks[1]},
        )
        with pytest.raises(RuntimeError, match=r"no master built for: dark;"):
            recipe.main(config, args)
        assert [call[0] for call in record["calls"]] == [
            "bias",
            "flat",
            "order_trace",
            "thar",
        ]
        # The summary is still logged on failure, with both failures in it.
        assert [entry[1] for entry in record["failed"]] == [darks[0], darks[1]]


class TestMastersRecipeStageSelection:
    """Config [STAGES] runs a subset, as the batch does one stage at a time."""

    @staticmethod
    def _wired(tmp_path, monkeypatch, stages):
        recipe, config, args, record = _wire_recipe(
            tmp_path,
            monkeypatch,
            {cal_type: [files] for cal_type, files in STACK_FILES.items()},
        )
        config.config["STAGES"] = {st: st in stages for st in recipe.RECIPE_STAGES}
        return recipe, config, args, record

    def test_runs_only_the_selected_stage(self, tmp_path, monkeypatch):
        recipe, config, args, record = self._wired(tmp_path, monkeypatch, ["dark"])
        recipe.main(config, args)
        assert [call[0] for call in record["calls"]] == ["dark"]
        assert [entry[0] for entry in record["built"]] == ["dark"]

    def test_traces_a_flat_written_by_an_earlier_run(self, tmp_path, monkeypatch):
        recipe, config, args, record = self._wired(
            tmp_path, monkeypatch, ["order_trace"]
        )
        flat = str(tmp_path / "KP.20240405.00020.86_master_flat_L1.fits")
        record["on_disk"]["flat"] = [flat]
        recipe.main(config, args)
        assert [call[:2] for call in record["calls"]] == [("order_trace", (flat,))]

    def test_fails_naming_only_selected_stages_that_built_nothing(
        self, tmp_path, monkeypatch
    ):
        recipe, config, args, record = self._wired(
            tmp_path, monkeypatch, ["order_trace"]
        )
        with pytest.raises(RuntimeError, match=r"no master built for: order_trace;"):
            recipe.main(config, args)


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
            ["bias", "thar"],
            [
                ("bias", "/m/kpf_20240405_bias_L1.fits", 32),
                ("thar", "/m/kpf_20240405_thar_L2.fits", 12),
            ],
            [],
            240.5,
        )
        assert "===== masters run summary: 20240405 (bias, thar) =====" in text
        assert "bias   kpf_20240405_bias_L1.fits  (32 frames)" in text
        assert "thar   kpf_20240405_thar_L2.fits  (12 frames)" in text
        assert "elapsed:  240.5 s" in text
        assert "FAILED" not in text
        assert text.startswith("\n\n") and text.endswith("\n\n")

    def test_no_masters_built(self):
        text = masters_run_summary("20240405", ["dark"], [], [], 1.0)
        assert "(no masters built)" in text

    def test_failed_masters_listed_under_sentinel(self):
        text = masters_run_summary(
            "20240405",
            ["bias", "dark", "thar"],
            [("bias", "/m/kpf_20240405_bias_L1.fits", 32)],
            [
                ("dark", "/l0/KP.20240405.04184.73.fits", ValueError("too few")),
                ("thar", "/l0/KP.20240405.63499.95.fits", OSError("disk full")),
            ],
            1.0,
        )
        sentinel, dark, thar = text.strip().splitlines()[2:5]
        assert sentinel == "  !!!!! 2 FAILED MASTERS !!!!!"
        assert dark == "  dark   KP.20240405.04184.73.fits  (ValueError: too few)"
        assert thar == "  thar   KP.20240405.63499.95.fits  (OSError: disk full)"
