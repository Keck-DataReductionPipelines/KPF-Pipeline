"""Tests for scripts/process/reduce.py: the ``kpfpipe run`` leaf.

Covers the shortcut/`-r`/`-c` resolution, the recipe-kind guards, and
clear_stale_outputs. Each test drives ``main(argv)`` against a tiny stub recipe
(which records the resolved config) with logging stubbed out, so no real
reduction runs. (``resolve_logging`` itself is unit-tested in test_logger.py.)
"""

import argparse
import logging
import os
import tempfile

import pytest

import recipes.kpf_drp_engineering as engineering_recipe
import recipes.kpf_drp_masters as masters_recipe
import recipes.kpf_drp_science as science_recipe
from kpfpipe.utils.header import HeaderParser
from kpfpipe.utils.io import load_junk_obs_ids
from scripts.process import reduce as red

from ._scripts import add_junk_obs_id, write_l0_tree

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

# A well-formed obs_id with no L0 on disk (see _pass_frame_gate).
_OID = "KP.20240405.40113.57"

# Stub recipes declare a science type with no stages.
_SCIENCE_DECL = "RECIPE_TYPE = 'science'\nRECIPE_STAGES = ()\n"


class _BothCcdsHeader:
    def __init__(self, path):
        self.DRP_KEYWORDS = {"GREEN": True, "RED": True}


@pytest.fixture(autouse=True)
def _pass_frame_gate(monkeypatch):
    # These tests use placeholder obs_ids with no L0 on disk; pass them through
    # the junk/single-CCD gate (TestFrameGate restores the real readers).
    monkeypatch.setattr(red, "HeaderParser", _BothCcdsHeader)
    monkeypatch.setattr(red, "load_junk_obs_ids", lambda data_input: set())


def _stub_recipe(tmp_path, sentinel):
    """A recipe file whose main() records the resolved KPF_DATA_INPUT."""
    recipe = tmp_path / "rec.py"
    recipe.write_text(
        _SCIENCE_DECL + "def main(config, args):\n"
        f"    with open({str(sentinel)!r}, 'w') as fh:\n"
        "        fh.write(config.get_params(['DATA_DIRS'])['KPF_DATA_INPUT'])\n"
    )
    return recipe


def _stub_logging(monkeypatch):
    """Stub setup_logging: no real handlers, but a real-looking .log path.

    The run.json sidecar is written beside the log, so the stub must return a
    path ending in .log inside a writable temp dir (never /dev/null).
    """
    base = tempfile.mkdtemp()
    fake_log = os.path.join(base, "logs", "20240405", "kpf_rec_x_20240405T000000.log")
    os.makedirs(os.path.dirname(fake_log), exist_ok=True)
    monkeypatch.setattr(red, "setup_logging", lambda **kw: fake_log)
    return fake_log


def _run(monkeypatch, argv):
    # Only config resolution is asserted; keep the real logging stack untouched.
    _stub_logging(monkeypatch)
    red.main(argv)


def _dirs_stub_recipe(tmp_path, sentinel):
    """A recipe whose main() records the resolved DATA_DIRS + [LOGGER] log_dir."""
    recipe = tmp_path / "rec.py"
    recipe.write_text(
        _SCIENCE_DECL + "def main(config, args):\n"
        "    d = config.get_params(['DATA_DIRS'])\n"
        "    lg = config.get_params(['LOGGER'])\n"
        f"    with open({str(sentinel)!r}, 'w') as fh:\n"
        "        fh.write('|'.join([d['KPF_DATA_INPUT'], d['KPF_MASTERS_OUTPUT'], "
        "d['KPF_SCIENCE_OUTPUT'], lg['log_dir']]))\n"
    )
    return recipe


def _base_cfg(tmp_path):
    cfg = tmp_path / "custom.toml"
    cfg.write_text(
        "[DATA_DIRS]\n"
        'KPF_DATA_INPUT = "/cfg/in"\n'
        'KPF_MASTERS_OUTPUT = "/cfg/m"\n'
        'KPF_SCIENCE_OUTPUT = "/cfg/s"\n'
        "[LOGGER]\n"
        'log_dir = "/cfg/l"\n'
    )
    return cfg


class TestDirShortcuts:
    def test_input_dir_overrides_data_input(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        sentinel = tmp_path / "seen.txt"
        recipe = _dirs_stub_recipe(tmp_path, sentinel)
        _run(
            monkeypatch,
            [
                "-r",
                str(recipe),
                "-c",
                str(cfg),
                "-o",
                _OID,
                "--input_dir",
                "/aliased/in",
            ],  # fmt: skip
        )
        data_input = sentinel.read_text().split("|")[0]
        assert data_input == "/aliased/in"

    def test_output_dir_sets_every_output_dir(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        sentinel = tmp_path / "seen.txt"
        recipe = _dirs_stub_recipe(tmp_path, sentinel)
        _run(
            monkeypatch,
            [
                "-r",
                str(recipe),
                "-c",
                str(cfg),
                "-o",
                _OID,
                "--output_dir",
                "/out",
            ],  # fmt: skip
        )
        data_input, masters, science, log_dir = sentinel.read_text().split("|")
        assert data_input == "/cfg/in"  # untouched by --output_dir
        assert masters == "/out" and science == "/out" and log_dir == "/out/logs"


class TestRunLogDir:
    """One run, one log directory -- created here, or joined from a parent script."""

    def _setup_kwargs(self, monkeypatch, tmp_path, extra):
        seen = {}
        # The run.json sidecar is written beside the log, so the stub must hand
        # back a writable .log path (never /dev/null).
        fake_log = tmp_path / "logs" / "kpf_rec_x_20240405T000000.log"
        fake_log.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            red, "setup_logging", lambda **kw: seen.update(kw) or str(fake_log)
        )
        recipe = _stub_recipe(tmp_path, tmp_path / "seen.txt")
        red.main(
            [
                "-r",
                str(recipe),
                "-c",
                str(_base_cfg(tmp_path)),
                "-o",
                _OID,
                *extra,
            ]  # fmt: skip
        )
        return seen

    def test_standalone_run_passes_no_run_id(self, monkeypatch, tmp_path):
        # The leaf knows nothing of run directories: it hands the configured parent
        # through untouched and leaves both minting and the join to setup_logging.
        seen = self._setup_kwargs(monkeypatch, tmp_path, [])
        assert seen["log_dir"] == "/cfg/l"
        assert seen["run_id"] is None

    def test_forwarded_run_id_is_used_verbatim(self, monkeypatch, tmp_path):
        # What an orchestrator forwards: join its run, never mint another.
        parent = "masters_20240405T010203"
        seen = self._setup_kwargs(monkeypatch, tmp_path, ["--run_id", parent])
        assert seen["run_id"] == parent

    def test_forwarded_log_dir_overrides_the_config(self, monkeypatch, tmp_path):
        # A parent forwards both halves; --log_dir arrives as a [LOGGER] override.
        seen = self._setup_kwargs(
            monkeypatch, tmp_path, ["--log_dir", "/parent", "--run_id", "masters_x"]
        )
        assert seen["log_dir"] == "/parent"


class TestShortcutOverride:
    def test_c_overrides_shortcut_config(self, monkeypatch, tmp_path):
        # --science supplies the recipe; -c overrides its default config.
        seen = {}

        def _capture(path, overrides=None):
            seen["config"] = path
            raise SystemExit(0)

        monkeypatch.setattr(red, "ConfigHandler", _capture)
        with pytest.raises(SystemExit):
            red.main(["--science", "-c", "/custom.toml", "-o", _OID])
        assert seen["config"] == "/custom.toml"

    @pytest.mark.parametrize("shortcut", ["--science", "--masters"])
    def test_shortcut_and_r_are_mutually_exclusive(self, shortcut, capsys):
        # A shortcut already names the recipe; a conflicting -r is a usage error.
        with pytest.raises(SystemExit) as exc:
            red.main([shortcut, "-r", "/x.py", "-c", "/c.toml", "-o", _OID])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


class TestGuards:
    # main() raises SystemExit with two different meanings -- argparse's 2 for a
    # usage error, and a string payload (exit 1) for a bad recipe -- so a bare
    # `raises(SystemExit)` cannot tell a rejected flag from a crash. Every case
    # here pins the code and the specific message.

    def test_masters_rejects_obs_id(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--masters", "-o", _OID])
        assert exc.value.code == 2
        assert "masters recipe takes -d/--datecode" in capsys.readouterr().err

    def test_science_rejects_datecode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--science", "-d", "20240405"])
        assert exc.value.code == 2
        assert "science recipe takes -o/--obs_id" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "recipe, selector, message",
        [
            (
                science_recipe.__file__,
                ["-d", "20240405"],
                "science recipe takes -o/--obs_id",
            ),
            (masters_recipe.__file__, ["-o", _OID], "masters recipe takes -d"),
        ],
    )
    def test_relative_r_gets_the_declared_types_guard(
        self, recipe, selector, message, tmp_path, monkeypatch, capsys
    ):
        # The guard keys on the loaded recipe's RECIPE_TYPE, not its path, so a
        # relative -r spelling of a default recipe is rejected before any clearing.
        cleared = []
        monkeypatch.setattr(red, "clear_stale_outputs", lambda *a: cleared.append(a))
        rel = os.path.relpath(recipe)
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", rel, "-c", str(_base_cfg(tmp_path)), *selector])
        assert exc.value.code == 2
        assert message in capsys.readouterr().err
        assert not cleared

    def test_engineering_rejects_datecode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--engineering", "-d", "20240405"])
        assert exc.value.code == 2
        assert "engineering recipe takes -o/--obs_id" in capsys.readouterr().err

    def test_unknown_recipe_type_is_rejected(self, tmp_path, capsys):
        # A typo must not silently downgrade a recipe to undeclared.
        recipe = tmp_path / "typo.py"
        recipe.write_text(
            "RECIPE_TYPE = 'sceince'\ndef main(config, args):\n    pass\n"
        )
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", str(recipe), "-c", str(_base_cfg(tmp_path)), "-o", _OID])
        assert exc.value.code == 2
        assert "unknown RECIPE_TYPE" in capsys.readouterr().err

    def test_undeclared_recipe_rejects_stages(self, tmp_path, capsys):
        recipe = tmp_path / "undeclared.py"
        recipe.write_text("def main(config, args):\n    pass\n")
        cfg = str(_base_cfg(tmp_path))
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", str(recipe), "-c", cfg, "--stages", "bias"])
        assert exc.value.code == 2
        assert "not in this recipe's stages" in capsys.readouterr().err

    def test_engineering_output_override_reaches_data_dirs(self, monkeypatch):
        seen = {}

        def _capture(path, overrides=None):
            seen.update(overrides or {})
            raise SystemExit(0)

        monkeypatch.setattr(red, "ConfigHandler", _capture)
        with pytest.raises(SystemExit):
            red.main(["--engineering", "-o", _OID, "--kpf_engineering_output", "/e"])
        assert seen["DATA_DIRS"] == {"KPF_ENGINEERING_OUTPUT": "/e"}

    def test_science_stages_override_config(self, monkeypatch):
        seen = {}

        def _capture(path, overrides=None):
            seen.update(overrides or {})
            raise SystemExit(0)

        monkeypatch.setattr(red, "ConfigHandler", _capture)
        with pytest.raises(SystemExit):
            red.main(["--science", "-o", _OID, "--stages", "image_assembly"])
        assert seen["STAGES"] == {
            st: st == "image_assembly" for st in science_recipe.RECIPE_STAGES
        }

    @pytest.mark.parametrize(
        "argv",
        [
            ["--science", "-o", _OID, "--stages", "bias"],
            ["--masters", "-d", "20240405", "--stages", "image_assembly"],
        ],
    )
    def test_rejects_the_other_recipes_stages(self, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(argv)
        assert exc.value.code == 2
        assert "not in this recipe's stages" in capsys.readouterr().err

    def test_stages_override_config(self, monkeypatch):
        # --stages switches on exactly the named stages in the config's [STAGES].
        seen = {}

        def _capture(path, overrides=None):
            seen.update(overrides or {})
            raise SystemExit(0)

        monkeypatch.setattr(red, "ConfigHandler", _capture)
        with pytest.raises(SystemExit):
            red.main(["--masters", "-d", "20240405", "--stages", "flat", "dark"])
        assert seen["STAGES"] == {
            "bias": False,
            "dark": True,
            "flat": True,
            "order_trace": False,
            "thar": False,
        }

    def test_unknown_stage_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--masters", "-d", "20240405", "--stages", "biass"])
        assert exc.value.code == 2
        assert "not in this recipe's stages" in capsys.readouterr().err

    def test_missing_recipe_errors(self, capsys):
        # No --masters/--science/-r.
        with pytest.raises(SystemExit) as exc:
            red.main(["-o", _OID])
        assert exc.value.code == 2
        assert "one of the arguments" in capsys.readouterr().err

    def test_r_requires_c(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", "/x.py", "-o", _OID])
        assert exc.value.code == 2
        assert "-r/--recipe requires -c/--config" in capsys.readouterr().err

    def test_masters_and_science_are_mutually_exclusive(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--masters", "--science", "-d", "20240405"])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err

    def test_datecode_and_obs_id_are_mutually_exclusive(self, capsys):
        # The guard against running a masters recipe on a science target.
        with pytest.raises(SystemExit) as exc:
            red.main(["--science", "-d", "20240405", "-o", _OID])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


class TestUndeclaredRecipe:
    """A custom recipe without RECIPE_TYPE runs as-is: any or no selector, a warning,
    and no clearing of any output type's products."""

    @pytest.mark.parametrize(
        "selector",
        [["-o", _OID], ["-d", "20240405"], []],
        ids=["obs_id", "date", "none"],
    )
    def test_runs_warns_and_clears_nothing(
        self, selector, monkeypatch, tmp_path, caplog
    ):
        science_root = tmp_path / "sci"
        cfg = tmp_path / "custom.toml"
        cfg.write_text(
            "[DATA_DIRS]\n"
            'KPF_DATA_INPUT = "/cfg/in"\n'
            'KPF_MASTERS_OUTPUT = "/cfg/m"\n'
            f'KPF_SCIENCE_OUTPUT = "{science_root}"\n'
            "[LOGGER]\n"
            'log_dir = "/cfg/l"\n'
        )
        product = red.kpf_filepath(_OID, "L2", data_root=str(science_root))
        os.makedirs(os.path.dirname(product))
        open(product, "w").close()
        sentinel = tmp_path / "ran.txt"
        recipe = tmp_path / "custom.py"
        recipe.write_text(
            f"def main(config, args):\n    open({str(sentinel)!r}, 'w').close()\n"
        )

        with caplog.at_level(logging.WARNING, logger=red.logger.name):
            _run(monkeypatch, ["-r", str(recipe), "-c", str(cfg), *selector])

        assert sentinel.exists()
        assert os.path.exists(product)
        assert "declares no RECIPE_TYPE" in caplog.text


class TestRecipeLoading:
    """reduce.py's recipe-loading failure branches. All in-process via main(),
    no subprocess."""

    def test_missing_recipe_file_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        _stub_logging(monkeypatch)
        with pytest.raises(SystemExit, match="Recipe file not found"):
            red.main(["-r", str(tmp_path / "absent.py"), "-c", str(cfg), "-o", _OID])

    def test_recipe_without_main_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "nomain.py"
        recipe.write_text("VALUE = 1\n")
        _stub_logging(monkeypatch)
        with pytest.raises(SystemExit, match="has no main"):
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", _OID])

    def test_recipe_exception_is_logged_before_reraise(
        self, monkeypatch, tmp_path, caplog
    ):
        # reduce.py's one sanctioned catch-log-reraise: the only place a
        # traceback is guaranteed to reach the log before the nonzero exit
        # (DRP-RUN-08). A bare `raise` here would lose the logged traceback.
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "boom.py"
        recipe.write_text(
            _SCIENCE_DECL + "def main(config, args):\n    raise RuntimeError('boom')\n"
        )
        _stub_logging(monkeypatch)
        caplog.set_level(logging.CRITICAL)

        with pytest.raises(RuntimeError, match="boom"):
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", _OID])

        assert "uncaught exception; pipeline aborted" in caplog.text

    def test_bad_recipe_leaves_prior_products_alone(self, monkeypatch, tmp_path):
        # A mistyped -r must not cost a night's L1/L2/L4: clearing is destructive and
        # unconditional, so it runs only once the recipe is known to load.
        cfg = tmp_path / "sci.toml"
        science_root = tmp_path / "sci"
        cfg.write_text(
            "[DATA_DIRS]\n"
            'KPF_DATA_INPUT = "/cfg/in"\n'
            'KPF_MASTERS_OUTPUT = "/cfg/m"\n'
            f'KPF_SCIENCE_OUTPUT = "{science_root}"\n'
            "[LOGGER]\n"
            'log_dir = "/cfg/l"\n'
        )
        oid = "KP.20240405.40113.57"
        product = red.kpf_filepath(oid, "L1", data_root=str(science_root))
        os.makedirs(os.path.dirname(product), exist_ok=True)
        open(product, "w").close()

        _stub_logging(monkeypatch)
        with pytest.raises(SystemExit, match="Recipe file not found"):
            red.main(["-r", str(tmp_path / "absent.py"), "-c", str(cfg), "-o", oid])

        assert os.path.exists(product)

    def test_missing_log_dir_is_a_usage_error(self, monkeypatch, tmp_path):
        # resolve_logging's ValueError must surface as a clean usage error
        # (DRP-RUN-07), not a traceback out of main().
        cfg = tmp_path / "nolog.toml"
        cfg.write_text(
            "[DATA_DIRS]\n"
            'KPF_DATA_INPUT = "/cfg/in"\n'
            'KPF_MASTERS_OUTPUT = "/cfg/m"\n'
            'KPF_SCIENCE_OUTPUT = "/cfg/s"\n'
            "[LOGGER]\n"
        )
        recipe = _stub_recipe(tmp_path, tmp_path / "seen.txt")
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", _OID])
        assert exc.value.code == 2


class _Config:
    """Minimal ConfigHandler stub yielding the DATA_DIRS and [STAGES] that
    clear_stale_outputs reads; `stages` (default all) are the ones switched on."""

    def __init__(self, data_dirs, stages=None):
        self._data_dirs = data_dirs
        self._stages = {
            st: st in (stages or masters_recipe.RECIPE_STAGES)
            for st in masters_recipe.RECIPE_STAGES
        }

    def get_params(self, keys):
        return self._stages if keys == ["STAGES"] else self._data_dirs


def _args(obs_id=None, datecode=None):
    return argparse.Namespace(obs_id=obs_id, datecode=datecode)


class TestClearStaleOutputs:
    _OID = "KP.20240405.40113.57"

    def test_science_removes_l1_l2_l4_for_obs_id(self, tmp_path):
        science_root = str(tmp_path / "sci")
        targets = []
        for level in ("L1", "L2", "L4"):
            p = red.kpf_filepath(self._OID, level, data_root=science_root)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
            targets.append(p)
        stray = os.path.join(os.path.dirname(targets[0]), "kpf_L1_other.fits")
        open(stray, "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_SCIENCE_OUTPUT": science_root}),
            _args(obs_id=self._OID),
            science_recipe,
        )

        assert not any(os.path.exists(p) for p in targets)
        assert os.path.exists(stray)  # a different obs_id's product is untouched

    def test_science_raises_when_output_root_unset(self):
        # No KPF_SCIENCE_OUTPUT -> can't know what to clear; fail loud.
        with pytest.raises(KeyError, match="KPF_SCIENCE_OUTPUT"):
            red.clear_stale_outputs(
                _Config({}), _args(obs_id=self._OID), science_recipe
            )

    def test_masters_raises_when_output_root_unset(self):
        # Same for the masters branch.
        with pytest.raises(KeyError, match="KPF_MASTERS_OUTPUT"):
            red.clear_stale_outputs(
                _Config({}), _args(datecode="20240405"), masters_recipe
            )

    def test_science_noop_for_invalid_obs_id(self, tmp_path):
        # A malformed obs_id can't build a path; skip rather than raise -- the
        # recipe reports the real error.
        red.clear_stale_outputs(
            _Config({"KPF_SCIENCE_OUTPUT": str(tmp_path)}),
            _args(obs_id="KP.x"),
            science_recipe,
        )

    def test_masters_removes_night_products_and_stack_subdir(self, tmp_path):
        masters_root = str(tmp_path / "m")
        night = os.path.join(masters_root, "masters", "20240405")
        os.makedirs(night)
        koaid = "KP.20240405.40113.57"
        removed = [
            f"{koaid}_master_bias_L1.fits",
            f"{koaid}_master_dark_L1.fits",
            f"{koaid}_master_thar_L2.fits",
        ]
        kept = [
            "20240405_L0.csv",  # a stray non-master artifact
            f"{koaid}_master_bias_L1.txt",  # not a .fits/.h5 product
        ]
        for name in removed + kept:
            open(os.path.join(night, name), "w").close()

        # The thar_L2/ subdir (per-frame ThAr L2s + diagnostics HDF5) goes
        # wholesale with the thar master.
        stack_subdir = os.path.join(night, "thar_L2")
        os.makedirs(stack_subdir)
        stack_files = [
            f"{koaid}_thar_l2_diagnostics.h5",
            "KP.20240405.40113.57_thar_L2.fits",
            "KP.20240405.40200.00_thar_L2.fits",
        ]
        for name in stack_files:
            open(os.path.join(stack_subdir, name), "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": masters_root}),
            _args(datecode="20240405"),
            masters_recipe,
        )

        for name in removed:
            assert not os.path.exists(os.path.join(night, name)), name
        assert not os.path.exists(stack_subdir)
        for name in kept:
            assert os.path.exists(os.path.join(night, name)), name

    @pytest.mark.parametrize(
        "stages, kept",
        [
            (["dark"], ["bias"]),
            (["order_trace"], ["bias", "dark", "flat"]),
            (["dark", "thar"], ["bias"]),  # from the earliest selected stage on
            (None, []),
        ],
    )
    def test_masters_stage_run_clears_its_stage_and_downstream(
        self, tmp_path, stages, kept
    ):
        # A staged batch run keeps the earlier stages' masters it builds on, but
        # never leaves a later stage's master built from a replaced one.
        masters_root = str(tmp_path / "m")
        night = os.path.join(masters_root, "masters", "20240405")
        os.makedirs(os.path.join(night, "thar_L2"))
        koaid = "KP.20240405.40113.57"
        products = {
            "bias": f"{koaid}_master_bias_L1.fits",
            "dark": f"{koaid}_master_dark_L1.fits",
            "flat": f"{koaid}_master_flat_L1.fits",
            "order_trace": f"{koaid}_master_order_trace.csv",
            "thar": f"{koaid}_master_thar_L2.fits",
        }
        for name in products.values():
            open(os.path.join(night, name), "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": masters_root}, stages=stages),
            _args(datecode="20240405"),
            masters_recipe,
        )

        assert sorted(os.listdir(night)) == sorted(products[st] for st in kept)

    def test_masters_noop_when_dir_absent(self, tmp_path):
        # A never-built night has no masters dir; glob matches nothing, no error.
        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": str(tmp_path)}),
            _args(datecode="20240405"),
            masters_recipe,
        )

    def test_engineering_removes_only_its_own_l2(self, tmp_path):
        # Same root for both: the engineering clear must leave the science L2 alone.
        root = str(tmp_path)
        eng = red.kpf_filepath(self._OID, "L2", data_root=root, engineering=True)
        sci = red.kpf_filepath(self._OID, "L2", data_root=root)
        for p in (eng, sci):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_ENGINEERING_OUTPUT": root, "KPF_SCIENCE_OUTPUT": root}),
            _args(obs_id=self._OID),
            engineering_recipe,
        )

        assert not os.path.exists(eng)
        assert os.path.exists(sci)


# ---------------------------------------------------------------------------
# run.json sidecar
# ---------------------------------------------------------------------------


def _record_cfg(tmp_path):
    cfg = tmp_path / "rec.toml"
    cfg.write_text(
        "[DATA_DIRS]\n"
        f'KPF_DATA_INPUT = "{tmp_path / "in"}"\n'
        f'KPF_MASTERS_OUTPUT = "{tmp_path / "m"}"\n'
        f'KPF_SCIENCE_OUTPUT = "{tmp_path / "s"}"\n'
        "[LOGGER]\n"
        f'log_dir = "{tmp_path / "logs"}"\n'
        "console = false\n"
    )
    return cfg


class TestFrameGate:
    """Junk and single-CCD frames return before the recipe runs."""

    def _main(self, monkeypatch, tmp_path, obs_id):
        monkeypatch.setattr(red, "HeaderParser", HeaderParser)
        monkeypatch.setattr(red, "load_junk_obs_ids", load_junk_obs_ids)
        sentinel = tmp_path / "ran.txt"
        recipe = _stub_recipe(tmp_path, sentinel)
        cfg = _base_cfg(tmp_path)
        argv = ["-r", str(recipe), "-c", str(cfg), "-o", obs_id]
        _run(monkeypatch, [*argv, "--kpf_data_input", str(tmp_path)])
        return sentinel.exists()

    def test_junk_frame_is_skipped(self, monkeypatch, tmp_path, caplog):
        obs_id = write_l0_tree(tmp_path, "20240405", 100)
        add_junk_obs_id(tmp_path, obs_id)
        with caplog.at_level(logging.INFO, logger=red.logger.name):
            assert not self._main(monkeypatch, tmp_path, obs_id)
        assert f"skipping {obs_id}: junk frame" in caplog.text

    def test_single_ccd_frame_is_skipped(self, monkeypatch, tmp_path, caplog):
        obs_id = write_l0_tree(tmp_path, "20240405", 100, extra_cards={"RED": "NO"})
        with caplog.at_level(logging.WARNING, logger=red.logger.name):
            assert not self._main(monkeypatch, tmp_path, obs_id)
        assert "DRP does not yet support single-CCD exposures" in caplog.text

    def test_both_ccds_frame_runs(self, monkeypatch, tmp_path):
        obs_id = write_l0_tree(tmp_path, "20240405", 100)
        assert self._main(monkeypatch, tmp_path, obs_id)
