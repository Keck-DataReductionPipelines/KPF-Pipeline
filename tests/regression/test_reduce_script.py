"""Tests for scripts/process/reduce.py: the ``kpfpipe run`` leaf.

Covers the shortcut/`-r`/`-c` resolution, the recipe-kind guards, and
clear_stale_outputs. Each test drives ``main(argv)`` against a tiny stub recipe
(which records the resolved config) with logging stubbed out, so no real
reduction runs. ``resolve_logging`` is unit-tested directly (TestResolveLogging).
"""

import argparse
import logging
import os

import pytest

import recipes.kpf_drp_engineering as engineering_recipe
import recipes.kpf_drp_masters as masters_recipe
import recipes.kpf_drp_science as science_recipe
from kpfpipe.utils.config import ConfigHandler
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


@pytest.fixture
def _stub_is_standard(monkeypatch, tmp_path):
    # A type is honored only for its standard recipe: stand the stub tmp/rec.py in
    # for the science one.
    monkeypatch.setitem(red._STANDARD_RECIPES, "science", str(tmp_path / "rec.py"))


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


def _stub_logging(monkeypatch, tmp_path):
    """Stub setup_logging: no real handlers, a .log path in tmp_path.

    The driver only reports the path, so nothing is created there. Returns it.
    """
    fake_log = str(tmp_path / "kpf_rec_x_20240405T000000.log")
    monkeypatch.setattr(red, "setup_logging", lambda **kw: fake_log)
    return fake_log


def _run(monkeypatch, tmp_path, argv):
    # Only config resolution is asserted; keep the real logging stack untouched.
    _stub_logging(monkeypatch, tmp_path)
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


@pytest.mark.usefixtures("_stub_is_standard")
class TestDirShortcuts:
    def test_input_dir_overrides_data_input(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        sentinel = tmp_path / "seen.txt"
        recipe = _dirs_stub_recipe(tmp_path, sentinel)
        _run(
            monkeypatch,
            tmp_path,
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
            tmp_path,
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


@pytest.mark.usefixtures("_stub_is_standard")
class TestRunLogDir:
    """One run, one log directory -- created here, or joined from a parent script."""

    def _setup_kwargs(self, monkeypatch, tmp_path, extra):
        seen = {}
        fake_log = str(tmp_path / "kpf_rec_x_20240405T000000.log")
        monkeypatch.setattr(
            red, "setup_logging", lambda **kw: seen.update(kw) or fake_log
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
        assert "kpfpipe run builds one masters night" in capsys.readouterr().err

    def test_science_rejects_datecode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--science", "-d", "20240405"])
        assert exc.value.code == 2
        assert "kpfpipe run reduces one science frame" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "recipe, selector, message",
        [
            (
                science_recipe.__file__,
                ["-d", "20240405"],
                "kpfpipe run reduces one science frame",
            ),
            (
                masters_recipe.__file__,
                ["-o", _OID],
                "kpfpipe run builds one masters night",
            ),
        ],
    )
    def test_relative_r_gets_the_declared_types_guard(
        self, recipe, selector, message, tmp_path, monkeypatch, capsys
    ):
        # The standard-recipe match is on the resolved path, so a relative -r
        # spelling of a default recipe keeps its type and is rejected before clearing.
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
        assert "kpfpipe run reduces one engineering frame" in capsys.readouterr().err

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

        argv = ["-r", str(recipe), "-c", str(cfg), *selector, "--force_run"]
        with caplog.at_level(logging.WARNING, logger=red.logger.name):
            _run(monkeypatch, tmp_path, argv)

        assert sentinel.exists()
        assert os.path.exists(product)
        assert "custom/undeclared recipe" in caplog.text


class TestCustomRecipeGuard:
    """Any recipe but a type's standard one runs as custom: confirmed at a [y/n]
    prompt (or --force_run), and never handed the declared type's clearing."""

    def _setup(self, tmp_path, decl=_SCIENCE_DECL):
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
        recipe = tmp_path / "spoof.py"  # a custom path
        recipe.write_text(
            decl + "def main(config, args):\n"
            f"    open({str(sentinel)!r}, 'w').close()\n"
        )
        return ["-r", str(recipe), "-c", str(cfg), "-o", _OID], product, sentinel

    def _tty(self, monkeypatch, answer):
        monkeypatch.setattr(red.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda prompt: answer)

    @pytest.mark.parametrize("decl", [_SCIENCE_DECL, "RECIPE_TYPE = 'sceince'\n"])
    def test_declared_type_is_ignored_and_nothing_cleared(
        self, decl, monkeypatch, tmp_path
    ):
        # A custom recipe claiming a known (or mistyped) type gets the custom path.
        argv, product, sentinel = self._setup(tmp_path, decl)
        self._tty(monkeypatch, "y")
        _run(monkeypatch, tmp_path, argv)
        assert sentinel.exists()
        assert os.path.exists(product)

    def test_non_tty_aborts_without_running(self, monkeypatch, tmp_path):
        argv, product, sentinel = self._setup(tmp_path)
        monkeypatch.setattr(red.sys.stdin, "isatty", lambda: False, raising=False)
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, tmp_path, argv)
        assert "Run aborted" in str(exc.value.code)
        assert "--force_run" in str(exc.value.code)
        assert not sentinel.exists()

    def test_yes_runs_and_points_at_force_run(self, monkeypatch, tmp_path, capsys):
        argv, _, sentinel = self._setup(tmp_path)
        self._tty(monkeypatch, " Yes\n")
        _run(monkeypatch, tmp_path, argv)
        assert sentinel.exists()
        out = capsys.readouterr().out
        assert "Executing" in out and "--force_run" in out

    @pytest.mark.parametrize("answer", ["n", "", "maybe"])
    def test_anything_but_yes_aborts(self, answer, monkeypatch, tmp_path):
        argv, product, sentinel = self._setup(tmp_path)
        self._tty(monkeypatch, answer)
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, tmp_path, argv)
        assert exc.value.code == "Run aborted"
        assert not sentinel.exists()
        assert os.path.exists(product)

    def test_force_run_skips_the_prompt(self, monkeypatch, tmp_path):
        argv, _, sentinel = self._setup(tmp_path)
        monkeypatch.setattr("builtins.input", pytest.fail)
        _run(monkeypatch, tmp_path, [*argv, "--force_run"])
        assert sentinel.exists()

    @pytest.mark.parametrize(
        "argv",
        [
            ["--science", "-o", _OID],
            ["--masters", "-d", "20240405"],
            ["--engineering", "-o", _OID],
        ],
    )
    def test_standard_recipes_do_not_prompt(self, argv, monkeypatch):
        monkeypatch.setattr("builtins.input", pytest.fail)

        def _stop(path, overrides=None):
            raise SystemExit(0)

        monkeypatch.setattr(red, "ConfigHandler", _stop)
        with pytest.raises(SystemExit) as exc:
            red.main(argv)
        assert exc.value.code == 0

    def test_symlink_to_standard_recipe_is_standard(self, tmp_path, capsys):
        # Its type is honored: the engineering selector check fires, no prompt.
        link = tmp_path / "link.py"
        link.symlink_to(red.DEFAULT_ENGINEERING_RECIPE)
        cfg = str(_base_cfg(tmp_path))
        with pytest.raises(SystemExit) as exc:
            red.main(["-r", str(link), "-c", cfg, "-d", "20240405"])
        assert exc.value.code == 2
        assert "kpfpipe run reduces one engineering frame" in capsys.readouterr().err


@pytest.mark.usefixtures("_stub_is_standard")
class TestRecipeLoading:
    """reduce.py's recipe-loading failure branches. All in-process via main(),
    no subprocess."""

    def test_missing_recipe_file_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        _stub_logging(monkeypatch, tmp_path)
        with pytest.raises(SystemExit, match="Recipe file not found"):
            red.main(["-r", str(tmp_path / "absent.py"), "-c", str(cfg), "-o", _OID])

    def test_recipe_without_main_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "nomain.py"
        recipe.write_text("VALUE = 1\n")
        _stub_logging(monkeypatch, tmp_path)
        with pytest.raises(SystemExit, match="has no main"):
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", _OID])

    def test_recipe_exception_is_logged_before_reraise(
        self, monkeypatch, tmp_path, caplog
    ):
        # reduce.py's one sanctioned catch-log-reraise: the only place a
        # traceback is guaranteed to reach the log before the nonzero exit
        # (DRP-RUN-08). A bare `raise` here would lose the logged traceback.
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "rec.py"
        recipe.write_text(
            _SCIENCE_DECL + "def main(config, args):\n    raise RuntimeError('boom')\n"
        )
        _stub_logging(monkeypatch, tmp_path)
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

        _stub_logging(monkeypatch, tmp_path)
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
            "science",
            science_recipe.RECIPE_STAGES,
        )

        assert not any(os.path.exists(p) for p in targets)
        assert os.path.exists(stray)  # a different obs_id's product is untouched

    def test_science_raises_when_output_root_unset(self):
        # No KPF_SCIENCE_OUTPUT -> can't know what to clear; fail loud.
        with pytest.raises(KeyError, match="KPF_SCIENCE_OUTPUT"):
            red.clear_stale_outputs(
                _Config({}),
                _args(obs_id=self._OID),
                "science",
                science_recipe.RECIPE_STAGES,
            )

    def test_masters_raises_when_output_root_unset(self):
        # Same for the masters branch.
        with pytest.raises(KeyError, match="KPF_MASTERS_OUTPUT"):
            red.clear_stale_outputs(
                _Config({}),
                _args(datecode="20240405"),
                "masters",
                masters_recipe.RECIPE_STAGES,
            )

    def test_science_noop_for_invalid_obs_id(self, tmp_path):
        # A malformed obs_id can't build a path; skip rather than raise -- the
        # recipe reports the real error.
        red.clear_stale_outputs(
            _Config({"KPF_SCIENCE_OUTPUT": str(tmp_path)}),
            _args(obs_id="KP.x"),
            "science",
            science_recipe.RECIPE_STAGES,
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
            "masters",
            masters_recipe.RECIPE_STAGES,
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
            "masters",
            masters_recipe.RECIPE_STAGES,
        )

        assert sorted(os.listdir(night)) == sorted(products[st] for st in kept)

    def test_masters_noop_when_dir_absent(self, tmp_path):
        # A never-built night has no masters dir; glob matches nothing, no error.
        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": str(tmp_path)}),
            _args(datecode="20240405"),
            "masters",
            masters_recipe.RECIPE_STAGES,
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
            "engineering",
            engineering_recipe.RECIPE_STAGES,
        )

        assert not os.path.exists(eng)
        assert os.path.exists(sci)


@pytest.mark.usefixtures("_stub_is_standard")
class TestFrameGate:
    """Junk and single-CCD frames return before the recipe runs."""

    def _main(self, monkeypatch, tmp_path, obs_id, extra=()):
        monkeypatch.setattr(red, "HeaderParser", HeaderParser)
        monkeypatch.setattr(red, "load_junk_obs_ids", load_junk_obs_ids)
        sentinel = tmp_path / "ran.txt"
        recipe = _stub_recipe(tmp_path, sentinel)
        cfg = _base_cfg(tmp_path)
        argv = ["-r", str(recipe), "-c", str(cfg), "-o", obs_id, *extra]
        _run(monkeypatch, tmp_path, [*argv, "--kpf_data_input", str(tmp_path)])
        return sentinel.exists()

    def test_junk_frame_products_are_deleted(self, monkeypatch, tmp_path):
        # A frame junked after it was reduced keeps no L1/L2/L4.
        obs_id = write_l0_tree(tmp_path, "20240405", 100)
        add_junk_obs_id(tmp_path, obs_id)
        out = str(tmp_path / "s")
        products = red.science_output_paths(obs_id, out)
        for path in products:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        extra = ["--kpf_science_output", out]
        assert not self._main(monkeypatch, tmp_path, obs_id, extra=extra)
        assert not any(os.path.exists(p) for p in products)

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


def _config_file(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


class TestResolveLogging:
    _LOGGER_TOML = '[LOGGER]\nlog_dir = "/logs/"\nlog_level = "DEBUG"\n'

    def test_resolves_config_and_tokens(self, tmp_path):
        config = ConfigHandler(_config_file(tmp_path, self._LOGGER_TOML))
        params = red.resolve_logging(
            config, "/repo/recipes/kpf_drp_science.py", "KP.1.2.3", None
        )
        assert params == {
            "log_dir": "/logs/",
            "recipe_name": "science",
            "target": "KP.1.2.3",
            "level": "DEBUG",
            "console": True,
        }

    def test_datecode_target_and_defaults(self, tmp_path):
        config = ConfigHandler(_config_file(tmp_path, '[LOGGER]\nlog_dir = "/l/"\n'))
        params = red.resolve_logging(config, "kpf_drp_masters.py", None, "20240923")
        assert params["recipe_name"] == "masters"
        assert params["target"] == "20240923"
        assert params["level"] == "INFO"  # default when config omits log_level

    def test_no_target_falls_back_to_run(self, tmp_path):
        config = ConfigHandler(_config_file(tmp_path, '[LOGGER]\nlog_dir = "/l/"\n'))
        assert red.resolve_logging(config, "custom.py", None, None)["target"] == "run"

    def test_missing_log_dir_raises(self, tmp_path):
        config = ConfigHandler(_config_file(tmp_path, "[LOGGER]\n"))
        with pytest.raises(ValueError, match="no log directory configured"):
            red.resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)

    def test_cli_override_wins(self, tmp_path):
        # The CLI passes --log_dir/--log_level through ConfigHandler overrides.
        config = ConfigHandler(
            _config_file(tmp_path, self._LOGGER_TOML),
            overrides={"LOGGER": {"log_dir": "/elsewhere/", "log_level": "INFO"}},
        )
        params = red.resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)
        assert params["log_dir"] == "/elsewhere/"
        assert params["level"] == "INFO"
