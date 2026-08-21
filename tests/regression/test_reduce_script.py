"""Tests for scripts/processing/reduce.py: the ``kpfpipe run`` leaf.

Covers the shortcut/`-r`/`-c` resolution, the recipe-kind guards, and
clear_stale_outputs. Each test drives ``main(argv)`` against a tiny stub recipe
(which records the resolved config) with logging stubbed out, so no real
reduction runs. (``resolve_logging`` itself is unit-tested in test_logger.py.)
"""

import argparse
import logging
import os

import pytest

from scripts.processing import reduce as red

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


def _stub_recipe(tmp_path, sentinel):
    """A recipe file whose main() records the resolved KPF_DATA_INPUT."""
    recipe = tmp_path / "rec.py"
    recipe.write_text(
        "def main(config, args):\n"
        f"    with open({str(sentinel)!r}, 'w') as fh:\n"
        "        fh.write(config.get_params(['DATA_DIRS'])['KPF_DATA_INPUT'])\n"
    )
    return recipe


def _run(monkeypatch, argv):
    # Only config resolution is asserted; keep the real logging stack untouched.
    monkeypatch.setattr(red, "setup_logging", lambda **kw: "/dev/null")
    red.main(argv)


def _dirs_stub_recipe(tmp_path, sentinel):
    """A recipe whose main() records the resolved DATA_DIRS + [LOGGER] log_dir."""
    recipe = tmp_path / "rec.py"
    recipe.write_text(
        "def main(config, args):\n"
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
                "--science",
                "-r",
                str(recipe),
                "-c",
                str(cfg),
                "-o",
                "KP.x",
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
                "--science",
                "-r",
                str(recipe),
                "-c",
                str(cfg),
                "-o",
                "KP.x",
                "--output_dir",
                "/out",
            ],  # fmt: skip
        )
        data_input, masters, science, log_dir = sentinel.read_text().split("|")
        assert data_input == "/cfg/in"  # untouched by --output_dir
        assert masters == "/out" and science == "/out" and log_dir == "/out/logs"


class TestShortcutOverride:
    def test_c_and_r_override_are_accepted(self, monkeypatch, tmp_path):
        # --science supplies the kind; -r/-c override its defaults rather than erroring.
        cfg = tmp_path / "custom.toml"
        cfg.write_text(
            "[DATA_DIRS]\n"
            'KPF_DATA_INPUT = "/custom/in"\n'
            'KPF_MASTERS_OUTPUT = "/m"\n'
            'KPF_SCIENCE_OUTPUT = "/s"\n'
            "[LOGGER]\n"
            'log_dir = "/l"\n'
        )
        sentinel = tmp_path / "seen.txt"
        recipe = _stub_recipe(tmp_path, sentinel)
        _run(
            monkeypatch,
            ["--science", "-r", str(recipe), "-c", str(cfg), "-o", "KP.x"],
        )
        assert sentinel.read_text() == "/custom/in"


class TestGuards:
    # main() raises SystemExit with two different meanings -- argparse's 2 for a
    # usage error, and a string payload (exit 1) for a bad recipe -- so a bare
    # `raises(SystemExit)` cannot tell a rejected flag from a crash. Every case
    # here pins the code and the specific message.

    def test_masters_rejects_obs_id(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--masters", "-o", "KP.x"])
        assert exc.value.code == 2
        assert "masters recipe takes -d/--datecode" in capsys.readouterr().err

    def test_science_rejects_datecode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--science", "-d", "20240405"])
        assert exc.value.code == 2
        assert "science recipe takes -o/--obs_id" in capsys.readouterr().err

    def test_missing_recipe_and_config_errors(self, capsys):
        # No --masters/--science and no explicit -r/-c pair.
        with pytest.raises(SystemExit) as exc:
            red.main(["-o", "KP.x"])
        assert exc.value.code == 2
        assert "must specify --masters, --science" in capsys.readouterr().err

    def test_masters_and_science_are_mutually_exclusive(self, capsys):
        with pytest.raises(SystemExit) as exc:
            red.main(["--masters", "--science", "-d", "20240405"])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err

    def test_datecode_and_obs_id_are_mutually_exclusive(self, capsys):
        # The guard against running a masters recipe on a science target.
        with pytest.raises(SystemExit) as exc:
            red.main(["--science", "-d", "20240405", "-o", "KP.x"])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


class TestRecipeLoading:
    """reduce.py's recipe-loading failure branches. All in-process via main(),
    no subprocess."""

    def test_missing_recipe_file_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        monkeypatch.setattr(red, "setup_logging", lambda **kw: "/dev/null")
        with pytest.raises(SystemExit, match="Recipe file not found"):
            red.main(["-r", str(tmp_path / "absent.py"), "-c", str(cfg), "-o", "KP.x"])

    def test_recipe_without_main_exits(self, monkeypatch, tmp_path):
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "nomain.py"
        recipe.write_text("VALUE = 1\n")
        monkeypatch.setattr(red, "setup_logging", lambda **kw: "/dev/null")
        with pytest.raises(SystemExit, match="has no main"):
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", "KP.x"])

    def test_recipe_exception_is_logged_before_reraise(
        self, monkeypatch, tmp_path, caplog
    ):
        # reduce.py's one sanctioned catch-log-reraise: the only place a
        # traceback is guaranteed to reach the log before the nonzero exit
        # (DRP-RUN-08). A bare `raise` here would lose the logged traceback.
        cfg = _base_cfg(tmp_path)
        recipe = tmp_path / "boom.py"
        recipe.write_text("def main(config, args):\n    raise RuntimeError('boom')\n")
        monkeypatch.setattr(red, "setup_logging", lambda **kw: "/dev/null")
        caplog.set_level(logging.CRITICAL)

        with pytest.raises(RuntimeError, match="boom"):
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", "KP.x"])

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

        monkeypatch.setattr(red, "setup_logging", lambda **kw: "/dev/null")
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
            red.main(["-r", str(recipe), "-c", str(cfg), "-o", "KP.x"])
        assert exc.value.code == 2


class _Config:
    """Minimal ConfigHandler stub yielding the DATA_DIRS clear_stale_outputs reads."""

    def __init__(self, data_dirs):
        self._data_dirs = data_dirs

    def get_params(self, keys):
        assert keys == ["DATA_DIRS"]
        return self._data_dirs


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
            _Config({"KPF_SCIENCE_OUTPUT": science_root}), _args(obs_id=self._OID)
        )

        assert not any(os.path.exists(p) for p in targets)
        assert os.path.exists(stray)  # a different obs_id's product is untouched

    def test_science_raises_when_output_root_unset(self):
        # No KPF_SCIENCE_OUTPUT -> can't know what to clear; fail loud.
        with pytest.raises(KeyError, match="KPF_SCIENCE_OUTPUT"):
            red.clear_stale_outputs(_Config({}), _args(obs_id=self._OID))

    def test_masters_raises_when_output_root_unset(self):
        # Same for the masters branch.
        with pytest.raises(KeyError, match="KPF_MASTERS_OUTPUT"):
            red.clear_stale_outputs(_Config({}), _args(datecode="20240405"))

    def test_science_noop_for_invalid_obs_id(self, tmp_path):
        # A malformed obs_id can't build a path; skip rather than raise -- the
        # recipe reports the real error.
        red.clear_stale_outputs(
            _Config({"KPF_SCIENCE_OUTPUT": str(tmp_path)}), _args(obs_id="KP.x")
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
            f"{koaid}_master_thar_diagnostics.h5",
            "KP.20240405.40113.57_thar_L2.fits",
            "KP.20240405.40200.00_thar_L2.fits",
        ]
        for name in stack_files:
            open(os.path.join(stack_subdir, name), "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": masters_root}), _args(datecode="20240405")
        )

        for name in removed:
            assert not os.path.exists(os.path.join(night, name)), name
        assert not os.path.exists(stack_subdir)
        for name in kept:
            assert os.path.exists(os.path.join(night, name)), name

    def test_masters_noop_when_dir_absent(self, tmp_path):
        # A never-built night has no masters dir; glob matches nothing, no error.
        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": str(tmp_path)}), _args(datecode="20240405")
        )
