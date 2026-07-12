"""Tests for scripts/processing/reduce.py: the ``kpfpipe run`` leaf.

reduce.py is the single-recipe, single-unit runner relocated out of the old
tools/cli.py. These cover the shortcut/`-r`/`-c` resolution and the recipe-kind
guards, and that ``--masters``/``--science`` set a default config an explicit
``-c`` overrides. Each test drives ``main(argv)`` against a tiny stub recipe
(which records the resolved config) with logging stubbed out, so no real
reduction runs.

(``resolve_logging`` itself is unit-tested in test_logger.py.)
"""

import argparse
import os

import pytest

from scripts.processing import reduce as red


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
    # Keep the real logging stack untouched; we only assert config resolution.
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
        # --output_dir fills masters/science output + log dir; input keeps the config.
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
        # masters/science outputs take the root; the log dir gets its subdir.
        assert masters == "/out" and science == "/out" and log_dir == "/out/logs"


class TestShortcutOverride:
    def test_c_and_r_override_are_accepted(self, monkeypatch, tmp_path):
        # A temp config with a distinctive data dir; --science supplies the kind,
        # -r/-c override its defaults. This combination is allowed (not an error).
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
    def test_masters_rejects_obs_id(self, monkeypatch):
        with pytest.raises(SystemExit):
            red.main(["--masters", "-o", "KP.x"])

    def test_science_rejects_datecode(self, monkeypatch):
        with pytest.raises(SystemExit):
            red.main(["--science", "-d", "20240405"])

    def test_missing_recipe_and_config_errors(self, monkeypatch):
        # No --masters/--science and no explicit -r/-c pair.
        with pytest.raises(SystemExit):
            red.main(["-o", "KP.x"])


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
        # Create the three deterministic per-obs_id products (plus a stray file
        # in the same tree that must survive).
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

    def test_science_noop_when_output_root_unset(self, tmp_path):
        # No KPF_SCIENCE_OUTPUT -> nothing to resolve, no error.
        red.clear_stale_outputs(_Config({}), _args(obs_id=self._OID))

    def test_science_noop_for_invalid_obs_id(self, tmp_path):
        # A malformed obs_id can't build a path; skip rather than raise (the recipe
        # reports the real error). This mirrors the -o KP.x guard-test case.
        red.clear_stale_outputs(
            _Config({"KPF_SCIENCE_OUTPUT": str(tmp_path)}), _args(obs_id="KP.x")
        )

    def test_masters_removes_night_products_and_sidecar(self, tmp_path):
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

        # The WLS sidecar dir: per-frame ThAr L2s + the diagnostics HDF5, all
        # removed wholesale with the thar master.
        sidecar = os.path.join(night, "thar_L2")
        os.makedirs(sidecar)
        sidecar_files = [
            f"{koaid}_master_thar_diagnostics.h5",
            "KP.20240405.40113.57_thar_L2.fits",
            "KP.20240405.40200.00_thar_L2.fits",
        ]
        for name in sidecar_files:
            open(os.path.join(sidecar, name), "w").close()

        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": masters_root}), _args(datecode="20240405")
        )

        for name in removed:
            assert not os.path.exists(os.path.join(night, name)), name
        assert not os.path.exists(sidecar)  # entire sidecar dir gone
        for name in kept:
            assert os.path.exists(os.path.join(night, name)), name

    def test_masters_noop_when_dir_absent(self, tmp_path):
        # A never-built night has no masters dir; glob matches nothing, no error.
        red.clear_stale_outputs(
            _Config({"KPF_MASTERS_OUTPUT": str(tmp_path)}), _args(datecode="20240405")
        )
