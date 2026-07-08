"""Tests for tools/cli.py: shortcut resolution and the recipe/config override.

The shortcut path resolution (``shortcut_paths``) is the single source of the
``--masters``/``--science`` -> recipe/config mapping, reused by the processing
drivers under ``scripts/`` so they no longer hardcode those paths. The shortcut
supplies defaults an explicit ``-r``/``-c`` overrides (it used to error on any
combination), which is what lets a driver forward a custom config as
``--science -c custom.toml``.
"""

import os
import sys

import pytest

from tools import cli


class TestShortcutPaths:
    def test_masters(self):
        recipe, config = cli.shortcut_paths("masters")
        assert os.path.isabs(recipe) and os.path.isabs(config)
        assert recipe.endswith(os.path.join("recipes", "kpf_drp_masters.py"))
        assert config.endswith(os.path.join("configs", "kpf_drp_masters.toml"))

    def test_science(self):
        recipe, config = cli.shortcut_paths("science")
        assert os.path.isabs(recipe) and os.path.isabs(config)
        assert recipe.endswith(os.path.join("recipes", "kpf_drp_science.py"))
        assert config.endswith(os.path.join("configs", "kpf_drp_science.toml"))

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError):
            cli.shortcut_paths("nope")


class TestShortcutOverride:
    def _run(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["kpfpipe", *argv])
        # Keep the real logging stack untouched; we only assert config resolution.
        monkeypatch.setattr(cli, "setup_logging", lambda **kw: "/dev/null")
        cli.main()

    def test_science_c_and_r_override_are_accepted(self, monkeypatch, tmp_path):
        # A temp config with a distinctive data dir and a temp recipe that records
        # the config it was handed. --science supplies the kind; -r/-c override its
        # defaults. This combination used to raise SystemExit; now it is allowed.
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
        recipe = tmp_path / "rec.py"
        recipe.write_text(
            "def main(config, args):\n"
            f"    with open({str(sentinel)!r}, 'w') as fh:\n"
            "        fh.write(config.get_params(['DATA_DIRS'])['KPF_DATA_INPUT'])\n"
        )
        self._run(
            monkeypatch,
            ["--science", "-r", str(recipe), "-c", str(cfg), "-o", "KP.x"],
        )
        assert sentinel.read_text() == "/custom/in"

    def test_masters_still_rejects_obs_id(self, monkeypatch):
        # The recipe-kind guard is preserved: masters takes -d, not -o.
        monkeypatch.setattr(sys, "argv", ["kpfpipe", "--masters", "-o", "KP.x"])
        with pytest.raises(SystemExit):
            cli.main()

    def test_science_still_rejects_datecode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["kpfpipe", "--science", "-d", "20240405"])
        with pytest.raises(SystemExit):
            cli.main()
