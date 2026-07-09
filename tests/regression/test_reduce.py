"""Tests for scripts/processing/reduce.py: the ``kpfpipe run`` leaf.

reduce.py is the single-recipe, single-unit runner relocated out of the old
tools/cli.py. These cover the shortcut/`-r`/`-c` resolution and the recipe-kind
guards, and that ``--masters``/``--science`` set a default config an explicit
``-c`` overrides. Each test drives ``main(argv)`` against a tiny stub recipe
(which records the resolved config) with logging stubbed out, so no real
reduction runs.

(``resolve_logging`` itself is unit-tested in test_logger.py.)
"""

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
