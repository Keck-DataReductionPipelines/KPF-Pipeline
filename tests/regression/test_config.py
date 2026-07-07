"""Tests for kpfpipe.utils.config: ConfigHandler TOML loading and section overrides."""

from kpfpipe.utils.config import ConfigHandler


def _write_toml(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestConfigHandler:
    def test_loads_sections(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n[TRACES]\nn = 3\n')
        handler = ConfigHandler(cfg)
        assert handler.config["DATA_DIRS"]["root"] == "/data"

    def test_get_params_default_sections(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n[TRACES]\nn = 3\n')
        params = ConfigHandler(cfg).get_params()  # sections=None -> defaults
        assert params == {"root": "/data", "n": 3}

    def test_get_params_flattens_nested_dict(self, tmp_path):
        toml = "[TRACES]\nsimple = 1\n[TRACES.nested]\nsub = 2\n"
        cfg = _write_toml(tmp_path, toml)
        params = ConfigHandler(cfg).get_params(["TRACES"])
        assert params["simple"] == 1
        assert params["nested_sub"] == 2

    def test_override_merges_into_dict_section(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n')
        handler = ConfigHandler(cfg, overrides={"DATA_DIRS": {"out": "/out"}})
        assert handler.config["DATA_DIRS"] == {"root": "/data", "out": "/out"}

    def test_override_replaces_when_section_absent_or_non_dict(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n')
        handler = ConfigHandler(cfg, overrides={"NEW_SECTION": 42})
        assert handler.config["NEW_SECTION"] == 42

    def test_load_config_with_explicit_path_switches_file(self, tmp_path):
        cfg1 = _write_toml(tmp_path, "[TRACES]\nn = 1\n", name="a.toml")
        cfg2 = _write_toml(tmp_path, "[TRACES]\nn = 2\n", name="b.toml")
        handler = ConfigHandler(cfg1)
        handler.load_config(cfg2)
        assert str(handler.path).endswith("b.toml")
        assert handler.config["TRACES"]["n"] == 2

    def test_get_params_reloads_when_config_empty(self, tmp_path):
        cfg = _write_toml(tmp_path, "[TRACES]\nn = 7\n")
        handler = ConfigHandler(cfg)
        handler.config = {}  # force the reload branch in get_params
        params = handler.get_params(["TRACES"])
        assert params["n"] == 7
