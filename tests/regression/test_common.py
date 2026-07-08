"""Tests for scripts/processing/_common.py: the tools-free shared helpers.

Covers the reduction registry (``shortcut_paths`` -- the single source of the
masters/science -> recipe/config mapping, reused by the leaf runner and the
orchestrators) and the small filesystem helpers (``_datecode_dirs``,
``_dir_params``). Synthetic tmp trees only -- no real testdata needed.
"""

import os

import pytest

from scripts.processing import _common


class TestShortcutPaths:
    def test_masters(self):
        recipe, config = _common.shortcut_paths("masters")
        assert os.path.isabs(recipe) and os.path.isabs(config)
        assert recipe.endswith(os.path.join("recipes", "kpf_drp_masters.py"))
        assert config.endswith(os.path.join("configs", "kpf_drp_masters.toml"))

    def test_science(self):
        recipe, config = _common.shortcut_paths("science")
        assert os.path.isabs(recipe) and os.path.isabs(config)
        assert recipe.endswith(os.path.join("recipes", "kpf_drp_science.py"))
        assert config.endswith(os.path.join("configs", "kpf_drp_science.toml"))

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError):
            _common.shortcut_paths("nope")


class TestDatecodeDirs:
    def test_filters_by_range_and_sorts(self, tmp_path):
        for name in ["20240101", "20240115", "20240201", "notadate", "20231231"]:
            (tmp_path / name).mkdir()
        (tmp_path / "20240110_file").write_text("x")  # datecode-ish, but not a dir
        got = _common._datecode_dirs(str(tmp_path), "20240101", "20240131")
        assert got == ["20240101", "20240115"]
