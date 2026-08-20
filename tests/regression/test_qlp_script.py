"""Tests for scripts/quality_control/qlp.py, the quicklook-plot generator.

One subprocess test proves the script is invocable as a program; the rest drive
``main()`` in-process, which costs milliseconds instead of an interpreter start
and lets each case assert the message as well as the exit code.
"""

import os
import sys

import pytest

from kpfpipe.utils.io import kpf_directory
from scripts.quality_control import qlp

from ._data_models import write_amp_l0
from ._scripts import run_script, write_config

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_SCRIPT = "scripts/quality_control/qlp.py"

# A valid obs_id (and matching datecode) the quicklook filenames/paths key off.
_OBS_ID = "KP.20240405.00004.00"


def _write_l0_image_fixture(path):
    """Write an L0 FITS fixture with the two amps per chip PlotL0 needs to stitch."""
    write_amp_l0(
        path,
        namps=2,
        shape=(32, 24),
        bias_level=1000.0,
        seed=123,
        primary_cards={
            "OBJECT": "small-2amp",
            "DATE-OBS": "2024-04-05T01:00:39",
            "PROGNAME": None,
        },
    )


def _run_qlp_script(*args):
    return run_script(_SCRIPT, *args)


def _main(monkeypatch, *args):
    """Drive qlp.main() in-process with ``args`` as argv."""
    monkeypatch.setattr(sys, "argv", [_SCRIPT, *map(str, args)])
    qlp.main()


def _png_names(obs_id):
    """The two stitched-image PNG basenames PlotL0 writes for a frame."""
    return [
        f"{obs_id}_L0_stitched_image_{chip}_zoomable.png" for chip in ("green", "red")
    ]


class TestQLPScript:
    def test_generates_plots_exit_0(self, tmp_path):
        fixture = tmp_path / f"{_OBS_ID}.fits"
        _write_l0_image_fixture(str(fixture))
        out_dir = tmp_path / "out"

        result = _run_qlp_script(
            "--input", fixture, "--level", "L0", "--output_dir", out_dir
        )

        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for name in _png_names(_OBS_ID):
            assert (out_dir / name).is_file(), f"missing plot {name} in {out_dir}"

    def test_config_output_resolution(self, tmp_path, monkeypatch):
        # --input supplies the frame, so only the output tree comes from the config.
        fixture = tmp_path / f"{_OBS_ID}.fits"
        _write_l0_image_fixture(str(fixture))
        science_out = tmp_path / "science-root"
        cfg = tmp_path / "cfg.toml"
        write_config(
            cfg,
            {"KPF_DATA_INPUT": str(tmp_path), "KPF_SCIENCE_OUTPUT": str(science_out)},
        )

        _main(monkeypatch, "--input", fixture, "--level", "L0", "--config", cfg)

        expected_dir = kpf_directory(
            kind="QLP", data_root=str(science_out), level="L0", obs_id=_OBS_ID
        )
        for name in _png_names(_OBS_ID):
            assert os.path.isfile(os.path.join(expected_dir, name)), (
                f"missing plot {name} in {expected_dir}"
            )

    def test_missing_science_output_key_fails_loud(self, tmp_path, monkeypatch):
        # The output path reads params["KPF_SCIENCE_OUTPUT"] with no default, so an
        # absent key must error rather than silently fall back to some other root.
        fixture = tmp_path / f"{_OBS_ID}.fits"
        _write_l0_image_fixture(str(fixture))
        cfg = tmp_path / "cfg.toml"
        write_config(cfg, {"KPF_DATA_INPUT": str(tmp_path)})  # no KPF_SCIENCE_OUTPUT

        with pytest.raises(KeyError, match="KPF_SCIENCE_OUTPUT"):
            _main(monkeypatch, "--input", fixture, "--level", "L0", "--config", cfg)

    def test_missing_file_exit_1(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "does_not_exist.fits"
        with pytest.raises(SystemExit) as exc:
            _main(
                monkeypatch,
                "--input",
                missing,
                "--level",
                "L0",
                "--output_dir",
                tmp_path,
            )
        assert exc.value.code == 1
        assert "file not found" in capsys.readouterr().err

    def test_no_args_exit_nonzero(self, monkeypatch, capsys):
        # --level is required, so bare argv is an argparse error.
        with pytest.raises(SystemExit) as exc:
            _main(monkeypatch)
        assert exc.value.code == 2
        assert "--level" in capsys.readouterr().err
