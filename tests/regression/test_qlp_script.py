"""Tests for scripts/quality_control/qlp.py: the standalone quicklook-plot generator.

Subprocess smoke tests that drive the CLI end-to-end on a synthetic, plottable L0
fixture: direct-path plotting (--input/--output_dir), the config-driven output
tree (--config → {KPF_SCIENCE_OUTPUT}/QLP/...), and the usual missing-file /
no-args guards. A fail-loud check confirms a config missing KPF_SCIENCE_OUTPUT
errors rather than substituting a default output root.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.utils.io import kpf_directory

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# A valid obs_id (and matching datecode) the quicklook filenames/paths key off.
_OBS_ID = "KP.20240405.00004.00"


def _write_l0_image_fixture(path):
    """Write a small, plottable two-amp L0 FITS fixture (green + red).

    Two amps per chip at 32x24 is enough for PlotL0 to stitch and render both
    CCDs; the pixel values are arbitrary noise.
    """
    rng = np.random.default_rng(123)
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "small-2amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:39"
    primary.header["OFNAME"] = os.path.basename(path)

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 3):
            data = rng.normal(1000.0, 3.0, (32, 24)).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_config(path, data_dirs):
    """Write a TOML config with a [DATA_DIRS] section from ``data_dirs``."""
    lines = ["[DATA_DIRS]"]
    lines += [f'{key} = "{value}"' for key, value in data_dirs.items()]
    path.write_text("\n".join(lines) + "\n")


def _run_qlp_script(*args):
    """Run scripts/quality_control/qlp.py via subprocess; return the result."""
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
    cmd = [sys.executable, "scripts/quality_control/qlp.py", *map(str, args)]
    return subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)


def _png_names(obs_id):
    """The two stitched-image PNG basenames PlotL0 writes for a frame."""
    return [
        f"{obs_id}_L0_stitched_image_{chip}_zoomable.png" for chip in ("green", "red")
    ]


class TestQLPScript:
    """Smoke tests for scripts/quality_control/qlp.py via subprocess."""

    def test_generates_plots_exit_0(self, tmp_path):
        """--input + --output_dir → exit 0, both CCD PNGs written."""
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

    def test_config_output_resolution(self, tmp_path):
        """--config (no --output_dir) → plots under {KPF_SCIENCE_OUTPUT}/QLP/...

        Exercises the fix that reads KPF_SCIENCE_OUTPUT (the science output root)
        rather than the never-defined KPF_DATA_OUTPUT. --input supplies the frame,
        so only the output tree comes from the config.
        """
        fixture = tmp_path / f"{_OBS_ID}.fits"
        _write_l0_image_fixture(str(fixture))
        science_out = tmp_path / "science-root"
        cfg = tmp_path / "cfg.toml"
        _write_config(
            cfg,
            {"KPF_DATA_INPUT": str(tmp_path), "KPF_SCIENCE_OUTPUT": str(science_out)},
        )

        result = _run_qlp_script("--input", fixture, "--level", "L0", "--config", cfg)

        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        expected_dir = kpf_directory(
            kind="QLP", data_root=str(science_out), level="L0", obs_id=_OBS_ID
        )
        for name in _png_names(_OBS_ID):
            assert os.path.isfile(os.path.join(expected_dir, name)), (
                f"missing plot {name} in {expected_dir}"
            )

    def test_missing_science_output_key_fails_loud(self, tmp_path):
        """--config without KPF_SCIENCE_OUTPUT → nonzero exit naming the key.

        The output path reads params["KPF_SCIENCE_OUTPUT"] directly (no default),
        so an absent key surfaces as an error rather than a silent fallback root.
        """
        fixture = tmp_path / f"{_OBS_ID}.fits"
        _write_l0_image_fixture(str(fixture))
        cfg = tmp_path / "cfg.toml"
        _write_config(cfg, {"KPF_DATA_INPUT": str(tmp_path)})  # no KPF_SCIENCE_OUTPUT

        result = _run_qlp_script("--input", fixture, "--level", "L0", "--config", cfg)

        assert result.returncode != 0
        assert "KPF_SCIENCE_OUTPUT" in result.stderr, (
            f"Expected 'KPF_SCIENCE_OUTPUT' in stderr:\n{result.stderr}"
        )

    def test_missing_file_exit_1(self, tmp_path):
        """Non-existent input file → exit code 1."""
        missing = tmp_path / "does_not_exist.fits"
        result = _run_qlp_script(
            "--input", missing, "--level", "L0", "--output_dir", tmp_path
        )
        assert result.returncode == 1

    def test_no_args_exit_nonzero(self):
        """No args → argparse error (--level required) → non-zero exit."""
        result = _run_qlp_script()
        assert result.returncode != 0
