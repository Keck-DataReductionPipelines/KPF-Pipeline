"""
KPF engineering data reduction recipe.

Runs the single-exposure pipeline for one obs_id, L0 -> L1 -> L2, using a
simplified data flow: assemble the FFI, extract the spectra, and append a
wavelength solution. Diagnostics/QC/quicklook run at each level; L2 is
written to disk.

Unlike the science recipe, this one completes the run: a failed QC
checkpoint or a missing ThAr master is logged as a warning and processing
continues. No external catalog or database queries are made.
"""

import logging
import time

from kpfpipe.data_models import KPF0
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
from kpfpipe.quality_control.checkpoints import (
    CheckpointL0,
    CheckpointL1,
    CheckpointL2,
)
from kpfpipe.quality_control.quicklook import PlotL0, PlotL1, PlotL2
from kpfpipe.utils.io import kpf_directory, kpf_filepath
from recipes._logging import engineering_run_summary

# Explicit name: the CLI execs recipes with __name__ == "recipe", so __name__
# would not identify this module in the log.
logger = logging.getLogger("kpfpipe.recipe.engineering")

# What this recipe produces; the scripts layer gates selector checks and clearing on it.
RECIPE_TYPE = "engineering"

# Recipe stages in run order. Config [STAGES] switches each on or off, and --stages
# overrides it. A stage switched off within a level passes the product through; one
# that creates the next level (image_assembly, spectral_extraction) ends the run
# there, since every later stage needs its product.
RECIPE_STAGES = (
    "image_assembly",
    "calibration_association",
    "spectral_extraction",
    "wavelength_calibration",
)


def _run_checkpoint(checkpoint, level):
    # Checkpoint.run writes diagnostics and QC flags before it raises, so the
    # product keeps its QC history; the run continues regardless.
    try:
        checkpoint.run()
    except ValueError as e:
        logger.warning("%s checkpoint failed; continuing: %s", level, e)


def main(config, args):
    t0 = time.monotonic()
    logger.info("entering kpf_drp_engineering pipeline")

    if not args.obs_id:
        raise SystemExit(
            "Error: --obs_id is required for the engineering recipe "
            "(e.g. -o KP.20240405.40113.57)"
        )

    obs_id = args.obs_id
    stages = [s for s in RECIPE_STAGES if config.get_params(["STAGES"])[s]]
    logger.info("reducing %s (stages: %s)", obs_id, ", ".join(stages))

    data_dirs = config.get_params(["DATA_DIRS"])
    data_root_in = data_dirs["KPF_DATA_INPUT"]
    data_root_out = data_dirs["KPF_ENGINEERING_OUTPUT"]

    fn = kpf_filepath(obs_id, "L0", data_root=data_root_in)
    l0 = KPF0.from_fits(fn, standardize=True)

    logger.info("generating L0 quicklook plots for %s", obs_id)
    l0_qlp_dir = kpf_directory(
        kind="QLP",
        data_root=data_root_out,
        level="L0",
        obs_id=obs_id,
        engineering=True,
    )
    PlotL0(l0, output_dir=l0_qlp_dir).run("all")

    # Run before assembly on purpose: QCL0 writes the L0 QC flags onto l0's
    # QUALITY_CONTROL, which to_kpf1 propagates downstream so L1/L2 carry
    # the full append-only QC history.
    logger.info("running L0 checkpoint for %s", obs_id)
    _run_checkpoint(CheckpointL0(l0), "L0")

    if "image_assembly" not in stages:
        logger.info("image_assembly off; stopping after L0")
        return
    logger.info("assembling L0 -> L1 FFI for %s", obs_id)
    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    # Only the ThAr (WLS) master is needed; no bias/dark/flat correction here.
    have_wls = False
    if "calibration_association" in stages:
        logger.info("associating ThAr master for %s", obs_id)
        calibration_association = CalibrationAssociation(l1, config)
        try:
            l1 = calibration_association.perform(["thar"])
            have_wls = True
        except FileNotFoundError as e:
            logger.warning("no ThAr master; skipping wavelength calibration: %s", e)

    logger.info("generating L1 quicklook plots for %s", obs_id)
    l1_qlp_dir = kpf_directory(
        kind="QLP",
        data_root=data_root_out,
        level="L1",
        obs_id=obs_id,
        engineering=True,
    )
    PlotL1(l1, output_dir=l1_qlp_dir).run("all")

    logger.info("running L1 checkpoint for %s", obs_id)
    _run_checkpoint(CheckpointL1(l1), "L1")

    if "spectral_extraction" not in stages:
        logger.info("spectral_extraction off; stopping after L1")
        return

    logger.info("extracting 1D spectra for %s", obs_id)
    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    # Per-fiber WAVE arrays from the WLS master: calibrated axis [Å, vacuum].
    if "wavelength_calibration" in stages and have_wls:
        logger.info("attaching wavelength solution for %s", obs_id)
        wavelength_calibration = WavelengthCalibration(l2, config)
        l2 = wavelength_calibration.perform()

    logger.info("generating L2 quicklook plots for %s", obs_id)
    l2_qlp_dir = kpf_directory(
        kind="QLP",
        data_root=data_root_out,
        level="L2",
        obs_id=obs_id,
        engineering=True,
    )
    PlotL2(l2, output_dir=l2_qlp_dir, obs_id=obs_id).run("all")

    logger.info("running L2 checkpoint for %s", obs_id)
    _run_checkpoint(CheckpointL2(l2), "L2")

    # Write L2 (extracted 1D spectra) to disk.
    l2_out_path = kpf_filepath(obs_id, "L2", data_root=data_root_out, engineering=True)
    # Deliberately non-EPRV kpf_EngL2 name: skip to_fits' filename-convention warning.
    l2.check_filename_convention = lambda _: True
    l2.to_fits(l2_out_path)

    # Compact end-of-run verdict (inputs, ThAr master, outputs, elapsed).
    logger.info(engineering_run_summary(l2, time.monotonic() - t0))

    logger.info("exiting kpf_drp_engineering pipeline")


if __name__ == "__main__":
    raise SystemExit("recipes run only via kpfpipe (see kpfpipe --help), not python")
