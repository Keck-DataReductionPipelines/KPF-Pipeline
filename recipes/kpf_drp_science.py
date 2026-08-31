"""
KPF science reduction recipe.

Runs the single-exposure pipeline end-to-end for one obs_id, L0 -> L1 -> L2 ->
L4: assemble the FFI, apply calibration masters, extract and calibrate the
spectra, apply the barycentric correction, and compute RVs from the CCF.
Diagnostics/QC/quicklook run at each level; L2 and L4 are written to disk.
"""

import logging
import time

from kpfpipe.data_models import KPF0
from kpfpipe.modules.astro_query import AstroQuery
from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.cross_correlation import CrossCorrelation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.radial_velocity import RadialVelocity
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
from kpfpipe.quality_control.checkpoints import (
    CheckpointL0,
    CheckpointL1,
    CheckpointL2,
    CheckpointL4,
)
from kpfpipe.quality_control.quicklook import PlotL0, PlotL1, PlotL2, PlotL4
from kpfpipe.utils.io import kpf_directory, kpf_filepath
from recipes._logging import science_run_summary

# Explicit name: the CLI execs recipes with __name__ == "recipe", so __name__
# would not identify this module in the log.
logger = logging.getLogger("kpfpipe.recipe.science")


def main(config, args):
    t0 = time.monotonic()
    logger.info("entering kpf_drp_science pipeline")

    if not args.obs_id:
        raise SystemExit(
            "Error: --obs_id is required for the science recipe "
            "(e.g. -o KP.20240405.40113.57)"
        )

    obs_id = args.obs_id
    logger.info("reducing %s", obs_id)

    data_dirs = config.get_params(["DATA_DIRS"])
    data_root_in = data_dirs["KPF_DATA_INPUT"]
    data_root_science = data_dirs["KPF_SCIENCE_OUTPUT"]

    fn = kpf_filepath(obs_id, "L0", data_root=data_root_in)
    l0 = KPF0.from_fits(fn, standardize=True)

    # Resolves into l0's CATALOG_RECORD, consumed by L0 diagnostics and
    # barycentric correction.
    logger.info("resolving catalog astrometry for %s", obs_id)
    astro_query = AstroQuery(l0, config)
    l0 = astro_query.perform()

    logger.info("generating L0 quicklook plots for %s", obs_id)
    l0_qlp_dir = kpf_directory(
        kind="QLP", data_root=data_root_science, level="L0", obs_id=obs_id
    )
    PlotL0(l0, output_dir=l0_qlp_dir).run("all")

    # Run before assembly on purpose: QCL0 writes the L0 QC flags onto l0's
    # QUALITY_CONTROL, which to_kpf1 propagates downstream so L1/L2/L4 carry
    # the full append-only QC history.
    logger.info("running L0 checkpoint for %s", obs_id)
    CheckpointL0(l0).run()

    logger.info("assembling L0 -> L1 FFI for %s", obs_id)
    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    # Flat frames are part of the desired set, but master-flat construction and
    # flat division aren't implemented yet, so the basic path doesn't need them.
    logger.info("associating calibration masters for %s", obs_id)
    calibration_association = CalibrationAssociation(l1, config)
    l1 = calibration_association.perform(["bias", "dark", "thar"])

    # Currently performs bias and dark subtraction; flat correction remains
    # disabled in config.
    logger.info("applying image processing for %s", obs_id)
    image_processing = ImageProcessing(l1, config)
    l1 = image_processing.perform()

    logger.info("generating L1 quicklook plots for %s", obs_id)
    l1_qlp_dir = kpf_directory(
        kind="QLP", data_root=data_root_science, level="L1", obs_id=obs_id
    )
    PlotL1(l1, output_dir=l1_qlp_dir).run("all")

    logger.info("running L1 checkpoint for %s", obs_id)
    CheckpointL1(l1).run()

    # RV analysis operates on per-order flux rather than the raw image.
    logger.info("extracting 1D spectra for %s", obs_id)
    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    # Per-fiber WAVE arrays from the WLS master: calibrated axis [Å, vacuum].
    logger.info("attaching wavelength solution for %s", obs_id)
    wavelength_calibration = WavelengthCalibration(l2, config)
    l2 = wavelength_calibration.perform()

    # Apply the per-order barycentric correction so the spectra are placed in
    # the Solar System barycentric frame for long-term RV stability. Writes
    # BJD_TDB, BARYCORR_KMS [km/s], and BARYCORR_Z to the headers.
    logger.info("applying barycentric correction for %s", obs_id)
    barycentric_correction = BarycentricCorrection(l2, config)
    l2 = barycentric_correction.perform()

    logger.info("generating L2 quicklook plots for %s", obs_id)
    l2_qlp_dir = kpf_directory(
        kind="QLP", data_root=data_root_science, level="L2", obs_id=obs_id
    )
    PlotL2(l2, output_dir=l2_qlp_dir, obs_id=obs_id).run("all")

    logger.info("running L2 checkpoint for %s", obs_id)
    CheckpointL2(l2).run()

    # Write L2 (extracted 1D spectra) to disk.
    l2_out_path = kpf_filepath(obs_id, "L2", data_root=data_root_science)
    l2.to_fits(l2_out_path)

    logger.info("cross-correlating spectra for %s", obs_id)
    cross_correlation = CrossCorrelation(l2, config)
    l4 = cross_correlation.perform()

    # Fit the CCFs to radial velocities -- the primary scientific product.
    logger.info("computing radial velocities for %s", obs_id)
    radial_velocity = RadialVelocity(l4, config)
    l4 = radial_velocity.perform()

    logger.info("generating L4 quicklook plots for %s", obs_id)
    l4_qlp_dir = kpf_directory(
        kind="QLP", data_root=data_root_science, level="L4", obs_id=obs_id
    )
    PlotL4(l4, output_dir=l4_qlp_dir, obs_id=obs_id).run("all")

    logger.info("running L4 checkpoint for %s", obs_id)
    CheckpointL4(l4).run()

    # Write L4 (RVs and CCFs) to disk.
    l4_out_path = kpf_filepath(obs_id, "L4", data_root=data_root_science)
    l4.to_fits(l4_out_path)

    # Compact end-of-run verdict (masters, inputs/outputs, combined RV, elapsed).
    logger.info(science_run_summary(l4, time.monotonic() - t0))

    logger.info("exiting kpf_drp_science pipeline")


if __name__ == "__main__":
    main()
