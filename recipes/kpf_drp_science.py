"""
KPF science reduction recipe.

Runs the full single-exposure science pipeline end-to-end for one obs_id,
L0 -> L1 -> L2 -> L4: read the raw L0 frame, assemble it into a full-frame
image, associate and apply calibration masters (bias, dark, flat, ThAr WLS),
extract 1D spectra, attach the wavelength solution, apply the barycentric
correction, and compute radial velocities from the cross-correlation function.
Diagnostics, QC, and quicklook layers run at each level, and the L2 and L4
data products are written to the output data root.
"""

import logging

from kpfpipe.data_models import KPF0
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
from kpfpipe.utils.io import build_filepath, build_qlp_dir

# Explicit name: the CLI execs recipes with __name__ == "recipe", so __name__
# would not identify this module in the log (style guide section 6).
logger = logging.getLogger("kpfpipe.recipe.science")


def main(config, args):
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

    l0 = KPF0.from_fits(build_filepath(obs_id, "L0", data_root=data_root_in))

    # Generate L0 quicklook plots
    logger.info("generating L0 quicklook plots for %s", obs_id)
    l0_qlp_dir = build_qlp_dir(obs_id, "L0", data_root=data_root_science)
    PlotL0(l0, output_dir=l0_qlp_dir).run("all")

    # L0 processing complete: CheckpointL0.run() folds in Diagnostics + QC, then
    # validates. Run before assembly on purpose -- QCL0 writes the L0 QC flags +
    # ISGOOD onto l0's QUALITY_CONTROL, which to_kpf1 propagates downstream so the
    # L1/L2/L4 products carry the full append-only QC history.
    logger.info("running L0 checkpoint for %s", obs_id)
    CheckpointL0(l0).run()

    # Assemble the raw L0 readout into a single L1 full-frame image (FFI)
    logger.info("assembling L0 -> L1 FFI for %s", obs_id)
    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    # Associate the implemented calibration masters closest to this frame so
    # image processing and wavelength calibration can use them. Flat frames are
    # still part of the desired data set, but master-flat construction and flat
    # division are not implemented yet, so the basic path does not require them.
    logger.info("associating calibration masters for %s", obs_id)
    calibration_association = CalibrationAssociation(l1, config)
    l1 = calibration_association.perform(["bias", "dark", "thar"])

    # Apply standard FFI image processing. The current runnable path performs
    # bias and dark subtraction; flat correction remains disabled in config.
    logger.info("applying image processing for %s", obs_id)
    image_processing = ImageProcessing(l1, config)
    l1 = image_processing.perform()

    # L1 quicklook plots
    logger.info("generating L1 quicklook plots for %s", obs_id)
    l1_qlp_dir = build_qlp_dir(obs_id, "L1", data_root=data_root_science)
    PlotL1(l1, output_dir=l1_qlp_dir).run("all")

    # L1 processing complete: CheckpointL1.run() folds in Diagnostics + QC,
    # then validates (science -> Diagnostics -> QC -> Checkpoints).
    logger.info("running L1 checkpoint for %s", obs_id)
    CheckpointL1(l1).run()

    # Extract the 2D FFI down to 1D spectra (2D --> 1D), since the RV analysis
    # operates on per-order flux rather than the raw image.
    logger.info("extracting 1D spectra for %s", obs_id)
    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    # Attach the precomputed wavelength solution (per-fiber WAVE arrays from the
    # WLS master) so each order has a calibrated wavelength axis [Å, vacuum].
    logger.info("attaching wavelength solution for %s", obs_id)
    wavelength_calibration = WavelengthCalibration(l2, config)
    l2 = wavelength_calibration.perform()

    # Apply the per-order barycentric correction so the spectra are placed in
    # the Solar System barycentric frame for long-term RV stability. Writes
    # BJD_TDB, BARYCORR_KMS [km/s], and BARYCORR_Z to the headers.
    logger.info("applying barycentric correction for %s", obs_id)
    barycentric_correction = BarycentricCorrection(l2, config)
    l2 = barycentric_correction.perform()

    # L2 quicklook plots
    logger.info("generating L2 quicklook plots for %s", obs_id)
    l2_qlp_dir = build_qlp_dir(obs_id, "L2", data_root=data_root_science)
    PlotL2(l2, output_dir=l2_qlp_dir, obs_id=obs_id).run("all")

    # L2 processing complete: CheckpointL2.run() folds in Diagnostics + QC,
    # then validates (science -> Diagnostics -> QC -> Checkpoints).
    logger.info("running L2 checkpoint for %s", obs_id)
    CheckpointL2(l2).run()

    # Write the L2 data products (extracted 1D spectra) to disk
    l2_out_path = build_filepath(obs_id, "L2", data_root=data_root_science)
    l2.to_fits(l2_out_path)

    # Cross-correlate the extracted spectra to build the per-order CCFs (L4).
    logger.info("cross-correlating spectra for %s", obs_id)
    cross_correlation = CrossCorrelation(l2, config)
    l4 = cross_correlation.perform()

    # Fit the CCFs to radial velocities -- the primary scientific product.
    logger.info("computing radial velocities for %s", obs_id)
    radial_velocity = RadialVelocity(l4, config)
    l4 = radial_velocity.perform()

    # L4 quicklook plots
    logger.info("generating L4 quicklook plots for %s", obs_id)
    l4_qlp_dir = build_qlp_dir(obs_id, "L4", data_root=data_root_science)
    PlotL4(l4, output_dir=l4_qlp_dir, obs_id=obs_id).run("all")

    # L4 processing complete: CheckpointL4.run() folds in Diagnostics (DiagL4)
    # and QC (QCL4), then validates (science -> Diagnostics -> QC -> Checkpoints).
    logger.info("running L4 checkpoint for %s", obs_id)
    CheckpointL4(l4).run()

    # Write the final L4 data product (RVs and CCFs) to disk
    l4_out_path = build_filepath(obs_id, "L4", data_root=data_root_science)
    l4.to_fits(l4_out_path)

    logger.info("exiting kpf_drp_science pipeline")
