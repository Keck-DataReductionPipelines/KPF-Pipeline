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

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.modules.calibration_association import CalibrationAssociation

# from kpfpipe.data_models.level1 import KPF1
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.radial_velocity import RadialVelocity
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
from kpfpipe.quality_control.checkpoints import CheckpointL1, CheckpointL2
from kpfpipe.quality_control.diagnostics import DiagL1, DiagL2
from kpfpipe.quality_control.qc_booleans import QCL1, QCL2
from kpfpipe.quality_control.quicklook.level0 import PlotL0
from kpfpipe.quality_control.quicklook.level1 import PlotL1
from kpfpipe.quality_control.quicklook.level2 import PlotL2
from kpfpipe.utils.io import build_filepath, build_qlp_dir


def main(config, args):
    print("\n\n=== entering kpf_drp_science pipeline ===\n\n")

    if not args.obs_id:
        raise SystemExit(
            "Error: --obs_id is required for the science recipe "
            "(e.g. -o KP.20240405.40113.57)"
        )

    obs_id = args.obs_id

    data_dirs = config.get_params(["DATA_DIRS"])
    data_root_in = data_dirs["KPF_DATA_INPUT"]
    data_root_science = data_dirs["KPF_SCIENCE_OUTPUT"]

    l0 = KPF0.from_fits(build_filepath(obs_id, "L0", data_root=data_root_in))

    l0_qlp_dir = build_qlp_dir(obs_id, "L0", data_root=data_root_science)
    PlotL0(l0, output_dir=l0_qlp_dir).run("all")

    # Assemble the raw L0 readout into a single L1 full-frame image (FFI) so
    # that downstream stages operate on a contiguous detector frame.
    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    # L1 QLP is computed on the assembled (pre-bias-subtraction) image because
    # ImageProcessing mutates GREEN_CCD/RED_CCD in place during bias subtraction.
    l1_qlp_dir = build_qlp_dir(obs_id, "L1", data_root=data_root_science)
    PlotL1(l1, output_dir=l1_qlp_dir).run("all")

    # Associate the implemented calibration masters closest to this frame so
    # image processing and wavelength calibration can use them. Flat frames are
    # still part of the desired data set, but master-flat construction and flat
    # division are not implemented yet, so the basic path does not require them.
    calibration_association = CalibrationAssociation(l1, config)
    l1 = calibration_association.perform(["bias", "dark", "thar"])

    # Apply standard FFI image processing. The current runnable path performs
    # bias and dark subtraction; flat correction remains disabled in config.
    image_processing = ImageProcessing(l1, config)
    l1 = image_processing.perform()

    # Run L1 diagnostics, QC, and checkpoints here so the pass/fail thresholds
    # see the processed image before extraction collapses it to 1D. Order is
    # Diagnostics (metrics) -> QC (0/1 flags) -> Checkpoints (warn/raise).
    DiagL1(l1).run()
    QCL1(l1).run()
    CheckpointL1(l1).run()

    # Extract the 2D FFI down to 1D spectra (2D --> 1D), since the RV analysis
    # operates on per-order flux rather than the raw image.
    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    # Attach the precomputed wavelength solution (per-fiber WAVE arrays from the
    # WLS master) so each order has a calibrated wavelength axis [Å, vacuum].
    wavelength_calibration = WavelengthCalibration(l2, config)
    l2 = wavelength_calibration.perform()

    # Run L2 diagnostics, QC, and checkpoints on the extracted spectra to flag
    # NaN counts and zero-flux orders before they propagate into the RV
    # measurement (Diagnostics -> QC -> Checkpoints).
    DiagL2(l2).run()
    QCL2(l2).run()
    CheckpointL2(l2).run()

    # Apply the per-order barycentric correction so the spectra are placed in
    # the Solar System barycentric frame for long-term RV stability. Writes
    # BJD_TDB, BARYCORR_KMS [km/s], and BARYCORR_Z to the headers.
    barycentric_correction = BarycentricCorrection(l2, config)
    l2 = barycentric_correction.perform()

    # L2 QLP (wavelength-aware extracted-spectrum plots; requires attached WAVE)
    l2_qlp_dir = build_qlp_dir(obs_id, "L2", data_root=data_root_science)
    PlotL2(l2, output_dir=l2_qlp_dir, obs_id=obs_id).run("all")

    # Write the L2 data product to disk before the RV step so the calibrated
    # spectra are preserved even if RV computation fails. to_fits creates the
    # parent directory as needed.
    l2_out_path = build_filepath(obs_id, "L2", data_root=data_root_science)
    l2.to_fits(l2_out_path)

    # Compute the radial velocity (RV) from the cross-correlation function
    # (CCF), which is the primary scientific product of the pipeline.
    radial_velocity = RadialVelocity(l2, config)
    l4 = radial_velocity.perform()

    # Write the final L4 data product (RVs and CCFs) to disk; to_fits creates
    # the parent directory as needed.
    l4_out_path = build_filepath(obs_id, "L4", data_root=data_root_science)
    l4.to_fits(l4_out_path)

    print("\n\n=== exiting kpf_drp_science pipeline ===\n\n")
