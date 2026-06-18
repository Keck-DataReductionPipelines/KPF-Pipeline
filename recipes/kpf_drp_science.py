import os

from kpfpipe.data_models.level0 import KPF0
#from kpfpipe.data_models.level1 import KPF1

from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.modules.radial_velocity import RadialVelocity

from kpfpipe.quality_control.diagnostics import DiagL1, DiagL2
from kpfpipe.quality_control.qc_booleans import QCL1, QCL2
from kpfpipe.quality_control.quicklook.level0 import PlotL0
from kpfpipe.quality_control.quicklook.level1 import PlotL1
from kpfpipe.quality_control.quicklook.level2 import PlotL2

from kpfpipe.utils.pipeline import build_filepath, build_qlp_dir


def main(config, args):
    print("\n\n=== entering kpf_drp_science pipeline ===\n\n")

    if not args.obs_id:
        raise SystemExit("Error: --obs_id is required for the science recipe (e.g. -o KP.20240405.40113.57)")

    obs_id = args.obs_id

    data_dirs = config.get_params(['DATA_DIRS'])
    data_root_in      = data_dirs['KPF_DATA_INPUT']
    data_root_science = data_dirs['KPF_SCIENCE_OUTPUT']

    l0 = KPF0.from_fits(build_filepath(obs_id, 'L0', data_root=data_root_in))

    l0_qlp_dir = build_qlp_dir(obs_id, 'L0', data_root=data_root_science)
    PlotL0(l0, output_dir=l0_qlp_dir).run('all')

    # read raw L0 file and assemble into L1 full frame image (FFI)
    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    # L1 QLP is computed on the assembled (pre-bias-subtraction) image because
    # ImageProcessing mutates GREEN_CCD/RED_CCD in place during bias subtraction.
    l1_qlp_dir = build_qlp_dir(obs_id, 'L1', data_root=data_root_science)
    PlotL1(l1, output_dir=l1_qlp_dir).run('all')

    # assign calibration masters (bias, dark, flat, wls) to this frame
    calibration_association = CalibrationAssociation(l1, config)
    l1 = calibration_association.perform(['bias', 'dark', 'flat', 'thar'])

    # apply stardard FFI image processing (bias, dark, flat)
    image_processing = ImageProcessing(l1, config)
    l1 = image_processing.perform()

    # Run L1 diagnostics (compute and write metrics to PRIMARY) and QC (apply thresholds)
    DiagL1(l1).run()
    QCL1(l1).run()

    # extract 2D --> 1D spectra
    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    # attach precomputed wavelength solution (per-fiber WAVE arrays from WLS master)
    wavelength_calibration = WavelengthCalibration(l2, config)
    l2 = wavelength_calibration.perform()

    # Run L2 diagnostics (compute NaN counts and zero-flux fraction) and QC
    DiagL2(l2).run()
    QCL2(l2).run()

    # apply per-order barycentric correction (writes BJD_TDB, BARYCORR_KMS, BARYCORR_Z)
    barycentric_correction = BarycentricCorrection(l2, config)
    l2 = barycentric_correction.perform()

    # L2 QLP (wavelength-aware extracted-spectrum plots; requires attached WAVE)
    l2_qlp_dir = build_qlp_dir(obs_id, 'L2', data_root=data_root_science)
    PlotL2(l2, output_dir=l2_qlp_dir, obs_id=obs_id).run('all')

    # write L2 data product to disk
    l2_out_path = build_filepath(obs_id, 'L2', data_root=data_root_science)
    os.makedirs(os.path.dirname(l2_out_path), exist_ok=True)
    l2.to_fits(l2_out_path)

    # compute radial velocity (RV) from cross-correlation function (CCF)
    radial_velocity = RadialVelocity(l2, config)
    l4 = radial_velocity.perform()

    # write L4 data product to disk
    l4_out_path = build_filepath(obs_id, 'L4', data_root=data_root_science)
    os.makedirs(os.path.dirname(l4_out_path), exist_ok=True)
    l4.to_fits(l4_out_path)

    print("\n\n=== exiting kpf_drp_science pipeline ===\n\n")
