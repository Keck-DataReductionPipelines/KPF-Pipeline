import os

from kpfpipe.data_models.level0 import KPF0
#from kpfpipe.data_models.level1 import KPF1

from kpfpipe.modules.image_assembly import ImageAssembly
#from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
#from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
#from kpfpipe.modules.barycentric_correction import BarycentricCorrection

from kpfpipe.utils.pipeline import build_filepath


def main(config, args):
    print("\n\n=== entering kpf_drp_science pipeline ===\n\n")

    if not args.obs_id:
        raise SystemExit("Error: --obs_id is required for the science recipe (e.g. -o KP.20240405.40113.57)")

    obs_id = args.obs_id

    data_dirs = config.get_params(['DATA_DIRS'])
    data_root_in  = data_dirs['KPF_DATA_INPUT']
    data_root_out = data_dirs['KPF_DATA_OUTPUT']

    l0 = KPF0.from_fits(build_filepath(obs_id, 'L0', data_root=data_root_in))

    image_assembly = ImageAssembly(l0, config)
    l1 = image_assembly.perform()

    #image_processing = ImageProcessing(l1, config)
    #l1 = image_processing.perform()

    spectral_extraction = SpectralExtraction(l1, config)
    l2 = spectral_extraction.perform()

    #wavelength_calibration = WavelengthCalibration(l2, config)
    #l2 = wavelength_calibration.perform()

    #barycentric_correction = BarycentricCorrection(l2, config)
    #l2 = barycentric_correction.perform()

    out_path = build_filepath(obs_id, 'L2', data_root=data_root_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    l2.to_fits(out_path)

    print("\n\n=== exiting kpf_drp_science pipeline ===\n\n")
