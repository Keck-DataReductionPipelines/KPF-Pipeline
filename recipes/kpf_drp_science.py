from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1

from kpfpipe.modules.image_assembly import ImageAssembly
#from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
#from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
#from kpfpipe.modules.barycentric_correction import BarycentricCorrection

from kpfpipe.utils.kpf import get_datecode, fetch_filepath


def main():
    print("\n\n=== entering kpf_drp_science pipeline ===\n\n")
    
    # Load target observation and corresponding masters
    obs_id = 'KP.YYYYMMDD.NNNNN.NN'
    datecode = get_datecode(obs_id)
    target_l0 = KPF0.from_fits(fetch_filepath(obs_id, level='L0'))

    #flat = KPF1.from_fits(fetch_filepath(datecode, master='flat'))
    #dark = KPF1.from_fits(fetch_filepath(datecode, master='dark'))
    bias = KPF1.from_fits(fetch_filepath(datecode, master='bias'))
    #wls = KPF1.from_fits(fetch_filepath(datecode, master='thar-wls'))

    # Perform L0 --> L1 data processing algorithms
    exposure_time = ExposureTime(target_l0)
    target_l0 = exposure_time.perform()

    image_assembly = ImageAssembly(target_l0)
    target_l1 = image_assembly.perform()

    #image_processing = ImageProcessing(target_l1)
    #target_l1 = image_processing.perform(flat, dark, bias)

    spectral_extraction = SpectralExtraction(target_l1)
    target_l2 = spectral_extraction.perform()

    #wavelength_calibration = WavelengthCalibration(target_l2)
    #target_l2 = wavelength_calibration.perform(wls)

    #barycentric_correction = BarycentricCorrection(target_l2)
    #target_l2 = barycentric_correction.perform()

    # Save L1 file to disk
    target_l2.to_fits()

    print("\n\n=== exiting kpf_drp_science pipeline ===\n\n")


if __name__ == '__main__':
    main()