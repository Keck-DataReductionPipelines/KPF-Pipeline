from kpfpipe.modules.masters.bias import Bias
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.modules.masters.flat import Flat
from kpfpipe.modules.masters.wls import WLS

from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction

from kpfpipe.utils import query_db_for_masters_stack

def main():
    print("\n\n=== entering kpf_drp_masters pipeline ===\n\n")
    
    datecode = 'YYYYMMDD'

    # make FFIs from stacks
    bias = Bias(query_db_for_masters_stack(datecode), 'bias').make_master()
    dark = Dark(query_db_for_masters_stack(datecode), 'dark').make_master()
    flat = Flat(query_db_for_masters_stack(datecode), 'flat').make_master()

    # reduce and extract individual wavecal files
    obs_ids = query_db_for_masters_stack(datecode)

    for i, obs_id in enumerate(obs_ids):
        filpath = fetch_filepath(obs_id)
        target_l0 = KPF0.from_fits(filpath)

        image_assembly = ImageAssembly(target_l0)
        target_ffi = image_assembly.perform()

        image_processing = ImageProcessing(target_ffi)
        target_ffi = image_processing.perform(flat, dark, bias)

        spectral_extraction = SpectralExtraction(target_ffi)
        target_l1 = spectral_extraction.perform()

        target_l1.to_fits()

    # calculate wavelength solution
    wls = WLS(query_db_for_masters_stack(datecode), 'thar-wls').make_master()

    print("\n\n=== exiting kpf_drp_masters pipeline ===\n\n")


if __name__ == '__main__':
    main()