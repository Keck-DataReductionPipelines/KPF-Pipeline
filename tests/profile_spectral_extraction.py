import os
import sys

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.kpf import get_datecode

from line_profiler import LineProfiler


def run():
    OBS_ID = 'KP.20240405.49597.71'    # 2-amp mode
    #OBS_ID = 'KP.20250419.84046.71'    # 4-amp mode

    datecode = get_datecode(OBS_ID)
    filepath = os.path.join('/data/kpf/L0/', datecode, f'{OBS_ID}.fits')

    target_l0 = KPF0.from_fits(filepath)
    
    image_assembly = ImageAssembly(target_l0)
    target_l1 = image_assembly.perform()

    spectral_extraction = SpectralExtraction(target_l1)
    target_l2 = spectral_extraction.perform()


if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(run)
    lp.run('run()')
    lp.print_stats(output_unit=1e-3)