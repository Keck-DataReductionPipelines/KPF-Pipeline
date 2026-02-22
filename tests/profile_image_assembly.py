import os
import sys

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.utils.kpf import get_datecode

from line_profiler import LineProfiler


def run():
    OBS_ID = 'KP.20240405.49597.71'    # 2-amp mode
    #OBS_ID = 'KP.20250419.84046.71'    # 4-amp mode

    datecode = get_datecode(OBS_ID)
    filepath = os.path.join('/data/kpf/L0/', datecode, f'{OBS_ID}.fits')

    target_l0 = KPF0.from_fits(filepath)
    image_assembly = ImageAssembly(target_l0)

    for chip in image_assembly.chips:
        image_assembly.count_amplifiers(chip)
        image_assembly.orient_channels(chip)
        image_assembly.apply_gain_conversion(chip)
        image_assembly.measure_read_noise(chip)
        image_assembly.subtract_overscan(chip, 'rowmedian')
        image_assembly.orient_channels(chip)
        
        ccd_ffi, var_ffi = image_assembly.stitch_ffi(chip)


if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(run)
    lp.run('run()')
    lp.print_stats(output_unit=1e-3)