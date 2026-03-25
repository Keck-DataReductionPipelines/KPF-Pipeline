"""
KPF Master Bias construction module.
"""
import numpy as np

from kpfpipe import DEFAULTS, DETECTOR
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels

DEFAULTS.update({
    'nframe_stream': 6,
    'stack_sigma': 5.0,
    'exptime_tolerance': 0.1,
})

NROW = DETECTOR['ccd']['nrow']
NCOL = DETECTOR['ccd']['ncol']


class Bias(BaseMasterModule):
    def __init__(self, l0_file_list, config={}):
        super().__init__(l0_file_list, config)


    def perform(self, l0_file_list=None, nstream=None, sigma=None):
        """
        Build master bias from stack
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nstream is None:
            nstream = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list, 
            nstream=nstream, 
            sigma=sigma
        )

        for chip in self.chips:
            img = l1_arrays[f'{chip}_IMG']
            snr = l1_arrays[f'{chip}_SNR']
            mask = l1_arrays[f'{chip}_MASK']

            l1_arrays[f'{chip}_IMG'] = interpolate_bad_pixels(img, mask)
            l1_arrays[f'{chip}_SNR'] = interpolate_bad_pixels(snr, mask)

            out = flag_outliers(l1_arrays[f'{chip}_IMG'], sigma)
            bad = (l1_arrays[f'{chip}_SNR']) <= 0 | (l1_arrays[f'{chip}_IMG'] == 0)

            l1_arrays[f'{chip}_MASK'] = ~(bad | out)

        # TODO: create L1 object (masters specific KPF1?)
        self.l1_arrays = l1_arrays

        return l1_arrays