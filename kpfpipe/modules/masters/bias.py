"""
KPF Master Bias construction module.
"""
from kpfpipe import DEFAULTS, DETECTOR
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels

DEFAULTS.update({
    'nframe_stream': 6,
    'stack_sigma': 5.0,
    'exptime_tolerance': 0.1,
})

NROW = DETECTOR['ccd']['nrow']
NCOL = DETECTOR['ccd']['ncol']


class Bias(BaseMasterModule):
    def __init__(self, l0_file_list, config=None):
        if isinstance(config, ConfigHandler):
            config = config.get_params(["DATA_DIRS", "KPFPIPE", "BIAS"])
        super().__init__(l0_file_list, config)


    def make_master_l1(self, l0_file_list=None, nstream=None, sigma=None):
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

            out = flag_outliers(l1_arrays[f'{chip}_IMG'], sigma, axis=0)
            bad = ((l1_arrays[f'{chip}_SNR'] <= 0) | (l1_arrays[f'{chip}_IMG'] == 0))

            l1_arrays[f'{chip}_MASK'] = ~(bad | out)

        self.ml1_obj = KPFMasterL1()

        for chip in self.chips:
            self.ml1_obj.set_data(f'{chip}_IMG',  l1_arrays[f'{chip}_IMG'])
            self.ml1_obj.set_data(f'{chip}_SNR',  l1_arrays[f'{chip}_SNR'])
            self.ml1_obj.set_data(f'{chip}_MASK', l1_arrays[f'{chip}_MASK'])

        self._set_input_files(l0_file_list)
        self.ml1_obj.receipt_add_entry('master_bias', 'PASS')

        return self.ml1_obj