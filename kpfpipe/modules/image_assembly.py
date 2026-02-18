"""
KPF Image Assembly Module.

Orients and assembles raw data from amplifers into 
a full frame image. Processes data from L0 to L1.
"""
import numpy as np
import pandas as pd

from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.stats import flag_outliers

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

class ImageAssembly:
    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj
        self.CHIPS = ['GREEN', 'RED']

        # temporarily hard-code config params during development
        # switch to config once that interface has been standardized
        self.overscan_method = 'rowmedian'

    
    def count_amplifiers(self, chip):
        """
        Determine if extensions are present for a given CCD and
        count the number of amplifier regions. Sets attributes to
        track chips, namp, and dims (i.e. channel dimensions)
        
        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
        
        Returns:
            None
        """
        if not hasattr(self, 'namp'):
            self.namp = {}
        if not hasattr(self, 'dims'):
            self.dims = {}

        chip = chip.upper()

        self.namp[chip] = 0
        for i in range(4):
            if f'{chip}_AMP{i+1}' in self.l0_obj.extensions:
                if np.size(self.l0_obj.data[f'{chip}_AMP{i+1}']) > 0:
                    self.namp[chip] += 1

        if self.namp[chip] == 2:
            self.dims[chip] = (4080, 2040)
        elif self.namp[chip] == 4:
            self.dims[chip] = (2040, 2040)
        else:
            raise ValueError(f"Only 2-amp and 4-amp mode supported, detected {self.namp[chip]} on {chip} CCD")
        

    def _read_orientation_reference(self, chip):
        chip = chip.upper()
        
        if not hasattr(self, 'orientation'):
            self.orientation = {}

        filepath = f'{REPO_ROOT}/static/ccd_orientation_{chip.lower()}.txt'
        with open(filepath, 'r') as f:
            self.orientation[chip] = pd.read_csv(f, delimiter=' ')

        return self.orientation[chip]

    
    def orient_channels(self, chip):
        """
        Reorients amplifier channels in place to standardize readout orientation.
            - serial overscan on right
            - parallel overscan on bottom
        All transformations are flips, so a second call to this function will undo the transformation.

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
        
        Returns:
            None
        """
        chip = chip.upper()
        orientation = self._read_orientation_reference(chip)

        for i in range(self.namp[chip]):
            channel_ext = f'{chip.upper()}_AMP{i+1}'
            channel_key = orientation.loc[orientation.CHANNEL_EXT == channel_ext, 'CHANNEL_KEY'].item()
            image = self.l0_obj.data[channel_ext]

            if channel_key == 1: # flip lr
                image_reoriented = np.flip(image,axis=1)
            elif channel_key == 2: # turn upside down and flip lr
                image_reoriented = np.flip(np.flip(image,axis=0),axis=1)
            elif channel_key == 3: # turn upside down
                image_reoriented = np.flip(image,axis=0)
            elif channel_key == 4: # no change
                image_reoriented = image

            self.l0_obj.data[channel_ext] = image_reoriented


    def apply_gain_conversion(self, chip):
        """
        Apply gain to convert ADU to photo-electrons
        """
        # TODO: move gain to static config file
        GAIN = {
            'GREEN_AMP1': 5.175,
            'GREEN_AMP2': 5.208,
            'GREEN_AMP3': 5.52,
            'GREEN_AMP4': 5.39,
            'RED_AMP1': 5.02,
            'RED_AMP2': 5.27,
            'RED_AMP3': 5.32,
            'RED_AMP4': 5.23,
        }
        
        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f'{chip}_AMP{i+1}'
            self.l0_obj.data[channel_ext] *= GAIN[channel_ext] / (2 ** 16)
                

    def _get_overscan_pixels(self, chip, amp_no, prescan=[0,4], buffer=[0,0]):
        """
        Gets array of overscan pixel from full amplifier region
        Assumes image orientaion has been standardized
            - serial overscan on right
            - parallel overscan on bottom

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp_no (int) : amplifier number

        Returns:
            oscan_pix_srl (np.ndarray): Array of serial overscan pixels
            oscan_pix_prl (np.ndarray): Array of parallel overscan pixels
        """
        chip = chip.upper()
        full_amplifier = np.array(self.l0_obj.data[f'{chip}_AMP{amp_no}'], dtype=np.float32)
        
        ncol_prescan = prescan[1] - prescan[0]
        nrow_imaging, ncol_imaging = self.dims[chip]

        oscan_pix_srl = full_amplifier[:nrow_imaging,ncol_prescan+ncol_imaging:]
        oscan_pix_prl = full_amplifier[nrow_imaging:,:ncol_prescan+ncol_imaging]

        start = buffer[0] if buffer[0] > 0 else None
        end = -buffer[1] if buffer[1] > 0 else None

        oscan_pix_srl = oscan_pix_srl[:, start:end]
        oscan_pix_prl = oscan_pix_prl[start:end, :]

        return oscan_pix_srl, oscan_pix_prl


    def _get_imaging_pixels(self, chip, amp_no, prescan=[0,4]):
        """
        Gets array of imaging pixels from full amplifier region
        Assumes image orientaion has been standardized
            - serial overscan on right
            - parallel overscan on bottom

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp_no (int) : amplifier number

        Returns:
            ndarray : data with only active imaging area pixels
        """
        chip = chip.upper()
        full_amplifier = np.array(self.l0_obj.data[f'{chip}_AMP{amp_no}'], dtype=np.float32)
        
        ncol_prescan = prescan[1] - prescan[0]
        nrow_imaging, ncol_imaging = self.dims[chip]

        image_pix = full_amplifier[:nrow_imaging,ncol_prescan:ncol_prescan+ncol_imaging]
        
        return image_pix


    def measure_read_noise(self, chip, prescan=[0,4], buffer=[5,5], sigma=10.0):
        """
        Measure read noise from overscan region
        """
        if not hasattr(self, 'readnoise'):
            self.readnoise = {}
        if not hasattr(self, 'rn_nongauss'):
            self.rn_nongauss = {}

        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f'{chip}_AMP{i+1}'

            oscan_srl, _ = self._get_overscan_pixels(chip, i+1, prescan, buffer)
            
            out = flag_outliers(oscan_srl, sigma, method='median')
            std = np.nanstd(oscan_srl[~out])
            mad = np.nanmean(np.abs(oscan_srl[~out] - np.nanmean(oscan_srl[~out])))
            
            self.readnoise[channel_ext] = std
            self.rn_nongauss[channel_ext] = np.sqrt(2/np.pi) * std / mad


    def _oscan_zero(self, chip, amp_no, **kwargs):
        """
        Returns overscan bias level of zero
        """
        return 0.0

    def _oscan_median(self, chip, amp_no, **kwargs):
        """
        Calculates single-value median of serial overscan region
        """
        oscan_srl, _ = self._get_overscan_pixels(chip, amp_no, **kwargs)
        bias = np.nanmedian(oscan_srl)
        return bias


    def _oscan_rowmedian(self, chip, amp_no, **kwargs):
        """
        Calculates row-by-row median of serial overscan region
        """
        oscan_srl, _ = self._get_overscan_pixels(chip, amp_no, **kwargs)
        bias = np.nanmedian(oscan_srl, axis=1)[:,None]
        return bias


    def subtract_overscan(self, chip, method, prescan=[0,4], buffer=[0,0]):
        """
        Performs the following operations
          - estimates overscan bias level
          - subtracts overscan bias from active imaging pixels
          - removes overscan region from channel, leaving only imaging pixels

        Supported methods are 'zero', 'median', and 'rowmedian'
        """
        try:
            oscan_fxn = self.__getattribute__(f'_oscan_{method}')
        except AttributeError as e:
            raise AttributeError(f"Unsupported overscan subtraction method: '{method}'")
        
        for i in range(self.namp[chip]):
            image = self._get_imaging_pixels(chip, i+1)
            bias = oscan_fxn(chip, i+1, prescan=prescan, buffer=buffer)
            self.l0_obj.data[f'{chip.upper()}_AMP{i+1}'] = np.array(image - bias, dtype=np.float32)


    def stitch_ffi(self, chip, prescan=[0,4]):
        chip = chip.upper()

        ccd_ffi = np.zeros((4080,4080), dtype=np.float32)
        var_ffi = np.zeros((4080,4080), dtype=np.float32)

        if self.namp[chip] == 2:
            ccd_ffi[:,:2040] = self.l0_obj.data[f'{chip}_AMP1']
            ccd_ffi[:,2040:] = self.l0_obj.data[f'{chip}_AMP2']
            var_ffi[:,:2040] = np.abs(ccd_ffi[:,:2040]) + self.readnoise[f'{chip}_AMP1']
            var_ffi[:,2040:] = np.abs(ccd_ffi[:,2040:]) + self.readnoise[f'{chip}_AMP2']

        elif self.namp[chip] == 4:
            ccd_ffi[:2040,:2040] = self.l0_obj.data[f'{chip}_AMP1']
            ccd_ffi[:2040,2040:] = self.l0_obj.data[f'{chip}_AMP2']
            ccd_ffi[2040:,:2040] = self.l0_obj.data[f'{chip}_AMP3']
            ccd_ffi[2040:,2040:] = self.l0_obj.data[f'{chip}_AMP4']
            var_ffi[:2040,:2040] = np.abs(ccd_ffi[:2040,:2040]) + self.readnoise[f'{chip}_AMP1']
            var_ffi[:2040,2040:] = np.abs(ccd_ffi[:2040,2040:]) + self.readnoise[f'{chip}_AMP2']
            var_ffi[2040:,:2040] = np.abs(ccd_ffi[2040:,:2040]) + self.readnoise[f'{chip}_AMP3']
            var_ffi[2040:,2040:] = np.abs(ccd_ffi[2040:,2040:]) + self.readnoise[f'{chip}_AMP4']
        
        else:
            raise ValueError(f"Only 2-amp and 4-amp mode supported, detected {self.namp[chip]} on {chip} CCD")


        if chip == 'GREEN':
            ccd_ffi = np.flip(ccd_ffi, axis=0)
            var_ffi = np.flip(var_ffi, axis=0)
                
        return ccd_ffi, var_ffi
    



    
    def perform(self):
        for chip in self.CHIPS:
            self.count_amplifiers(chip)
            self.orient_channels(chip)
            self.apply_gain_conversion(chip)
            self.measure_read_noise(chip)
            self.subtract_overscan(chip)
            self.assemble_ffi(chip)

        # TODO: create KPF1 object and return