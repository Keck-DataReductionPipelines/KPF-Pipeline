"""
KPF Image Assembly Module.

Orients and assembles raw data from amplifers into 
a full frame image. Processes data from L0 to L1.
"""
import numpy as np
import pandas as pd

from kpfpipe.data_models.level1 import KPF1


class ImageAssembly:
    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj
        
        CHIPS = ['GREEN', 'RED']
        for chip in CHIPS:
            self.count_amplifiers(chip)
            self.orient_channels(chip)

        # temporarily hard-code config params during development
        # switch to config once that interface has been standardized
        overscan_method = 'rowmedian'
        overscan_sigma_clip = 2.1

    
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
        if not hasattr(self.namp):
            self.namp = {}
        if not hasattr(self.dims):
            self.dims = {}

        chip = chip.upper()

        self.namp[chip] = 0
        for i in range(4):
            if f'{chip}_AMP{i+1}' in self.l0_obj.extensions:
                if np.size(self.l0_obj[f'{chip}_AMP{i+1}']) > 0:
                    self.namp[chip] += 1

        if self.namp[chip] == 2:
            self.dims[chip] = (4080, 2040)
        elif self.namp[chip] == 4:
            self.dims[chip] = (2040, 2040)
        else:
            raise ValueError(f"Only 2-amp and 4-amp mode supported, detected {self.namp[chip]} on {chip} CCD")
        

    def orient_channels(self, chip):
        """
        Extracts and flips amplifier channels to standardize readout orientation.
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
            channel_key = int(orientation.loc[orientation.CHANNEL_EXT == channel_ext, 'CHANNEL_KEY'])
            image = np.array(self.l0_obj[channel_ext])

            if channel_key == 1: # flip lr
                image_reoriented = np.flip(image,axis=1)
            elif channel_key == 2: # turn upside down and flip lr
                image_reoriented = np.flip(np.flip(image,axis=0),axis=1)
            elif channel_key == 3: # turn upside down
                image_reoriented = np.flip(image,axis=0)
            elif channel_key == 4: # no change
                image_reoriented = image

            self.l0_obj[channel_ext] = image_reoriented


    def _read_orientation_reference(self, chip):
        chip = chip.upper()
        
        if not hasattr(self, 'orientation'):
            self.orientation = {}

        # TODO: fix filepath handling and make it robust
        filepath = f'static/ccd_orientation_{chip.lower()}_{self.namp[chip]}amp.txt'
        with open(filepath, 'r') as f:
            self.orientation[chip] = pd.read_csv(f, delimiter=' ')

        return self.orientation[chip]

    
    def _get_overscan_pixels(self, chip, amp_no, prescan=[0,4], buffer=[5,5]):
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
        full_amplifier = self.l0_obj[f'{chip}_AMP{amp_no}']
        
        ncol_prescan = prescan[1] - prescan[0]
        nrow_imaging, ncol_imaging = self.dims[chip]

        oscan_pix_srl = full_amplifier[:,ncol_prescan+ncol_imaging:]
        oscan_pix_prl = full_amplifier[nrow_imaging:,:]

        oscan_pix_srl = oscan_pix_srl[:,buffer[0]:-buffer[1]-1]
        oscan_pix_prl = oscan_pix_prl[buffer[0]:-buffer[1]-1,:]

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
        full_amplifier = self.l0_obj[f'{chip}_AMP{amp_no}']
        
        ncol_prescan = prescan[1] - prescan[0]
        nrow_imaging, ncol_imaging = self.dims[chip]

        image_pix = full_amplifier[:nrow_imaging,ncol_prescan:ncol_prescan+ncol_imaging]
        
        return image_pix

    
    
    
    
    
    
    
    
    
    
    
    
    # from AnalyzeL0.measure_read_noise_overscan
    # also AnalyzeL0.measure_std_mad_norm_ratio_overscan
    def measure_read_noise(self, nparallel, nserial, sigma_clip):
        """
        Measure read noise from overscan region
        """