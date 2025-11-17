import numpy as np
import pandas as pd
from copy import deepcopy
from astropy.stats import mad_std

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler

class ImageAssemblyAlg:
    """
    Docstring
    """
    def __init__(self, 
                 target_L0,
                 data_type,
                 default_config_path,
                 logger=None
                 ):
        # start logger
        if logger == None:
            self.log = start_logger('ImageAssembly', default_config_path)
        else:
            self.log = logger

        # data inputs
        self.target_L0 = target_L0
        self.data_type = data_type

        # config inputs
        self.config = ConfigClass(default_config_path)
        self.cfg_params = ConfigHandler(self.config, 'PARAM')

        for chip in ['GREEN', 'RED']:
            self._read_orientation_reference(chip)
            #self._read_channel_datasec_config(chip)
        
        self.prescan_region = self.cfg_params.get_config_value('prescan_region')
        self.overscan_clip = int(self.cfg_params.get_config_value('overscan_clip'))
        self.overscan_sigma = float(self.cfg_params.get_config_value('overscan_sigma'))

        # GJG: temporarily hard-coding number of amplifers for development
        # GJG: need to write function to infer number of amplifers from headers/extensions
        self.namp = {'GREEN':2, 'RED':2}


    def _read_orientation_reference(self, chip):
        if not hasattr(self, 'orientation'):
            self.orientation = {}

        filepath = str(self.cfg_params.get_config_value(f'channel_orientation_ref_path_{chip.lower()}'))
        with open(filepath, 'r') as f:
            self.orientation[chip.upper()] = pd.read_csv(f, delimiter=' ')


    def _read_channel_datasec_config(self, chip):
        if not hasattr(self, 'channel_datasec'):
            self.channel_datasec = {}

        ncol = int(self.cfg_params.get_config_value(f'channel_datasec_ncols_{chip.lower()}'))
        nrow = int(self.cfg_params.get_config_value(f'channel_datasec_nrows_{chip.lower()}'))
        self.channel_datasec[f'{chip.upper()}_NCOL'] = ncol
        self.channel_datasec[f'{chip.upper()}_NROW'] = nrow


    def perform_overscan_subtraction(self, chip):
        """
        Performs overscan subtraction steps, in order: 
            1. orient frame (positions overscan on right and bottom)
            2. subtract overscan (method chosen by user) 
            3. cuts off overscan region

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'

        Returns:
            image_stitched (np.ndarray) : Stiched-together full frame image, with overscan subtracted and removed
        """
        for amp_no in range(1,namp+1):
            channel = f'{chip.upper()}_AMP{amp_no}'
        
            image_raw_with_oscan = deepcopy(np.array(self.target_L0[channel]))
            image_reoriented_with_oscan = self.adjust_orientation(chip.upper(), amp_no)

    
    def orient_channel(self, chip, amp_no):
        """
        Extracts and flips single-amplifier image to standardize readout orientation for overscan subtraction.
            - serial overscan on right
            - parallel overscan on bottom
        All transformations are flips, so a second call to this function will undo the transformation.

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp (int) : amplifier number
        
        Returns:
            np.ndarray : Correctly-oriented raw image for single amplifier region
        """
        chip = chip.upper()
        channel = f'{chip}_AMP{amp_no}'
        image = deepcopy(np.array(self.target_L0[channel]))

        orientation = self.orientation[chip]
        channel_key = int(orientation.loc[orientation.CHANNEL_EXT == channel, 'CHANNEL_KEY'])

        if channel_key == 1: # flip lr
            image_reoriented = np.flip(image,axis=1)
        elif channel_key == 2: # turn upside down and flip lr
            image_reoriented = np.flip(np.flip(image,axis=0),axis=1)
        elif channel_key == 3: # turn upside down
            image_reoriented = np.flip(image,axis=0)
        elif channel_key == 4: # no change
            image_reoriented = image

        return image_reoriented


    # GJG TODO: replace clip = True --> skip_cols = (int,int)
    def get_overscan_pixels(self, chip, amp_no, clip=True):
        """
        Extracts array of overscan pixel from full amplifier region
        Assumes image orientaion has not been altered from raw L0 file. 

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp (int) : amplifier number

        Returns:
            oscan_pix_srl (np.ndarray): Array of serial overscan pixels
            oscan_pix_prl (np.ndarray): Array of parallel overscan pixels
        """
        image = self.orient_channel(chip, amp_no)
        
        if self.namp[chip.upper()] == 2:
            ncol_datasec = 2040
            nrow_datasec = 4080
        elif self.namp[chip.upper()] == 4:
            ncol_datasec = 2040
            nrow_datasec = 2040
        else:
            raise ValueError("Only 2-amp and 4-amp modes supported")

        ncol_prescan = self.prescan_region[1] - self.prescan_region[0]

        oscan_pix_srl = image[:,ncol_prescan+ncol_datasec:]
        oscan_pix_prl = image[nrow_datasec:,:]

        if clip:
            oscan_pix_srl = oscan_pix_srl[:,self.overscan_clip:-self.overscan_clip-1]
            oscan_pix_prl = oscan_pix_prl[self.overscan_clip:-self.overscan_clip-1,:]

        return oscan_pix_srl, oscan_pix_prl


    def zero(self, chip, amp_no, clip=True):
        """
        Sets overscan subtraction to zero, returns raw image

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp_no (int) : amplifier number
            
        Returns:
            np.ndarray : raw image with zero overscan subtracted
        """
        chip = chip.upper()
        channel = f'{chip}_AMP{amp_no}'
        image = deepcopy(np.array(self.target_L0[channel]))

        return image

    
    def rowmedian(self, chip, amp_no, clip=True):
        """
        Calculates median of parallel overscan region; subtracts from raw image

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp_no (int) : amplifier number
            
        Returns:
            np.ndarray : raw image with row-by-row median overscan subtracted
        """
        chip = chip.upper()
        channel = f'{chip}_AMP{amp_no}'
        image = deepcopy(np.array(self.target_L0[channel]))

        oscan_pix_srl, _ = self.get_overscan_pixels(chip, amp_no, clip=clip)

        return image - np.nanmedian(oscan_pix_srl)

    
    def clippedmean(self, chip, amp_no, clip=True, sigma=None):
        """
        Calculates clippedmean of parallel overscan region; subtracts from raw image

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp_no (int) : amplifier number
            
        Returns:
            np.ndarray : raw image with median overscan subtracted
        """
        chip = chip.upper()
        channel = f'{chip}_AMP{amp_no}'
        image = deepcopy(np.array(self.target_L0[channel]))

        oscan_pix_srl, _ = self.get_overscan_pixels(chip, amp_no, clip=clip)

        if sigma is None:
            sigma = self.overscan_sigma

        p16, p50, p84 = np.nanpercentile(oscan_pix_srl, [16,50,84])
        dispersion = 0.5 * (p84 - p16)
        out = np.abs(oscan_pix_srl - p50)/dispersion > sigma

        return image - np.nanmean(oscan_pix_srl[~out])