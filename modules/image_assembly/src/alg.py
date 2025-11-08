import numpy as np
import pandas as pd
from copy import deepcopy

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
            self._read_channel_datasec_config(chip)
        
        self.prescan_region = self.cfg_params.get_config_value('prescan_region')


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


    def perform_overscan_subtraction(self, chip, namp):
        """
        Performs overscan subtraction steps, in order: 
            1. orient frame (positions overscan on right and bottom)
            2. subtract overscan (method chosen by user) 
            3. cuts off overscan region

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            namp (int) : number of amplifiers

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


    def make_overscan_arrays(self, image, chip, amp_no, clip=True):
        """
        Makes array of overscan pixels
        Assumes image orientaion: 
            - serial overscan on right
            - parallel overscan on bottom

        Args:
            image (ndarray) : data image, oriented using ImageAssemblyAlg.orient_channel()

        Returns:
            overscan_pixels_srl (np.ndarray): Array of serial overscan region pixels
            overscan_pixels_prl (np.ndarray): Array of parallel overscan region pixels
        """
        ncol_prescan = self.prescan_reg[1] - self.prescan_reg[0]
        ncol_datasec = self.channel_datasec[f'{chip.upper()}_NCOL']
        nrow_datasec = self.channel_datasec[f'{chip.upper()}_NROW']

        overscan_pixels_srl = np.arange(ncol_prescan + ncol_datasec, image.shape[1], 1)
        overscan_pixels_prl = np.arange(nrow_datasec, image.shape[0], 1)

        if clip:
            overscan_pixels_srl = overscan_pixels_srl[self.oscan_clip_no:-self.oscan_clip_no-1]
            overscan_pixels_prl = overscan_pixels_prl[self.oscan_clip_no:-self.oscan_clip_no-1]

        return overscan_pixels_srl, overscan_pixels_prl