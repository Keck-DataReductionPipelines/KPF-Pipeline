import numpy as np
import pandas as pd
from copy import deepcopy
from astropy.stats import mad_std

from kpfpipe.config.pipeline_config import ConfigClass
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.logger import start_logger
from modules.quicklook.src.diagnostics import add_headers_L0_read_noise

class ImageAssemblyAlg:
    """
    Docstring
    """
    def __init__(self, 
                 target_l0,
                 default_config_path,
                 logger=None
                 ):
        # start logger
        if logger == None:
            self.log = start_logger('ImageAssembly', default_config_path)
        else:
            self.log = logger

        # data inputs
        self.target_l0 = target_l0

        # config inputs
        self.config = ConfigClass(default_config_path)
        self.cfg_params = ConfigHandler(self.config, 'PARAM')

        for chip in ['GREEN', 'RED']:
            self._infer_amplifier_mode(chip)
            self._read_orientation_reference(chip)
        
        self.prescan_region = self.cfg_params.get_config_value('prescan_region')
        self.overscan_method = self.cfg_params.get_config_value('overscan_method')
        self.overscan_clip = int(self.cfg_params.get_config_value('overscan_clip'))
        self.overscan_sigma = float(self.cfg_params.get_config_value('overscan_sigma'))

        # recompute readnoise
        self.target_l0 = add_headers_L0_read_noise(self.target_l0)

    
    # GJG: legacy code infers amplifer mode from header['{chip}AMP{i}]['NAXIS{1,2}]
    def _infer_amplifier_mode(self, chip):
        if not hasattr(self, 'namp'):
            self.namp = {}
        
        chip = chip.upper()
        extensions = list(self.target_l0.extensions.keys())
        matches = [x for x in extensions if x.startswith(f'{chip}_AMP')]

        if len(matches) == 2:
            if np.all(np.isin([f'{chip}_AMP{i}' for i in [1,2]], matches)):
                self.namp[chip] = 2
            else:
                raise ValueError(f"Unexpected extensions for namp = 2 : {matches}")
        elif len(matches) == 4:
            if np.all(np.isin([f'{chip}_AMP{i}' for i in [1,2,3,4]], matches)):
                self.namp[chip] = 4
            else:
                raise ValueError(f"Unexpected extensions for namp = 4 : {matches}")
        else:
            raise ValueError(f"Expected 2 or 4 amplifers, detected {len(matches)}")

        keyword_map = {'GREEN':'GRNAMPS', 'RED','REDAMPS'}
        self.target_l0['PRIMARY'][keyword_map[chip]] = self.namp[chip]


    def _read_orientation_reference(self, chip):
        if not hasattr(self, 'orientation'):
            self.orientation = {}

        filepath = str(self.cfg_params.get_config_value(f'channel_orientation_ref_path_{chip.lower()}'))
        with open(filepath, 'r') as f:
            self.orientation[chip.upper()] = pd.read_csv(f, delimiter=' ')


    def _get_datasec_ncol_nrow(self, chip):
        if self.namp[chip.upper()] == 2:
            ncol_datasec = 2040
            nrow_datasec = 4080
        elif self.namp[chip.upper()] == 4:
            ncol_datasec = 2040
            nrow_datasec = 2040
        else:
            raise ValueError("Only 2-amp and 4-amp modes supported")

        return ncol_datasec, nrow_datasec
                
    
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
        image = deepcopy(np.array(self.target_l0[channel]))

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
        
        ncol_prescan = self.prescan_region[1] - self.prescan_region[0]
        ncol_datasec, nrow_datasec = self._get_datasec_ncol_nrow(chip)

        oscan_pix_srl = image[:,ncol_prescan+ncol_datasec:]
        oscan_pix_prl = image[nrow_datasec:,:]

        if clip:
            oscan_pix_srl = oscan_pix_srl[:,self.overscan_clip:-self.overscan_clip-1]
            oscan_pix_prl = oscan_pix_prl[self.overscan_clip:-self.overscan_clip-1,:]

        return oscan_pix_srl, oscan_pix_prl


    def remove_overscan_pixels(self, chip, amp_no):
        """
        Removes overscan pixels from full amplifier region
        Assumes image orientaion has not been altered from raw L0 file. 

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            amp (int) : amplifier number

        Returns:
            ndarray : image with only datasec pixels
        """
        chip = chip.upper()
        channel = f'{chip}_AMP{amp_no}'
        image = self.orient_channel(chip, amp_no)
        
        ncol_prescan = self.prescan_region[1] - self.prescan_region[0]
        ncol_datasec, nrow_datasec = self._get_datasec_ncol_nrow(chip)

        self.target_l0[channel] = image[:nrow_datasec,ncol_prescan:ncol_prescan+ncol_datasec]
        self.target_l0[channel] = self.orient_channel(chip, amp_no)
        
        return self.target_l0[channel]


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

        self.target_l0[channel] = deepcopy(np.array(self.target_l0[channel]))
        self.target_l0[channel] = self.remove_overscan_pixels(chip, amp_no)

        return self.target_l0[channel]

    
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
        image = self.orient_channel(chip, amp_no)

        oscan_pix_srl, _ = self.get_overscan_pixels(chip, amp_no, clip=clip)

        self.target_l0[channel] = (image.T - np.nanmedian(oscan_pix_srl, axis=1)).T
        self.target_l0[channel] = self.orient_channel(chip, amp_no)
        self.target_l0[channel] = self.remove_overscan_pixels(chip, amp_no)

        return self.target_l0[channel]

    
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
        image = self.orient_channel(chip, amp_no)

        oscan_pix_srl, _ = self.get_overscan_pixels(chip, amp_no, clip=clip)

        if sigma is None:
            sigma = self.overscan_sigma

        p16, p50, p84 = np.nanpercentile(oscan_pix_srl, [16,50,84])
        dispersion = 0.5 * (p84 - p16)
        out = np.abs(oscan_pix_srl - p50)/dispersion > sigma

        self.target_l0[channel] = self.target_l0[channel] - np.nanmean(oscan_pix_srl[~out])
        self.target_l0[channel] = self.remove_overscan_pixels(chip, amp_no)

        return self.target_l0[channel]


    def stitch_channels(self, chip):
        """
        Stitch together all amplifier regions (i.e. channels) from a chip
        Automatically checks for 2 vs 4 amplifier mode
        Applies gain correction
        Calculates 2D variance image
        
        Assumes amplifier regions are already in proper orientation with overscan pre-subtracted
       
        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'

        Returns:
            ndarray : 4080 x 4080 data image with only datasec pixels
        """
        chip = chip.upper()
        image_ffi = np.zeros((4080,4080))
        var2d_ffi = np.zeros((4080,4080))

        if self.namp[chip] == 2:
            image_ffi[:,:2040] = self.target_l0[f'{chip}_AMP1'] * self.target_l0.header[f'{chip}_AMP1']['CCDGAIN']
            image_ffi[:,2040:] = self.target_l0[f'{chip}_AMP2'] * self.target_l0.header[f'{chip}_AMP2']['CCDGAIN']
            var2d_ffi[:,:2040] = np.abs(image_ffi[:,:2040]) + self.target_l0.header['PRIMARY'][f'RN{chip}1']
            var2d_ffi[:,2040:] = np.abs(image_ffi[:,2040:]) + self.target_l0.header['PRIMARY'][f'RN{chip}2']

        elif self.namp[chip] == 4:
            raise ValueError("4-amp mode not yet implemented")
        
        else:
            raise ValueError("Only 2-amp and 4-amp modes supported")

        # flip green ccd
        if chip == 'GREEN':
            image_ffi = np.flip(image_ffi, axis=0)
            var2d_ffi = np.flip(var2d_ffi, axis=0)

        # GJG: 2**16 correction was hard-coded in previous version -- why?
        self.target_l0[f'{chip}_CCD'] = image_ffi / (2**16)
        self.target_l0[f'{chip}_VAR'] = var2d_ffi / (2**16)

        self.target_l0.header[f'{chip}_CCD']['BUNIT'] = ('electrons','Units of image data')
        self.target_l0.header[f'{chip}_VAR']['BUNIT'] = ('electrons','Units of image data')

        return self.target_l0[f'{chip}_CCD'], self.target_l0[f'{chip}_VAR']


    def assemble_image(self, chip, overscan_method=None):
        """
        Performs all image assembly steps, in order: 
            1. subtract overscan from each channel
            2. cuts off overscan region from each channel
            3. orient channels and stitch together full frame image
            4. calculate 2D variance image

        Args:
            chip (str) : which CCD to use, 'GREEN' or 'RED'
            method (str) : method to use for overscan subtraction

        Returns:
            kpf_l0 : KPF L0 object with full frame extensions and keywords
        """
        if overscan_method is None:
            try:
                overscan_method = self.__getattribute__(self.overscan_method)
            except AttributeError:
                self.log.error(f'Overscan correction method {self.overscan_method} not implemented.')
                raise(AttributeError)

        for amp_no in range(1, 1+self.namp[chip.upper()]):
            print(chip, amp_no)
            self.target_l0[f'{chip.upper()}_AMP{amp_no}'] = overscan_method(chip, amp_no)

        self.target_l0[f'{chip}_CCD'], self.target_l0[f'{chip}_VAR'] = self.stitch_channels(chip)

        return self.target_l0


    def add_keywords(self):
        self.target_l0.header[self.ffi_exts[frame]]['BUNIT'] = ('electrons','Units of image data')