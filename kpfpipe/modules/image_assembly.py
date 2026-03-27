"""
KPF Image Assembly module.
"""
import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT, DEFAULTS, DETECTOR
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers

DEFAULTS.update({
    'overscan_method': 'rowmedian',
})

DEFAULTS.update(DETECTOR)

_RN_KEYS = {
    'GREEN_AMP1': ['RNGRN1', 'RNNGGR1'],
    'GREEN_AMP2': ['RNGRN2', 'RNNGGR2'],
    'GREEN_AMP3': ['RNGRN3', 'RNNGGR3'],
    'GREEN_AMP4': ['RNGRN4', 'RNNGGR4'],
    'RED_AMP1': ['RNRED1', 'RNNGRD1'],
    'RED_AMP2': ['RNRED2', 'RNNGRD2'],
    'RED_AMP3': ['RNRED3', 'RNNGRD3'],
    'RED_AMP4': ['RNRED4', 'RNNGRD4'],
}

# TODO: use prescan from detector.toml config
# TODO: add overscan and readnoise params to main config

class ImageAssembly:
    """
    This class performs CCD-level processing to convert L0 data to L1.

    Operations include:
      - orienting amplifier channels
      - applying gain conversion (ADU --> photo-electrons)
      - measuring read noise
      - subtracting overscan bias
      - assembling full-frame images (FFI)
    """
    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_IMAGE_ASSEMBLY"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        for k, v in self.ccd.items():
            setattr(self, k, v)

        self._parse_amplifier_reference()

    
    def _parse_amplifier_reference(self):
        """
        Load orientation mapping and gain for amplifier channels.

        Orientation keys indicate how to flip/rotate each amplifier channel to 
        standard orientation (serial overscan on right, parallel overscan on bottom).
        Cached in `self.orientation` for repeated use.
        """
        if not hasattr(self, 'orientation'):
            self.orientation = {}
        if not hasattr(self, 'gain'):
            self.gain = {}

        for chip in self.chips:
            chip = chip.upper()
            df = pd.DataFrame(self.amplifiers[chip]).set_index('channel_id')
            self.orientation.update(dict(zip(df['ext_name'], df['flip'])))
            self.gain.update(dict(zip(df['ext_name'], df['gain'])))


    def count_amplifiers(self, chip):
        """
        Count the number of amplifier extensions present for a given CCD and
        determine their channel dimensions.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.

        Returns
        -------
        None

        Notes
        -----
        Sets instance attributes:
        - `self.namp[chip]` : number of amplifier regions detected.
        - `self.dims[chip]` : shape of each amplifier channel.
        Only 2-amp and 4-amp configurations are supported.
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
            self.dims[chip] = (self.nrow, self.ncol // 2)
        elif self.namp[chip] == 4:
            self.dims[chip] = (self.nrow // 2, self.ncol // 2)
        else:
            raise ValueError(f"Only 2-amp and 4-amp mode supported, detected {self.namp[chip]} on {chip} CCD")
        

    def orient_channels(self, chip):
        """
        Reorient amplifier channels to a standard orientation in-place.
        (serial overscan on right, parallel overscan on bottom)

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.

        Returns
        -------
        None

        Notes
        -----
        The transformations are flips; calling twice will undo the operation.
        """
        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f'{chip.upper()}_AMP{i+1}'
            flip = self.orientation[channel_ext]
            image = self.l0_obj.data[channel_ext]

            if flip == 'rows':
                image_reoriented = np.flip(image,axis=0)
            elif flip == 'cols':
                image_reoriented = np.flip(image,axis=1)
            elif flip == 'both':
                image_reoriented = np.flip(image,axis=(0,1))
            elif flip == 'none':
                image_reoriented = image
            else:
                raise ValueError("unexpected 'flip' entry found in orientation reference")

            self.l0_obj.data[channel_ext] = image_reoriented


    def apply_gain_conversion(self, chip):
        """
        Convert pixel values from ADU to photo-electrons using amplifier-specific gain.
        Amplifier channels are modified in-place.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.

        Returns
        -------
        None

        Notes
        -----
        Conversion formula: pixel_electrons = pixel_ADU * gain / 65536
        """
        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f'{chip}_AMP{i+1}'
            self.l0_obj.data[channel_ext] *= self.gain[channel_ext] / (2 ** 16)
                

    def _get_overscan_pixels(self, chip, amp_no, prescan=[0,4], buffer=[0,0]):
        """
        Extract overscan pixels for a given amplifier.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        amp_no : int
            Amplifier number (1-4).
        prescan : list of int, optional
            Columns corresponding to prescan region [start, end].
        buffer : list of int, optional
            Number of pixels to ignore at edges [start, end].

        Returns
        -------
        oscan_pix_srl : ndarray
            Serial overscan pixels (columns beyond imaging area).
        oscan_pix_prl : ndarray
            Parallel overscan pixels (rows beyond imaging area).

        Notes
        -----
        Assumes image orientation has been standardized.
        """
        chip = chip.upper()
        full_amplifier = self.l0_obj.data[f'{chip}_AMP{amp_no}']
        
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
        Extract imaging pixels (active CCD area) for a given amplifier.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        amp_no : int
            Amplifier number (1-4).
        prescan : list of int, optional
            Columns corresponding to prescan region [start, end].

        Returns
        -------
        ndarray
            2D array of imaging pixels.

        Notes
        -----
        Assumes image orientation has been standardized.
        """
        chip = chip.upper()
        full_amplifier = self.l0_obj.data[f'{chip}_AMP{amp_no}']
        
        ncol_prescan = prescan[1] - prescan[0]
        nrow_imaging, ncol_imaging = self.dims[chip]

        image_pix = full_amplifier[:nrow_imaging,ncol_prescan:ncol_prescan+ncol_imaging]
        
        return image_pix


    def measure_read_noise(self, chip, prescan=[0,4], buffer=[5,5], sigma=10.0):
        """
        Estimate read noise for each amplifier from overscan pixels.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        prescan : list of int, optional
            Columns corresponding to prescan region [start, end].
        buffer : list of int, optional
            Number of pixels to ignore at the edges [start, end].
        sigma : float, optional
            Threshold for sigma clipping overscan pixels.

        Returns
        -------
        None

        Notes
        -----
        Stores results in:
        - `self.readnoise[channel_ext]` : standard deviation of cleaned overscan.
        - `self.rn_nongauss[channel_ext]` : non-Gaussian factor computed as std/mad.
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


    def subtract_overscan(self, chip, method=None, prescan=[0,4], buffer=[0,0]):
        """
        Subtract overscan bias from imaging pixels for each amplifier. Also
        removes overscan region from amplifier channel, leaving only active
        imaging area pixels. Amplifier channels are modified in-place.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        method : str
            Overscan subtraction method ('zero', 'median', 'rowmedian').
        prescan : list of int, optional
            Columns corresponding to prescan region.
        buffer : list of int, optional
            Number of pixels to ignore at edges.

        Returns
        -------
        None
        """
        if method is None:
            method = self.overscan_method

        try:
            oscan_fxn = self.__getattribute__(f'_oscan_{method}')
        except AttributeError as e:
            raise AttributeError(f"Unsupported overscan subtraction method: '{method}'")
        
        for i in range(self.namp[chip]):
            image = self._get_imaging_pixels(chip, i+1)
            bias = oscan_fxn(chip, i+1, prescan=prescan, buffer=buffer)
            self.l0_obj.data[f'{chip.upper()}_AMP{i+1}'] = image - bias


    def stitch_ffi(self, chip, prescan=[0,4]):
        """
        Combine individual amplifier channels into a full-frame image (FFI).

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        prescan : list of int, optional
            Columns corresponding to prescan region.

        Returns
        -------
        ccd_ffi : ndarray
            Full-frame data image.
        var_ffi : ndarray
            Full-frame variance image, incorporating read noise.

        Notes
        -----
        Supports 2-amp and 4-amp CCD configurations. Raises an error if
        any other number of amplifiers is detected.
        """
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
    

    def set_kpf1_headers(self, l1_obj):
        """
        Populate KPF1 header keywords related to read noise measurement
        and overscan subtraction.

        Parameters
        ----------
        l1_obj : KPF1
            L1 data object whose PRIMARY header will be updated with
            read noise and overscan metadata.

        Returns
        -------
        None

        Notes
        -----
        Header updates:
        1. Read noise per amplifier channel (e.g. RNGRN1)
        2. Non-Gaussian read noise per amplifier channel (e.g. RNNGGR1)
        3. Overscan subtraction method (OSCANMET)
        """
        for channel_ext, rn in self.readnoise.items():
            key_read, key_rnng = _RN_KEYS[channel_ext]
            l1_obj.headers['PRIMARY'][key_read] = (
                round(float(rn), 4), f'Read noise {channel_ext} [e-]'
            )
            l1_obj.headers['PRIMARY'][key_rnng] = (
                round(float(rn), 4), f'Non-Gaussian read noise {channel_ext} [e-]'
            )

        l1_obj.headers['PRIMARY']['OSCANMET'] = (
            self.overscan_method, 'Overscan subtraction method'
        )


    def perform(self, chips=None, overscan_method=None):
        """
        Execute the image assembly algorithm. Optional keyword arguments
        default to config settings.

        Parameters
        ----------
        chips : list of str, optional
            CCD identifiers to process, i.e. 'GREEN', 'RED'
        overscan_method : str, optional
            Method for overscan subtraction ('zero', 'median', 'rowmedian').

        Returns
        -------
        l1_obj : KPF1
            L1 data object containing assembled full frame images (FFIs)
            for data and variance.

        Notes
        -----
        Pipeline steps:
        1. Count amplifiers and determine dimensions
        2. Orient amplifier channels
        3. Apply gain conversion (ADU --> electrons)
        4. Measure read noise
        5. Subtract overscan bias
        6. Re-orient channels if needed
        7. Stitch channels into a full-frame image
        """
        if chips is None:
            chips = self.chips
        if overscan_method is None:
            overscan_method = self.overscan_method

        self.chips = chips
        self.overscan_method = overscan_method

        l1_obj = self.l0_obj.to_kpf1()

        for chip in chips:
            self.count_amplifiers(chip)
            self.orient_channels(chip)
            self.apply_gain_conversion(chip)
            self.measure_read_noise(chip)
            self.subtract_overscan(chip, overscan_method)
            self.orient_channels(chip)

            ccd_ffi, var_ffi = self.stitch_ffi(chip)
            l1_obj.set_data(f'{chip}_CCD', ccd_ffi)
            l1_obj.set_data(f'{chip}_VAR', var_ffi)

        self.set_kpf1_headers(l1_obj)
        l1_obj.receipt_add_entry('image_assembly', 'PASS')

        return l1_obj