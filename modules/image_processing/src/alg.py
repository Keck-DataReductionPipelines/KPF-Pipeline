#packages
import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from modules.Utils.config_parser import ConfigHandler

# https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html
from astroscrappy import detect_cosmics


class ImageProcessingAlg():
    """
    Bias subtraction calculation.

    This module defines 'BiasSubtraction' and methods to perform bias
    subtraction by subtracting a master bias frame from the raw data frame.

    Attributes:
        rawimage (KPF0): From parameter 'rawimage'.
        ffi_exts (list): From parameter 'ffi_exts'.
        quicklook (bool): From parameter 'quicklook'.
        data_type (str): From parameter 'data_type'.
        config (configparser.ConfigParser, optional): From parameter 'config'.
        logger (logging.Logger, optional): From parameter 'logger'.

    Raises:
        Exception: If raw image and bias frame don't have the same dimensions.
    """

    def __init__(self, rawimage, ffi_exts, quicklook, data_type,
                 config=None, logger=None):

        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            rawimage (KPF0): The FITS raw data.

            ffi_exts (list): The extensions in L0 FITS files where FFIs (full
            frame images) are stored.

            quicklook (bool): If true, quicklook pipeline version of bias
            subtraction is run, outputting information and plots.

            data_type (str): Instrument name, currently choice between KPF and
            NEID.

            config (configparser.ConfigParser, optional): Config context.
            Defaults to None.

            logger (logging.Logger, optional): Instance of logging.Logger.
            Defaults to None.
        """
        self.rawimage=rawimage
        self.ffi_exts=ffi_exts
        self.quicklook=quicklook
        self.data_type=data_type
        self.config=config
        self.logger=logger
        cfg_params = ConfigHandler(config, 'PARAM')
        ins = cfg_params.get_config_value('instrument', '') if cfg_params is not None else ''
        self.config_ins = ConfigHandler(config, ins, cfg_params)

    def num_amps(self):
        amps = {'GREEN_CCD': 2, 'RED_CCD': 2}
        header = self.rawimage.header['PRIMARY']
        amps['RED_CCD'] = int(header['REDAMPS'])
        amps['GREEN_CCD'] = int(header['GRNAMPS'])

        return amps

    def bias_subtraction(self, masterbias):
        """Subtracts bias data from raw data.
        In pipeline terms: inputs are:
        1. Raw data in the form of a 2D FITS-image object (after OverscanSubtraction
           from modules.Utils.overscan_subtract has been applied to a L0 FITS-image object),
           produces one 2D FITS-image object; and
        2. Master bias data in the form of a 2D FITS-image object.

        Args:
            masterbias (FITS File): The master bias data.
        """
        header = self.rawimage.header['PRIMARY']

        # Write BIASFILE keyword to header regardless of whether bias is subtracted.
        # BIASDONE keyword will be set to 1 if bias is subtracted and 0 otherwise.

        header['BIASFILE'] = masterbias.filename
        header['BIASDIR'] = masterbias.dirname

        if header['IMTYPE'].lower() == 'bias':
            self.logger.info("Image is bias, skipping bias correction.")
            header['BIASDONE'] = 0
            return
        else:
            header['BIASDONE'] = 1

        for ffi in self.ffi_exts:
            try:
                self.rawimage[ffi] = self.rawimage[ffi] - masterbias[ffi]
            except Exception as e:
                if self.logger:
                    self.logger.info('*** Exception raised: {}'.format(e))
                else:
                    print("*** Exception raised:",e)

    def flat_division(self, flat_frame):
        """Performs flat frame division.
        In pipeline terms: inputs two L0 files, produces one L0 file.

        Args:
            flat_frame (FITS File): L0 FITS file object

        """
        header = self.rawimage.header['PRIMARY']

        # Write FLATFILE keyword to header regardless of whether flat is divided.
        # FLATDONE keyword will be set to 1 if flat is divided and 0 otherwise.

        header['FLATFILE'] = flat_frame.filename
        header['FLATDIR'] = flat_frame.dirname

        if header['IMTYPE'].lower() == 'bias' or \
            header['IMTYPE'].lower() == 'dark':
            self.logger.info("Image is {}, skipping flat correction.".format(header['IMTYPE']))
            header['FLATDONE'] = 0
            return
        else:
            header['FLATDONE'] = 1

        for ffi in self.ffi_exts:
            try:
                self.rawimage[ffi] = self.rawimage[ffi] / flat_frame[ffi]
            except Exception as e:
                if self.logger:
                    self.logger.info('*** Exception raised: {}'.format(e))
                else:
                    print("*** Exception raised:", e)


    def dark_subtraction(self, dark_frame):
        """Performs dark frame subtraction.
        In pipeline terms: inputs two L0 files, produces one L0 file.

        Args:
            dark_frame (FITS File): L0 FITS file object

        """
        header = self.rawimage.header['PRIMARY']

        image_exptime = float(self.rawimage.header['PRIMARY']['ELAPSED'])
        dark_exptime = 1.0   # master darks are already normalized

        # Write DARKFILE keyword to header regardless of whether dark is subtracted.
        # DARKDONE keyword will be set to 1 if dark is subtracted and 0 otherwise.

        header['DARKFILE'] = dark_frame.filename
        header['DARKDIR'] = dark_frame.dirname

        if header['IMTYPE'].lower() == 'bias' or \
            header['IMTYPE'].lower() == 'dark':
            self.logger.info("Image is {}, skipping dark correction.".format(header['IMTYPE']))
            header['DARKDONE'] = 0
            return
        else:
            header['DARKDONE'] = 1

        for ffi in self.ffi_exts:
            try:
                self.rawimage[ffi] = self.rawimage[ffi] - dark_frame[ffi]*(image_exptime/dark_exptime)
            except Exception as e:
                if self.logger:
                    self.logger.info('*** Exception raised: {}'.format(e))
                else:
                    print("*** Exception raised:", e)


    def cosmic_ray_masking(self, verbose=True):
        """Masks cosmic rays from input rawimage.
        """

        # Andrew quotes read noise of 3.5 electrons.
        # https://github.com/California-Planet-Search/KPF-Pipeline/issues/277
        # All of these parameters are fed into astroscrappy.
        cosmic_args={'sigclip':4, 'sigfrac':0.1, 'readnoise':3.5}

        for ffi in self.ffi_exts:

            # gain: preferably, these would be read from the appropriate FITS
            # headers.  for now, I am trying to make a MWE.
            gain = 5.

            # see astroscrappy docs: this pre-determined background image can
            # improve performance of the algorithm.
            inbkg = None

            # NOTE: I would love to get logger writing to work, but it does
            # not, probably for configuration reasons that I do not undersatnd.
            if verbose:
                N = np.sum(np.isnan(self.rawimage[ffi]))
                #self.logger.info(
                print(
                    f'Number NaNs before cosmic ray masking: {N}'
                )
                #)

            # detect cosmic rays in the resulting image
            recov_mask, clean_img = detect_cosmics(
                self.rawimage[ffi],
                inbkg=inbkg,
                gain=gain,
                readnoise=cosmic_args['readnoise'],
                sigclip=cosmic_args['sigclip'],
                sigfrac=cosmic_args['sigfrac']
            )

            # NaN-mask cosmic ray pixels.
            clean_img[recov_mask] = np.nan

            self.rawimage[ffi] = clean_img

            if verbose:
                N = np.sum(np.isnan(clean_img))
                #self.logger.info(
                print(
                    f'Number NaNs after CR masking: {N}'
                )
                #)

    def background_subtraction(self, order_masks):
        """ Background subtraction
        Args:
            order_masks(KPF0): order mask in Level 0 data format with fiber based extensions

        Returns:
            KPF0: KPF0 instance of raw image with background subtraction.
        """
        for ffi in self.ffi_exts:
            # no rawimage of order_mask for ffi extension
            if not hasattr(self.rawimage, ffi) or not self.rawimage[ffi].any() \
                    or not hasattr(order_masks, ffi) or not order_masks[ffi].any():
                continue

            raw = self.rawimage[ffi]
            clip = SigmaClip(sigma=3.)
            est = MedianBackground()
            bkg = np.zeros_like(raw)
            t_box = self.config_ins.get_config_value('BS_BOX', '(40, 28)')
            t_fs = self.config_ins.get_config_value('BS_FILTER', '(5, 5)')
            box = eval(t_box)  # box size for estimating background
            fs = eval(t_fs)    # window size for 2D low resolution median filtering
            mask_val = order_masks[ffi].astype(bool)
            if self.logger:
                self.logger.info(f"Background Subtraction box_size: "+ t_box + ' filter_size: '+t_fs)

            bkg[:, :] = Background2D(raw, box, mask=mask_val, filter_size=fs, sigma_clip=clip,
                                                bkg_estimator=est).background
            self.rawimage[ffi] = self.rawimage[ffi] - bkg

        return self.rawimage

    def bad_pixel_mask(self, mask):
        """Performs bad pixel masking
        In pipeline terms: inputs two 2D files, produces one 2D file.

        Args:
            rawfile (KPF0): KPF0 object of the raw L0 file (not 2D)
            mask (FITS File): L0 FITS file object with boolean values,
            One's where pixels are good and zeros where pixels are bad

        """
        header = self.rawimage.header['PRIMARY']
        if header['IMTYPE'].lower() != 'object':
            self.logger.info("Image is {}, skipping BPM correction.".format(header['IMTYPE']))
            return

        num_amps = self.num_amps()

        for ffi in self.ffi_exts:
            if 'amplifier' in mask.filename.lower() and \
                num_amps[ffi] < 4:
                continue
            mask[ffi][mask[ffi]==0] = np.nan
            try:
                self.rawimage[ffi] = self.rawimage[ffi] * mask[ffi]
                self.rawimage.header['PRIMARY']['PIXMASK'] = True
            except Exception as e:
                if self.logger:
                    self.logger.info('*** Exception raised: {}'.format(e))
                else:
                    print("*** Exception raised:", e)

    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The bias-corrected data.
        """
        return self.rawimage

#quicklook TODO: raise flag when counts are significantly diff from master bias, identify bad pixels
