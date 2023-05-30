from os.path import exists
import numpy as np
import configparser as cp
from datetime import datetime, timezone

from modules.Utils.kpf_fits import FitsHeaders
from modules.Utils.frame_stacker import FrameStacker

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/master_dark/configs/default.cfg'

#
# Documentation:
#
# Required inputs for generating a master-dark file are 2D L0 FITS files (under (/data/kpf/2D).
#
# Requirements for FITS-header keywords of inputs:
# 1. IMTYPE = 'Dark'
# 2. EXPTIME >= 300 seconds
#
# Assumptions and caveats:
# 1. Dark current is measured more reliably from longer exposures (e.g., >300 seconds).
# 2. Averaging many exposures increases chance of removing rad hits.
# 3. Currently makes master darks for GREEN_CCD, RED_CCD, and CA_HK.
#
# Algorithm:
# 1. Marshal inputs with above specifications for a given observation date.
# 2. Subtract master bias from inputs.
# 3. Separately normalize debiased images by EXPTIME.
# 4. Perform image-stacking with data-clipping at 2.1 sigma (aggressive to
#    eliminate rad hits and possible saturation).
# 5. Normalize dark by the image average.
# 6. Set appropriate infobit if number of pixels with less than 10 samples
#    is greater than 1% of total number of image pixels.
#
# Full-frame-image FITS extensions in output master dark
#
# EXTNAME = 'GREEN_CCD'          / GREEN dark-field corrections
# EXTNAME = 'RED_CCD '           / RED dark-field corrections
# EXTNAME = 'GREEN_CCD_UNC'      / GREEN dark-field uncertainties
# EXTNAME = 'GREEN_CCD_CNT'      / GREEN stack sample numbers (after data-clipping)
# EXTNAME = 'RED_CCD_UNC'        / RED dark-field uncertainties
# EXTNAME = 'RED_CCD_CNT'        / RED stack sample numbers (after data-clipping)
#

class MasterDarkFramework(KPF0_Primitive):

    """
    Description:
        This class works within the Keck pipeline framework to compute the master dark
        by stacking input images for exposures with IMTYPE.lower() == 'dark'
        and specified minimum EXPTIME, selected from the given path that can include
        many kinds of FITS files, not just darks.
        Subtract master bias image from each input dark 2D raw image.
        Separately normalize debiased images by EXPTIME.
        Stack all normalized debiased images.
        Set appropriate infobit if number of pixels with less than 10 samples
        is greater than 1% of total number of image pixels.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_master_bias.fits).
        masterdark_path (str): Pathname of output master dark (e.g., /testdata/kpf_master_dark.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_master_bias.fits).
        masterdark_path (str): Pathname of output master dark (e.g., /testdata/kpf_master_dark.fits).
        imtype_keywords (str): FITS keyword for filtering input dark files (fixed as 'IMTYPE').
        imtype_values_str (str): Value of FITS keyword (fixed as 'Dark'), to be converted to lowercase for test.
        module_config_path (str): Location of default config file (modules/master_dark/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
        exptime_minimum (float): Minimum EXPTIME of darks to use in computing master dark (default = 300.0 seconds)

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.n_sigma = self.action.args[1]
        self.all_fits_files_path = self.action.args[2]
        self.lev0_ffi_exts = self.action.args[3]
        self.masterbias_path = self.action.args[4]
        self.masterdark_path = self.action.args[5]

        self.imtype_keywords = 'IMTYPE'       # Unlikely to be changed.
        self.imtype_values_str = 'Dark'

        try:
            self.module_config_path = context.config_path['master_dark']
            print("--->MasterDarkFramework class: self.module_config_path =",self.module_config_path)
        except:
            self.module_config_path = DEFAULT_CFG_PATH

        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.module_config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        module_param_cfg = module_config_obj['PARAM']

        self.exptime_minimum = float(module_param_cfg.get('exptime_minimum', 300.0))

        self.logger.info('self.exptime_minimum = {}'.format(self.exptime_minimum))

    def _perform(self):

        """
        Returns [exitcode, infobits] after computing and writing master-dark FITS file.

        """

        masterbias_path_exists = exists(self.masterbias_path)
        if not masterbias_path_exists:
            raise FileNotFoundError('File does not exist: {}'.format(self.masterbias_path))
        self.logger.info('self.masterbias_path = {}'.format(self.masterbias_path))
        self.logger.info('masterbias_path_exists = {}'.format(masterbias_path_exists))

        master_bias_data = KPF0.from_fits(self.masterbias_path,self.data_type)
        
        master_dark_exit_code = 0
        master_dark_infobits = 0

        # Filter dark files with IMTYPE=‘Dark’ and the specified exposure time.

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_dark_files = fh.get_good_darks(self.exptime_minimum)

        mjd_obs_list = []
        exp_time_list = []
        for dark_file_path in (all_dark_files):
            dark_file = KPF0.from_fits(dark_file_path,self.data_type)
            mjd_obs = float(dark_file.header['PRIMARY']['MJD-OBS'])
            mjd_obs_list.append(mjd_obs)
            exp_time = float(dark_file.header['PRIMARY']['EXPTIME'])
            exp_time_list.append(exp_time)
            self.logger.debug('dark_file_path,exp_time = {},{}'.format(dark_file_path,exp_time))

        tester = KPF0.from_fits(all_dark_files[0])
        del_ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'CA_HK' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                del_ext_list.append(i)
        master_holder = tester

        for ffi in self.lev0_ffi_exts:

            self.logger.debug('Loading dark data, ffi = {}'.format(ffi))
            keep_ffi = 0

            frames_data=[]
            for path in all_dark_files:
                obj = KPF0.from_fits(path)
                np_obj_ffi = np.array(obj[ffi])
                np_obj_ffi_shape = np.shape(np_obj_ffi)
                n_dims = len(np_obj_ffi_shape)
                self.logger.debug('path,ffi,n_dims = {},{},{}'.format(path,ffi,n_dims))
                if n_dims == 2:       # Check if valid data extension
                     keep_ffi = 1
                     frames_data.append(obj[ffi])

            if keep_ffi == 0:
                self.logger.debug('ffi,keep_ffi = {},{}'.format(ffi,keep_ffi))
                del_ext_list.append(ffi)
                break

            frames_data = np.array(frames_data) - np.array(master_bias_data[ffi])      # Subtract master bias.

            self.logger.debug('Subtracting master bias from dark data...')

            normalized_frames_data=[]
            n_frames = (np.shape(frames_data))[0]
            self.logger.debug('Number of frames in stack = {}'.format(n_frames))
            for i in range(0, n_frames):
                single_frame_data = frames_data[i]

                # Separately normalize debiased images by EXPTIME.

                exp_time = exp_time_list[i]
                self.logger.debug('Normalizing dark image: i,fitsfile,ffi,exp_time = {},{},{},{}'.format(i,all_dark_files[i],ffi,exp_time))
                single_normalized_frame_data = single_frame_data / exp_time
                normalized_frames_data.append(single_normalized_frame_data)

            #
            # Stack the frames.
            #

            normalized_frames_data = np.array(normalized_frames_data)

            fs = FrameStacker(normalized_frames_data,self.n_sigma,self.logger)
            stack_avg,stack_var,cnt,stack_unc = fs.compute()

            # Already normalized by exposure time.
            dark = stack_avg
            dark_unc = stack_unc

            ### kpf master file creation ###
            master_holder[ffi] = dark

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.create_extension(ffi_unc_ext_name,ext_type=np.array)
            master_holder[ffi_unc_ext_name] = dark_unc.astype(np.float32)

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.create_extension(ffi_cnt_ext_name,ext_type=np.array)
            master_holder[ffi_cnt_ext_name] = cnt.astype(np.int32)

            n_samples_lt_10 = (cnt < 10).sum()
            rows = np.shape(master_holder[ffi])[0]
            cols = np.shape(master_holder[ffi])[1]
            n_pixels = rows * cols
            pcent_diff = 100 * n_samples_lt_10 / n_pixels

            # Set appropriate infobit if number of pixels with less than 10 samples in
            # current FITS extension is greater than 1% of total number of pixels in image.

            if pcent_diff > 1.0:
                self.logger.info('ffi,n_samples_lt_10 = {},{}'.format(ffi,n_samples_lt_10))
                if "GREEN_CCD" in (ffi).upper():
                   master_dark_infobits |= 2**0
                elif "RED_CCD" in (ffi).upper():
                   master_dark_infobits |= 2**1
                elif "CA_HK" in (ffi).upper():
                   master_dark_infobits |= 2**2

        for ext in del_ext_list:
            master_holder.del_extension(ext)

        # Add informational keywords to FITS header.

        master_holder.header['PRIMARY']['IMTYPE'] = ('Dark','Master dark')

        for ffi in self.lev0_ffi_exts:
            if ffi in del_ext_list: continue
            master_holder.header[ffi]['BUNIT'] = ('DN/sec','Units of master dark')
            master_holder.header[ffi]['NFRAMES'] = (len(all_dark_files),'Number of frames in input stack')
            master_holder.header[ffi]['MINEXPTM'] = (self.exptime_minimum,'Minimum exposure time of input darks (seconds)')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (min(mjd_obs_list),'Minimum MJD of dark observations')
            master_holder.header[ffi]['MAXMJD'] = (max(mjd_obs_list),'Maximum MJD of dark observations')
            datetimenow = datetime.now(timezone.utc)
            createdutc = datetimenow.strftime("%Y-%m-%dT%H:%M:%SZ")
            master_holder.header[ffi]['CREATED'] = (createdutc,'UTC of master-dark creation')
            master_holder.header[ffi]['INFOBITS'] = (master_dark_infobits,'Bit-wise flags defined below')

            master_holder.header[ffi]['BIT00'] = ('2**0 = 1', 'GREEN_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT01'] = ('2**1 = 2', 'RED_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT02'] = ('2**2 = 4', 'CA_HK" has gt 1% pixels with lt 10 samples')

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.header[ffi_unc_ext_name]['BUNIT'] = ('DN/sec','Units of master-dark uncertainty')

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.header[ffi_cnt_ext_name]['BUNIT'] = ('Count','Number of stack samples')

        master_holder.to_fits(self.masterdark_path)

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_dark_exit_code,master_dark_infobits]

        return Arguments(exit_list)
