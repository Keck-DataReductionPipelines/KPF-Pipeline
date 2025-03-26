from os.path import exists
from os import getenv
import numpy as np
import numpy.ma as ma
import configparser as cp
from datetime import datetime, timezone
from scipy.ndimage import gaussian_filter
from scipy.stats import mode
from astropy.io import fits
from astropy.time import Time
import re
import pandas as pd

import database.modules.utils.kpf_db as db
import modules.quality_control.src.quality_control as qc
from modules.Utils.kpf_fits import FitsHeaders
from modules.Utils.frame_stacker import FrameStacker

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/master_flat/configs/default.cfg'


class MasterFlatFramework(KPF0_Primitive):

    """
    Description:
        This class works within the Keck pipeline framework to compute the master flat
        by stacking input images for exposures with IMTYPE.lower() == 'flatlamp'
        (and other selection criteria), selected from the given path that can include
        many kinds of FITS files, not just flats.

        Requirements for FITS-header keywords of inputs:
        1. IMTYPE = 'Flatlamp'
        2. OBJECT = 'autocal-flat-all'
        3. EXPTIME < maximum allowed time (these are default.cfg parameters for GREEN, RED, and CA_HK).

        Assumptions and caveats:
        1. Does not include correcting for the color of the lamp, and other subtleties
           specific to spectral data.
        2. Currently "master" flat-lamp pattern made "on the fly" by
           2-D Gaussian blurring (sigma=2 pixel) the stacked-image mean.
        3. Further modifications to this recipe are needed in order to use
           a master flat-lamp pattern from a prior night.
        4. Low-light pixels cannot be reliably used to
           compute the flat-field correction (e.g., less than 5 electrons/sec).
        5. Currently makes master flats for GREEN_CCD, RED_CCD, and CA_HK.

        Algorithm:
        Marshal inputs with above specifications for a given observation date.
        Subtract master bias and master dark from each input flat 2D raw image.
        Separately normalize debiased images by EXPTIME.
        Perform image-stacking with data-clipping at 2.1 sigma (aggressive to
        eliminate rad hits and possible saturation).
        Divide clipped mean of stack by the smoothed Flatlamp pattern.
        Normalize flat by the image average.
        Reset flat values to unity if corresponding stacked-image value
        is less than low-light limit (insufficient illumination) or are
        outside of the order mask (if available for the current FITS extension).
        Set appropriate infobit if number of pixels with less than 10 samples
        is greater than 1% of total number of image pixels.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_master_bias.fits).
        masterdark_path (str): Pathname of input master dark (e.g., /testdata/kpf_master_dark.fits).
        masterflat_path (str): Pathname of output master flat (e.g., /testdata/kpf_master_flat.fits).
        smoothlamppattern_path (str): Pathname of input smooth lamp pattern (e.g., /data/reference_fits/kpf_20230628_smooth_lamp_made20230720_float32.fits).
        ordermask_path (str): Pathname of input order mask (e.g., /data/reference_fits/order_mask_G4-3_2_R4-3_2_20230717.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_green_red_bias.fits).
        masterdark_path (str): Pathname of input master dark (e.g., /testdata/kpf_master_dark.fits).
        masterflat_path (str): Pathname of output master flat (e.g., /testdata/kpf_green_red_flat.fits).
        smoothlamppattern_path (str): Pathname of input smooth lamp pattern (e.g., /data/reference_fits/kpf_20230628_smooth_lamp_made20230720_float32.fits).
        ordermask_path (str): Pathname of input order mask (e.g., /data/reference_fits/order_mask_G4-3_2_R4-3_2_20230717.fits).
        imtype_keywords (str): FITS keyword for filtering input flat files (fixed as 'IMTYPE').
        imtype_values_str (str): Value of FITS keyword (fixed as 'Flatlamp'), to be converted to lowercase for test.
        module_config_path (str): Location of default config file (modules/master_flat/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
        gaussian_filter_sigma (float): 2-D Gaussian-blur sigma for smooth lamp pattern calculation (default = 2.0 pixels)
        low_light_limit = Low-light limit where flat is set to unity (default = 5.0 electrons/sec)

    Outputs:
        Full-frame-image FITS extensions in output master flat:
        EXTNAME = 'GREEN_CCD'          / GREEN flat-field corrections
        EXTNAME = 'RED_CCD '           / RED flat-field corrections
        EXTNAME = 'GREEN_CCD_UNC'      / GREEN flat-field uncertainties
        EXTNAME = 'GREEN_CCD_CNT'      / GREEN stack sample numbers (after data-clipping)
        EXTNAME = 'GREEN_CCD_STACK'    / GREEN stacked-image averages
        EXTNAME = 'GREEN_CCD_LAMP'     / GREEN smooth flat-lamp pattern
        EXTNAME = 'RED_CCD_UNC'        / RED flat-field uncertainties
        EXTNAME = 'RED_CCD_CNT'        / RED stack sample numbers (after data-clipping)
        EXTNAME = 'RED_CCD_STACK'      / RED stacked-image averages
        EXTNAME = 'RED_CCD_LAMP'       / RED smooth flat-lamp pattern

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.n_sigma = self.action.args[1]
        self.all_fits_files_path = self.action.args[2]
        self.lev0_ffi_exts = self.action.args[3]
        self.masterbias_path = self.action.args[4]
        self.masterdark_path = self.action.args[5]
        self.masterflat_path = self.action.args[6]
        self.smoothlamppattern_path = self.action.args[7]
        self.ordermask_path = self.action.args[8]
        self.flat_object = self.action.args[9]

        self.imtype_keywords = ['IMTYPE','OBJECT']       # Unlikely to be changed.
        #self.imtype_values_str = ['Flatlamp','autocal-flat-all']
        #self.imtype_values_str = ['Flatlamp','test-flat-all']
        self.imtype_values_str = ['Flatlamp',self.flat_object]

        try:
            self.module_config_path = context.config_path['master_flat']
            print("--->MasterFlatFramework class: self.module_config_path =",self.module_config_path)
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

        self.gaussian_filter_sigma = float(module_param_cfg.get('gaussian_filter_sigma', 2.0))
        self.low_light_limit = float(module_param_cfg.get('low_light_limit', 5.0))
        self.green_ccd_flat_exptime_maximum = float(module_param_cfg.get('green_ccd_flat_exptime_maximum', 2.0))
        self.red_ccd_flat_exptime_maximum = float(module_param_cfg.get('red_ccd_flat_exptime_maximum', 1.0))
        self.ca_hk_flat_exptime_maximum = float(module_param_cfg.get('ca_hk_flat_exptime_maximum', 1.0))

        self.logger.info('self.gaussian_filter_sigma = {}'.format(self.gaussian_filter_sigma))
        self.logger.info('self.low_light_limit = {}'.format(self.low_light_limit))
        self.logger.info('self.green_ccd_flat_exptime_maximum = {}'.format(self.green_ccd_flat_exptime_maximum))
        self.logger.info('self.red_ccd_flat_exptime_maximum = {}'.format(self.red_ccd_flat_exptime_maximum))
        self.logger.info('self.ca_hk_flat_exptime_maximum = {}'.format(self.ca_hk_flat_exptime_maximum))

    def _perform(self):

        """
        Returns [exitcode, infobits] after computing and writing master-flat FITS file.

        """


        # Initialization.

        master_flat_exit_code = 0
        master_flat_infobits = 0
        input_master_type = 'Flat'


        # Filter flat files with IMTYPE=‘flatlamp’ and that match the input object specification with OBJECT.
        # Parse obsdate

        self.logger.info('self.flat_object = {}'.format(self.flat_object))

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_flat_files = fh.match_headers_string_lower()
        n_all_flat_files = len(all_flat_files)

        if n_all_flat_files == 0:
            self.logger.info('n_all_flat_files = {}'.format(n_all_flat_files))
            master_flat_exit_code = 8
            exit_list = [master_flat_exit_code,master_flat_infobits]
            return Arguments(exit_list)

        obsdate_match = re.match(r".*(\d\d\d\d\d\d\d\d).*", all_flat_files[0])
        try:
            obsdate = obsdate_match.group(1)
            self.logger.info('obsdate = {}'.format(obsdate))
        except:
            self.logger.info("obsdate not parsed from first input filename")
            obsdate = None    # This should never happen
            self.logger.info('*** Warning: Observation date not available from first input flat frame; returning...')
            master_flat_exit_code = 10
            exit_list = [master_flat_exit_code,master_flat_infobits]
            return Arguments(exit_list)


        # Optionally override self.smoothlamppattern_path from input argument with environment-variable setting.

        smoothlamppattern_envar = getenv('SMOOTH_LAMP_PATTERN')
        if smoothlamppattern_envar is not None:
            self.logger.info('Override smoothlamppattern_path with SMOOTH_LAMP_PATTERN setting...')
            self.smoothlamppattern_path = smoothlamppattern_envar

        smoothlamppattern_path_exists = exists(self.smoothlamppattern_path)
        if not smoothlamppattern_path_exists:
            raise FileNotFoundError('File does not exist: {}'.format(self.smoothlamppattern_path))
        self.logger.info('self.smoothlamppattern_path = {}'.format(self.smoothlamppattern_path))
        self.logger.info('smoothlamppattern_path_exists = {}'.format(smoothlamppattern_path_exists))

        smooth_lamp_pattern_data = KPF0.from_fits(self.smoothlamppattern_path,self.data_type)

        ordermask_path_exists = exists(self.ordermask_path)
        if not ordermask_path_exists:
            raise FileNotFoundError('File does not exist: {}'.format(self.ordermask_path))
        self.logger.info('self.ordermask_path = {}'.format(self.ordermask_path))
        self.logger.info('ordermask_path_exists = {}'.format(ordermask_path_exists))

        order_mask_data = KPF0.from_fits(self.ordermask_path,self.data_type)
        self.logger.debug('Finished loading order-mask data from FITS file = {}'.format(self.ordermask_path))


        # Get required bias and dark master calibration files from database, if those provided by input arguments to not exist.
        # This code section is needed because this cannot be easily handled via module/calibration_lookup in kpf_masters_drp.recipe.
        # This code section is only necessary in the rare case where no daily bias and/or dark data are available.

        dbh = db.KPFDB()             # Open database connection (if needed for fallback master calibration file)

        cal_file_level = 0           # Parameters for querying database fallback master calibration file.
        contentbitmask = 3

        masterbias_path_exists = exists(self.masterbias_path)

        self.logger.info('masterbias_path_exists = {}'.format(masterbias_path_exists))

        if not masterbias_path_exists:
            if obsdate != None:
                cal_type_pair = ['bias','autocal-bias']                    # Query database for fallback master bias.
                dbh.get_nearest_master_file(obsdate,cal_file_level,contentbitmask,cal_type_pair)
                self.logger.info('database-query exit_code = {}'.format(dbh.exit_code))
                self.logger.info('Master dark database-query filename = {}'.format(dbh.filename))
                if dbh.exit_code == 0:
                    self.masterbias_path = dbh.filename
                else:
                     self.logger.info('Master bias file cannot be queried from database; returning...')
                     master_flat_exit_code = 5
                     exit_list = [master_flat_exit_code,master_flat_infobits]
                     return Arguments(exit_list)
            else:
                self.logger.info('Observation date not available so master bias file cannot be queried from database; returning...')
                master_flat_exit_code = 10
                exit_list = [master_flat_exit_code,master_flat_infobits]
                return Arguments(exit_list)

        self.logger.info('self.masterbias_path = {}'.format(self.masterbias_path))

        masterdark_path_exists = exists(self.masterdark_path)
        self.logger.info('masterdark_path_exists = {}'.format(masterdark_path_exists))

        if not masterdark_path_exists:
            if obsdate != None:
                cal_type_pair = ['dark','autocal-dark']                    # Query database for fallback master dark.
                dbh.get_nearest_master_file(obsdate,cal_file_level,contentbitmask,cal_type_pair)
                self.logger.info('database-query exit_code = {}'.format(dbh.exit_code))
                self.logger.info('Master dark database-query filename = {}'.format(dbh.filename))
                if dbh.exit_code == 0:
                     self.masterdark_path = dbh.filename
                else:
                     self.logger.info('Master dark file cannot be queried from database; returning...')
                     master_flat_exit_code = 5
                     exit_list = [master_flat_exit_code,master_flat_infobits]
                     return Arguments(exit_list)
            else:
                self.logger.info('Observation date not available so master dark file cannot be queried from database; returning...')
                master_flat_exit_code = 10
                exit_list = [master_flat_exit_code,master_flat_infobits]
                return Arguments(exit_list)

        self.logger.info('self.masterdark_path = {}'.format(self.masterdark_path))

        dbh.close()      # Close database connection.

        master_bias_data = KPF0.from_fits(self.masterbias_path,self.data_type)
        master_dark_data = KPF0.from_fits(self.masterdark_path,self.data_type)

        mjd_obs_list = []
        exp_time_list = []
        for flat_file_path in (all_flat_files):
            flat_file = KPF0.from_fits(flat_file_path,self.data_type)
            mjd_obs = float(flat_file.header['PRIMARY']['MJD-OBS'])
            mjd_obs_list.append(mjd_obs)
            exp_time = float(flat_file.header['PRIMARY']['EXPTIME'])
            exp_time_list.append(exp_time)
            self.logger.debug('flat_file_path,exp_time = {},{}'.format(flat_file_path,exp_time))


        # Ensure prototype FITS header for product file has matching OBJECT and contains both
        # GRNAMPS and REDAMPS keywords (indicating that the data exist).

        for flat_file_path in (all_flat_files):

            tester = KPF0.from_fits(flat_file_path)
            tester_object = tester.header['PRIMARY']['OBJECT']

            if tester_object == self.flat_object:

                try:
                    tester_grnamps = tester.header['PRIMARY']['GRNAMPS']
                except KeyError as err:
                    continue

                try:
                    tester_redamps = tester.header['PRIMARY']['REDAMPS']
                except KeyError as err:
                    continue

                self.logger.info('Prototype FITS header from {}'.format(flat_file_path))

                date_obs = tester.header['PRIMARY']['DATE-OBS']

                break

            else:

                tester = None

        if tester is None:
            master_flat_exit_code = 6
            exit_list = [master_flat_exit_code,master_flat_infobits]
            return Arguments(exit_list)


        del_ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'CA_HK' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                del_ext_list.append(i)
        master_holder = tester

        filenames_kept = {}
        n_frames_kept = {}
        mjd_obs_min = {}
        mjd_obs_max = {}
        for ffi in self.lev0_ffi_exts:

            self.logger.debug('Loading flat data, ffi = {}'.format(ffi))
            keep_ffi = 0

            filenames_kept_list = []
            frames_data = []
            frames_data_exptimes = []
            frames_data_mjdobs = []
            frames_data_path = []
            n_all_flat_files = len(all_flat_files)
            for i in range(0, n_all_flat_files):

                exp_time = exp_time_list[i]
                mjd_obs = mjd_obs_list[i]
                self.logger.debug('i,fitsfile,ffi,exp_time = {},{},{},{}'.format(i,all_flat_files[i],ffi,exp_time))

                if not (ffi == 'GREEN_CCD' or ffi == 'RED_CCD' or ffi == 'CA_HK'):
                    raise NameError('FITS extension {} not supported; check recipe config file.'.format(ffi))

                if ffi == 'GREEN_CCD' and exp_time > self.green_ccd_flat_exptime_maximum:
                    self.logger.debug('---->ffi,exp_time,self.green_ccd_flat_exptime_maximum = {},{},{}'.format(ffi,exp_time,self.green_ccd_flat_exptime_maximum))
                    continue
                if ffi == 'RED_CCD' and exp_time > self.red_ccd_flat_exptime_maximum:
                    self.logger.debug('---->ffi,exp_time,self.red_ccd_flat_exptime_maximum = {},{},{}'.format(ffi,exp_time,self.red_ccd_flat_exptime_maximum))
                    continue
                if ffi == 'CA_HK' and exp_time > self.ca_hk_flat_exptime_maximum:
                    continue

                path = all_flat_files[i]
                obj = KPF0.from_fits(path)


                # Check QC keywords and skip image if it does not pass QC checking.

                skip = qc.check_all_qc_keywords(obj,path,input_master_type,self.logger)
                self.logger.debug('After calling qc.check_all_qc_keywords: i,path,skip = {},{},{}'.format(i,path,skip))
                if skip:
                    continue


                np_obj_ffi = np.array(obj[ffi])
                np_obj_ffi_shape = np.shape(np_obj_ffi)
                n_dims = len(np_obj_ffi_shape)
                self.logger.debug('path,ffi,n_dims = {},{},{}'.format(path,ffi,n_dims))
                if n_dims == 2:       # Check if valid data extension
                     keep_ffi = 1
                     filenames_kept_list.append(all_flat_files[i])
                     frames_data.append(obj[ffi])
                     frames_data_exptimes.append(exp_time)
                     frames_data_mjdobs.append(mjd_obs)
                     frames_data_path.append(path)
                     self.logger.debug('Keeping flat image: i,fitsfile,ffi,mjd_obs,exp_time = {},{},{},{},{}'.format(i,all_flat_files[i],ffi,mjd_obs,exp_time))

            np_frames_data = np.array(frames_data)
            np_bias_data = np.array(master_bias_data[ffi])

            self.logger.debug('ffi,np.shape(np_frames_data),np.shape(np_bias_data) = {},{},{}'.format(ffi,np.shape(np_frames_data),np.shape(np_bias_data)))

            if len(np.shape(np_bias_data)) != 2:
                self.logger.debug('Master bias missing for ffi = {}'.format(ffi))
                keep_ffi = 0

            if keep_ffi == 0:
                self.logger.debug('ffi,keep_ffi = {},{}'.format(ffi,keep_ffi))
                del_ext_list.append(ffi)
                continue

            frames_data = np_frames_data - np_bias_data      # Subtract master bias.

            self.logger.debug('Subtracted master bias from flat data...')

            normalized_frames_data=[]
            n_frames = (np.shape(frames_data))[0]
            self.logger.debug('Number of frames in stack = {}'.format(n_frames))

            # Skip extension if number of frames to stack is less than 2.

            if n_frames < 2:
                self.logger.debug('n_frames < 2 for ffi,n_frames = {},{}'.format(ffi,n_frames))
                del_ext_list.append(ffi)
                continue

            filenames_kept[ffi] = filenames_kept_list
            n_frames_kept[ffi] = n_frames
            mjd_obs_min[ffi] = min(frames_data_mjdobs)
            mjd_obs_max[ffi] = max(frames_data_mjdobs)

            for i in range(0, n_frames):
                single_frame_data = frames_data[i]
                exp_time = frames_data_exptimes[i]

                self.logger.debug('Normalizing flat image: i,fitsfile,ffi,exp_time = {},{},{},{}'.format(i,frames_data_path[i],ffi,exp_time))

                single_normalized_frame_data = single_frame_data / exp_time       # Separately normalize by EXPTIME.

                # Sometimes the CA_HK dark is empty.
                try:
                    single_normalized_frame_data -= np.array(master_dark_data[ffi])   # Subtract master-dark-current rate.
                except:
                    self.logger.debug('Could not subtract dark: np.shape(np.array(master_dark_data[ffi])) = {},{},{}'.format(i,ffi,np.shape(np.array(master_dark_data[ffi]))))

                normalized_frames_data.append(single_normalized_frame_data)

            #
            # Stack the frames.
            #

            normalized_frames_data = np.array(normalized_frames_data)

            fs = FrameStacker(normalized_frames_data,self.n_sigma,self.logger)
            stack_avg,stack_var,cnt,stack_unc = fs.compute()

            # Divide by the smoothed Flatlamp pattern.  For GREEN_CCD and RED_CCD, use a fixed lamp pattern
            # to "flatten" of all stacked-image data for the current observation date within the orderlet mask.
            # The fixed lamp pattern is made from a stacked image from a specific observation date
            # (e.g., 100 Flatlamp frames, 30-second exposures each, were acquired on 20230628). The fixed lamp
            # pattern is smoothed with a sliding-window kernel 200-pixels wide (along dispersion dimension)
            # by 1-pixel high (along cross-dispersion dimension) by computing the clipped mean
            # with 3-sigma double-sided outlier rejection.   The fixed smooth lamp pattern enables the flat-field
            # correction to remove dust and debris signatures on the optics of the instrument and telescope.
            # The local median filtering smooths, yet minimizes undesirable effects at the orderlet edges.
            # For CA_HK, use dynmaic 2-D Gaussian blurring with width sigma to remove the large scale structure
            # in the flats (as a stop-gap method).

            if (ffi == 'GREEN_CCD' or ffi == 'RED_CCD'):
                smooth_lamp_pattern = np.array(smooth_lamp_pattern_data[ffi])
            else:
                smooth_lamp_pattern = gaussian_filter(stack_avg, sigma=self.gaussian_filter_sigma)

            unnormalized_flat = stack_avg / smooth_lamp_pattern
            unnormalized_flat_unc = stack_unc / smooth_lamp_pattern


            # Apply order mask, if available for the current FITS extension.  Otherwise, use the low-light pixels as a mask.

            np_om_ffi = np.array(np.rint(order_mask_data[ffi])).astype(int)   # Ensure rounding to nearest integer.
            np_om_ffi_shape = np.shape(np_om_ffi)
            order_mask_n_dims = len(np_om_ffi_shape)
            self.logger.debug('ffi,order_mask_n_dims = {},{}'.format(ffi,order_mask_n_dims))
            if order_mask_n_dims == 2:      # Check if valid data extension

                # Loop over 5 orderlets in the KPF instrument and normalize separately for each.
                flat = unnormalized_flat
                flat_unc = unnormalized_flat_unc
                for orderlet_val in range(1,6):         # Order mask has them numbered from 1 to 5 (bottom to top).
                    np_om_ffi_bool = np.where(np_om_ffi == orderlet_val,True,False)
                    np_om_ffi_bool = np.where(stack_avg > self.low_light_limit,np_om_ffi_bool,False)

                    # Compute mean for comparison with mode of distribution.
                    unmx = ma.masked_array(unnormalized_flat, mask = ~ np_om_ffi_bool)  # Invert mask for numpy.ma operation.
                    unnormalized_flat_mean = ma.getdata(unmx.mean()).item()

                    # Compute mode of distribution for normalization factor.
                    vals_for_mode_calc = np.where(np_om_ffi == orderlet_val,np.rint(100.0 * unnormalized_flat),np.nan)
                    vals_for_mode_calc = np.where(stack_avg > self.low_light_limit,vals_for_mode_calc,np.nan)
                    mode_vals,mode_counts = mode(vals_for_mode_calc,axis=None,nan_policy='omit')

                    dump_data_str = getenv('DUMP_MASTER_FLAT_DATA')
                    if dump_data_str is None:
                        dump_data_str = "0"
                    dump_data = int(dump_data_str)
                    if dump_data == 1:
                        fname = 'vals_for_mode_' + ffi + '_orderlet' + str(orderlet_val) + '.txt'
                        np.savetxt(fname, vals_for_mode_calc.flatten(), fmt = '%10.5f', newline = '\n', header = 'value')


                    self.logger.debug('type(mode_vals),type(mode_counts) = {},{}'.format(type(mode_vals),type(mode_counts)))

                    # Try if mod_vals is returned as array; upon failure, try if mod_vals is returned as just one value.

                    try:
                        normalization_factor = mode_vals[0] / 100.0      # Divide by 100 to account for above binning.
                    except:
                        try:
                            normalization_factor = mode_vals / 100.0      # Divide by 100 to account for above binning.
                        except:
                            normalization_factor = unnormalized_flat_mean

                    self.logger.debug('orderlet_val,unnormalized_flat_mean,normalization_factor,mode_vals,mode_counts = {},{},{},{},{}'.\
                        format(orderlet_val,unnormalized_flat_mean,normalization_factor,mode_vals,mode_counts))


                    flat = np.where(np_om_ffi_bool == True, flat / normalization_factor, flat)
                    flat_unc = np.where(np_om_ffi_bool == True, flat_unc / normalization_factor, flat_unc)

                # Set unity flat values for unmasked pixels.
                np_om_ffi_bool_all_orderlets = np.where(np_om_ffi > 0.5, True, False)
                flat = np.where(np_om_ffi_bool_all_orderlets == False, 1.0, flat)

                # Less than low-light pixels cannot be reliably adjusted.  Reset below-threshold pixels to have unity flat values.
                flat = np.where(stack_avg < self.low_light_limit, 1.0, flat)

            else:
                np_om_ffi_bool = np.where(stack_avg > self.low_light_limit, True, False)
                np_om_ffi_shape = np.shape(np_om_ffi_bool)
                self.logger.debug('np_om_ffi_shape = {}'.format(np_om_ffi_shape))

                # Compute mean of unmasked pixels in unnormalized flat.
                unmx = ma.masked_array(unnormalized_flat, mask = ~ np_om_ffi_bool)    # Invert the mask for mask_array operation.
                unnormalized_flat_mean = ma.getdata(unmx.mean()).item()
                self.logger.debug('unnormalized_flat_mean = {}'.format(unnormalized_flat_mean))

                # Normalize flat.
                flat = unnormalized_flat / unnormalized_flat_mean                     # Normalize the master flat by the mean.
                flat_unc = unnormalized_flat_unc / unnormalized_flat_mean             # Normalize the uncertainties.

                # Less than low-light pixels cannot be reliably adjusted.  Reset below-threshold pixels to have unity flat values.
                flat = np.where(stack_avg < self.low_light_limit, 1.0, flat)

            # Reset flat to unity if flat < 0.1 or flat > 2.5.
            flat = np.where(flat < 0.1, 1.0, flat)
            flat = np.where(flat > 2.5, 1.0, flat)

            ### kpf master file creation ###
            master_holder[ffi] = flat

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.create_extension(ffi_unc_ext_name,ext_type=np.array)
            master_holder[ffi_unc_ext_name] = flat_unc.astype(np.float32)

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.create_extension(ffi_cnt_ext_name,ext_type=np.array)
            master_holder[ffi_cnt_ext_name] = cnt.astype(np.int32)

            ffi_stack_ext_name = ffi + '_STACK'
            master_holder.create_extension(ffi_stack_ext_name,ext_type=np.array)
            master_holder[ffi_stack_ext_name] = stack_avg.astype(np.float32)

            ffi_lamp_ext_name = ffi + '_LAMP'
            master_holder.create_extension(ffi_lamp_ext_name,ext_type=np.array)
            master_holder[ffi_lamp_ext_name] = smooth_lamp_pattern.astype(np.float32)

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
                   master_flat_infobits |= 2**0
                elif "RED_CCD" in (ffi).upper():
                   master_flat_infobits |= 2**1
                elif "CA_HK" in (ffi).upper():
                   master_flat_infobits |= 2**2

        for ext in del_ext_list:
            master_holder.del_extension(ext)

        # Remove confusing or non-relevant keywords, if existing.

        try:
            del master_holder.header['GREEN_CCD']['OSCANV1']
            del master_holder.header['GREEN_CCD']['OSCANV2']
            del master_holder.header['GREEN_CCD']['OSCANV3']
            del master_holder.header['GREEN_CCD']['OSCANV4']
        except KeyError as err:
            pass

        try:
            del master_holder.header['RED_CCD']['OSCANV1']
            del master_holder.header['RED_CCD']['OSCANV2']
            del master_holder.header['RED_CCD']['OSCANV3']
            del master_holder.header['RED_CCD']['OSCANV4']
        except KeyError as err:
            pass

        for ffi in self.lev0_ffi_exts:
            if ffi in del_ext_list: continue
            master_holder.header[ffi]['BUNIT'] = ('Dimensionless','Units of master flat')
            master_holder.header[ffi]['NFRAMES'] = (n_frames_kept[ffi],'Number of frames in input stack')
            master_holder.header[ffi]['GAUSSSIG'] = (self.gaussian_filter_sigma,'2-D Gaussian-smoother sigma (pixels)')
            master_holder.header[ffi]['LOWLTLIM'] = (self.low_light_limit,'Low-light limit (electrons)')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (mjd_obs_min[ffi],'Minimum MJD of flat observations')
            master_holder.header[ffi]['MAXMJD'] = (mjd_obs_max[ffi],'Maximum MJD of flat observations')

            mjd_obs_mid = (mjd_obs_min[ffi] + mjd_obs_max[ffi]) * 0.5
            master_holder.header[ffi]['MIDMJD'] = (mjd_obs_mid,'Middle MJD of flat observations')
            t_object = Time(mjd_obs_mid,format='mjd')
            t_iso_string = str(t_object.iso)
            t_iso_string += "Z"
            t_iso_for_hdr = t_iso_string.replace(" ","T")
            master_holder.header[ffi]['DATE-MID'] = (t_iso_for_hdr,'Middle timestamp of flat observations')

            filename_match_bias = re.match(r".+/(kpf_.+\.fits)", self.masterbias_path)
            try:
                masterbias_path_filename_only = filename_match_bias.group(1)
            except:
                masterbias_path_filename_only = self.masterbias_path

            master_holder.header[ffi]['INPBIAS'] = masterbias_path_filename_only

            filename_match_dark = re.match(r".+/(kpf_.+\.fits)", self.masterdark_path)
            try:
                masterdark_path_filename_only = filename_match_dark.group(1)
            except:
                masterdark_path_filename_only = self.masterdark_path

            master_holder.header[ffi]['INPDARK'] = masterdark_path_filename_only

            datetimenow = datetime.now(timezone.utc)
            createdutc = datetimenow.strftime("%Y-%m-%dT%H:%M:%SZ")
            master_holder.header[ffi]['CREATED'] = (createdutc,'UTC of master-flat creation')
            master_holder.header[ffi]['INFOBITS'] = (master_flat_infobits,'Bit-wise flags defined below')

            master_holder.header[ffi]['BIT00'] = ('2**0 = 1', 'GREEN_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT01'] = ('2**1 = 2', 'RED_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT02'] = ('2**2 = 4', 'CA_HK" has gt 1% pixels with lt 10 samples')

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.header[ffi_unc_ext_name]['BUNIT'] = ('Dimensionless','Units of master-flat uncertainty')

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.header[ffi_cnt_ext_name]['BUNIT'] = ('Count','Number of stack samples')

            ffi_stack_ext_name = ffi + '_STACK'
            master_holder.header[ffi_stack_ext_name]['BUNIT'] = ('electrons/sec','Stacked-data mean per exposure time')

            ffi_lamp_ext_name = ffi + '_LAMP'
            master_holder.header[ffi_lamp_ext_name]['BUNIT'] = ('electrons/sec','Lamp pattern per exposure time')

            if (ffi == 'GREEN_CCD' or ffi == 'RED_CCD'):
                master_holder.header[ffi]['ORDRMASK'] = self.ordermask_path
                master_holder.header[ffi]['LAMPPATT'] = self.smoothlamppattern_path

                # Reconstruct name of order-trace file from observation date parsed from order-mask filename.

                ordtrace_match = re.match(r".+kpf_(\d\d\d\d\d\d\d\d).+\.fits",self.ordermask_path)

                try:
                    order_trace_obsdate = ordtrace_match.group(1)
                    order_trace_filename = "kpf_" + order_trace_obsdate + "_master_flat_" + ffi + ".csv"
                    master_holder.header[ffi]['ORDTRACE'] = order_trace_filename
                except:
                    pass

            n_filenames_kept = len(filenames_kept[ffi])
            for i in range(0, n_filenames_kept):
                input_filename_keyword = 'INFL' + str(i)

                filename_match = re.match(r".+/(KP.+\.fits)", filenames_kept[ffi][i])

                try:
                    filename_for_header = filename_match.group(1)
                except:
                    filename_for_header = filenames_kept[ffi][i]

                master_holder.header[ffi][input_filename_keyword] = filename_for_header

        # Clean up PRIMARY header.
        #keep_cards = ['SIMPLE','BITPIX','NAXIS','EXTEND','EXTNAME','OBJECT','IMTYPE',
        #              'SCI-OBJ','SKY-OBJ','CAL-OBJ','GRNAMPS','REDAMPS','EXPTIME',
        #              'INSTRUME','GREEN','RED','CA_HK','TARGNAME','TARGRV']
        #primary_hdu = fits.PrimaryHDU()
        #for keep_card in keep_cards:
        #    try:
        #        primary_hdu.header[keep_card] = master_holder.header['PRIMARY'][keep_card]
        #    except:
        #        self.logger.debug('Not found in PRIMARY header: keep_card = {}'.format(keep_card))
        #        pass
        #master_holder.header['PRIMARY'] = primary_hdu.header

        # Add informational to FITS header.  This is the only way I know of to keep the keyword comment.

        master_holder.header['PRIMARY']['IMTYPE'] = ('Flat','Master flat')

        master_holder.to_fits(self.masterflat_path)


        # Overwrite the newly created FITS file with one having a cleaned-up primary header.

        new_primary_hdr = fits.Header()
        new_primary_hdr['EXTNAME'] = 'PRIMARY'
        new_primary_hdr['DATE-OBS'] = date_obs
        new_primary_hdr['IMTYPE'] = ('Flat','Master flat')
        new_primary_hdr['TARGOBJ'] = (self.flat_object,'Target object of stacking')
        new_primary_hdr['INSTRUME'] = ('KPF','Doppler Spectrometer')
        new_primary_hdr['OBSERVAT'] = ('KECK','Observatory name')
        new_primary_hdr['TELESCOP'] = ('Keck I','Telescope')

        #FitsHeaders.cleanup_primary_header(self.masterflat_path,self.masterflat_path,new_primary_hdr)


        # Return list of values.

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_flat_exit_code,master_flat_infobits]

        return Arguments(exit_list)
