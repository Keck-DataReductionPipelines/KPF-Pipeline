from os.path import exists
import numpy as np
import numpy.ma as ma
import configparser as cp
from datetime import datetime, timezone
from scipy.ndimage import gaussian_filter
from scipy.stats import mode

from modules.Utils.kpf_fits import FitsHeaders
from modules.Utils.frame_stacker import FrameStacker

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments

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
        2. SCI-OBJ = CAL-OBJ = SKY-OBJ
        3. SCI-OBJ <> 'None' and SCI-OBJ not blank
        4. EXPTIME <= 2.0 seconds (GREEN), 1.0 seconds (RED) to avoid saturation

        Assumptions and caveats:
        1. Does not include correcting for the color of the lamp, and other subtleties
           specific to spectral data.
        2. Currently "master" flat-lamp pattern made "on the fly" by
           2-D Gaussian blurring (sigma=2 pixel) the stacked-image mean.
        3. Further modifications to this recipe are needed in order to use
           a master flat-lamp pattern from a prior night.
        4. Low-light pixels cannot be reliably used to
           compute the flat-field correction (e.g., less than 500 DN/sec).
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
        ordermask_path (str): Pathname of input order mask (e.g., /testdata/order_mask_3_2_20230414.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_green_red_bias.fits).
        masterflat_path (str): Pathname of output master flat (e.g., /testdata/kpf_green_red_flat.fits).
        ordermask_path (str): Pathname of input order mask (e.g., /testdata/order_mask_3_2_20230414.fits).
        imtype_keywords (str): FITS keyword for filtering input flat files (fixed as 'IMTYPE').
        imtype_values_str (str): Value of FITS keyword (fixed as 'Flatlamp'), to be converted to lowercase for test.
        module_config_path (str): Location of default config file (modules/master_flat/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
        gaussian_filter_sigma (float): 2-D Gaussian-blur sigma for smooth lamp pattern calculation (default = 2.0 pixels)
        low_light_limit = Low-light limit where flat is set to unity (default = 500.0 DN/sec)

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
        self.ordermask_path = self.action.args[7]

        self.imtype_keywords = 'IMTYPE'       # Unlikely to be changed.
        self.imtype_values_str = 'Flatlamp'

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
        self.low_light_limit = float(module_param_cfg.get('low_light_limit', 500.0))
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

        order_mask_data = KPF0.from_fits(self.ordermask_path,self.data_type)
        self.logger.debug('Finished loading order-mask data from FITS file = {}'.format(self.ordermask_path))

        masterbias_path_exists = exists(self.masterbias_path)
        if not masterbias_path_exists:
            raise FileNotFoundError('File does not exist: {}'.format(self.masterbias_path))
        self.logger.info('self.masterbias_path = {}'.format(self.masterbias_path))
        self.logger.info('masterbias_path_exists = {}'.format(masterbias_path_exists))

        masterdark_path_exists = exists(self.masterdark_path)
        if not masterdark_path_exists:
            raise FileNotFoundError('File does not exist: {}'.format(self.masterdark_path))
        self.logger.info('self.masterdark_path = {}'.format(self.masterdark_path))
        self.logger.info('masterdark_path_exists = {}'.format(masterdark_path_exists))

        master_bias_data = KPF0.from_fits(self.masterbias_path,self.data_type)
        master_dark_data = KPF0.from_fits(self.masterdark_path,self.data_type)

        master_flat_exit_code = 0
        master_flat_infobits = 0

        # Filter flat files with IMTYPE=‘flatlamp’, but exclude those that either don't have
        # SCI-OBJ == CAL-OBJ and SKY-OBJ == CALOBJ or those with SCI-OBJ == "" or SCI-OBJ == "None"
        # or those with EXPTIME > 2.0 seconds (GREEN) or 1.0 seconds (RED) to avoid saturation.

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_flat_files = fh.get_good_flats()

        mjd_obs_list = []
        exp_time_list = []
        for flat_file_path in (all_flat_files):
            flat_file = KPF0.from_fits(flat_file_path,self.data_type)
            mjd_obs = flat_file.header['PRIMARY']['MJD-OBS']
            mjd_obs_list.append(mjd_obs)
            exp_time = flat_file.header['PRIMARY']['EXPTIME']
            exp_time_list.append(exp_time)
            self.logger.debug('flat_file_path,exp_time = {},{}'.format(flat_file_path,exp_time))

        tester = KPF0.from_fits(all_flat_files[0])
        del_ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'CA_HK' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                del_ext_list.append(i)
        master_holder = tester

        n_frames_kept = {}
        mjd_obs_min = {}
        mjd_obs_max = {}
        for ffi in self.lev0_ffi_exts:

            self.logger.debug('Loading flat data, ffi = {}'.format(ffi))
            keep_ffi = 0

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
                np_obj_ffi = np.array(obj[ffi])
                np_obj_ffi_shape = np.shape(np_obj_ffi)
                n_dims = len(np_obj_ffi_shape)
                self.logger.debug('path,ffi,n_dims = {},{},{}'.format(path,ffi,n_dims))
                if n_dims == 2:       # Check if valid data extension
                     keep_ffi = 1
                     frames_data.append(obj[ffi])
                     frames_data_exptimes.append(exp_time)
                     frames_data_mjdobs.append(mjd_obs)
                     frames_data_path.append(path)
                     self.logger.debug('Keeping flat image: i,fitsfile,ffi,mjd_obs,exp_time = {},{},{},{},{}'.format(i,all_flat_files[i],ffi,mjd_obs,exp_time))

            if keep_ffi == 0:
                self.logger.debug('ffi,keep_ffi = {},{}'.format(ffi,keep_ffi))
                del_ext_list.append(ffi)
                break

            frames_data = np.array(frames_data) - np.array(master_bias_data[ffi])      # Subtract master bias.

            self.logger.debug('Subtracting master bias from flat data...')

            normalized_frames_data=[]
            n_frames = (np.shape(frames_data))[0]
            self.logger.debug('Number of frames in stack = {}'.format(n_frames))

            n_frames_kept[ffi] = n_frames
            mjd_obs_min[ffi] = min(frames_data_mjdobs)
            mjd_obs_max[ffi] = max(frames_data_mjdobs)

            for i in range(0, n_frames):
                single_frame_data = frames_data[i]
                exp_time = frames_data_exptimes[i]

                self.logger.debug('Normalizing flat image: i,fitsfile,ffi,exp_time = {},{},{},{}'.format(i,frames_data_path[i],ffi,exp_time))

                single_normalized_frame_data = single_frame_data / exp_time       # Separately normalize by EXPTIME.

                single_normalized_frame_data -= np.array(master_dark_data[ffi])   # Subtract master-dark-current rate.

                normalized_frames_data.append(single_normalized_frame_data)

            #
            # Stack the frames.
            #

            normalized_frames_data = np.array(normalized_frames_data)

            fs = FrameStacker(normalized_frames_data,self.n_sigma,self.logger)
            stack_avg,stack_var,cnt,stack_unc = fs.compute()

            # Divide by the smoothed Flatlamp pattern.
            # Nominal 2-D Gaussian blurring at sigma=2.0 to smooth pixel-to-pixel variations.
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

                    normalization_factor = mode_vals[0] / 100.0      # Divide by 100 to account for above binning.

                    flat = np.where(np_om_ffi_bool == True, flat / normalization_factor, flat)
                    flat_unc = np.where(np_om_ffi_bool == True, flat_unc / normalization_factor, flat_unc)

                    self.logger.debug('orderlet_val,unnormalized_flat_mean,normalization_factor,mode_count = {},{},{},{}'.format(orderlet_val,unnormalized_flat_mean,normalization_factor,mode_counts[0]))

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

        # Add informational keywords to FITS header.  Remove non-relevant keywords too.

        master_holder.header['PRIMARY']['IMTYPE'] = ('Flat','Master flat')

        del master_holder.header['GREEN_CCD']['OSCANV1']
        del master_holder.header['GREEN_CCD']['OSCANV2']
        del master_holder.header['GREEN_CCD']['OSCANV3']
        del master_holder.header['GREEN_CCD']['OSCANV4']
        del master_holder.header['RED_CCD']['OSCANV1']
        del master_holder.header['RED_CCD']['OSCANV2']

        for ffi in self.lev0_ffi_exts:
            if ffi in del_ext_list: continue
            master_holder.header[ffi]['BUNIT'] = ('Dimensionless','Units of master flat')
            master_holder.header[ffi]['NFRAMES'] = (n_frames_kept[ffi],'Number of frames in input stack')
            master_holder.header[ffi]['GAUSSSIG'] = (self.gaussian_filter_sigma,'2-D Gaussian-smoother sigma (pixels)')
            master_holder.header[ffi]['LOWLTLIM'] = (self.low_light_limit,'Low-light limit (DN)')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (mjd_obs_min[ffi],'Minimum MJD of flat observations')
            master_holder.header[ffi]['MAXMJD'] = (mjd_obs_max[ffi],'Maximum MJD of flat observations')
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
            master_holder.header[ffi_stack_ext_name]['BUNIT'] = ('DN/sec','Stacked-data mean per exposure time')

            ffi_lamp_ext_name = ffi + '_LAMP'
            master_holder.header[ffi_lamp_ext_name]['BUNIT'] = ('DN/sec','Lamp pattern per exposure time')

        master_holder.to_fits(self.masterflat_path)

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_flat_exit_code,master_flat_infobits]

        return Arguments(exit_list)
