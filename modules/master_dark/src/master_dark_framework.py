from os.path import exists
import numpy as np
import configparser as cp
from datetime import datetime, timezone
from astropy.io import fits
from astropy.time import Time
import re

import database.modules.utils.kpf_db as db
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
        This class works within the Keck pipeline framework to compute the master dark,
        for a specific type of dark given by OBJECT in the primary FITS header,
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
        dark_object (str): Desired kind of bias (e.g., autocal-dark).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_master_bias.fits).
        masterdark_path (str): Pathname of output master dark (e.g., /testdata/kpf_master_dark.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        dark_object (str): Desired kind of bias (e.g., autocal-dark).
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
        self.dark_object = self.action.args[4]
        self.masterbias_path = self.action.args[5]
        self.masterdark_path = self.action.args[6]

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


        # Initialization.

        master_dark_exit_code = 0
        master_dark_infobits = 0


        # Filter dark files with IMTYPE=‘dark’ and that match the input object specification with OBJECT.
        # Parse obsdate

        self.logger.info('self.dark_object = {}'.format(self.dark_object))

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_dark_files,all_dark_objects = fh.get_good_darks(self.exptime_minimum)
        n_all_dark_files = len(all_dark_files)

        if n_all_dark_files == 0:
            self.logger.info('n_all_dark_files = {}'.format(n_all_dark_files))
            master_dark_exit_code = 8
            exit_list = [master_dark_exit_code,master_dark_infobits]
            return Arguments(exit_list)

        obsdate_match = re.match(r".*(\d\d\d\d\d\d\d\d).*", all_dark_files[0])
        try:
            obsdate = obsdate_match.group(1)
            self.logger.info('obsdate = {}'.format(obsdate))
        except:
            self.logger.info("obsdate not parsed from input filename")
            obsdate = None


        # Get master calibration files.

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
                     master_dark_exit_code = 5
                     exit_list = [master_dark_exit_code,master_dark_infobits]
                     return Arguments(exit_list)
            else:
                self.logger.info('Observation date not available so master bias file cannot be queried from database; returning...')
                master_dark_exit_code = 10
                exit_list = [master_dark_exit_code,master_dark_infobits]
                return Arguments(exit_list)

        dbh.close()      # Close database connection.

        master_bias_data = KPF0.from_fits(self.masterbias_path,self.data_type)

        mjd_obs_list = []
        exp_time_list = []
        dark_object_list = []
        for dark_file_path in (all_dark_files):
            dark_file = KPF0.from_fits(dark_file_path,self.data_type)
            mjd_obs = float(dark_file.header['PRIMARY']['MJD-OBS'])
            mjd_obs_list.append(mjd_obs)
            exp_time = float(dark_file.header['PRIMARY']['ELAPSED'])
            exp_time_list.append(exp_time)
            self.logger.debug('dark_file_path,exp_time = {},{}'.format(dark_file_path,exp_time))
            header_object = dark_file.header['PRIMARY']['OBJECT']
            dark_object_list.append(header_object)
            #self.logger.debug('dark_file_path,exp_time,header_object = {},{},{}'.format(dark_file_path,exp_time,header_object))


        # Ensure prototype FITS header for product file has matching OBJECT and contains both
        # GRNAMPS and REDAMPS keywords (indicating that the data exist).

        for dark_file_path in (all_dark_files):

            tester = KPF0.from_fits(dark_file_path)
            tester_object = tester.header['PRIMARY']['OBJECT']

            if tester_object == self.dark_object:

                try:
                    tester_grnamps = tester.header['PRIMARY']['GRNAMPS']
                except KeyError as err:
                    continue

                try:
                    tester_redamps = tester.header['PRIMARY']['REDAMPS']
                except KeyError as err:
                    continue

                self.logger.info('Prototype FITS header from {}'.format(dark_file_path))

                date_obs = tester.header['PRIMARY']['DATE-OBS']

                break

            else:

                tester = None

        if tester is None:
            master_dark_exit_code = 6
            exit_list = [master_dark_exit_code,master_dark_infobits]
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

            self.logger.debug('Loading dark data, ffi = {}'.format(ffi))
            keep_ffi = 0

            filenames_kept_list = []
            frames_data = []
            frames_data_exptimes = []
            frames_data_mjdobs = []
            frames_data_path = []
            n_all_dark_files = len(all_dark_files)
            for i in range(0, n_all_dark_files):

                exp_time = exp_time_list[i]
                mjd_obs = mjd_obs_list[i]
                header_object = dark_object_list[i]

                #self.logger.debug('i,fitsfile,ffi,exp_time,dark_object_list[i] = {},{},{},{},{}'.format(i,all_dark_files[i],ffi,exp_time,dark_object_list[i]))

                if header_object != self.dark_object:
                    #self.logger.debug('---->ffi,header_object,self.dark_object = {},{},{}'.format(ffi,header_object,self.dark_object))
                    continue

                path = all_dark_files[i]
                obj = KPF0.from_fits(path)

                try:
                    obj_not_junk = obj.header['PRIMARY']['NOTJUNK']
                    self.logger.debug('----========-------========------>path,obj_not_junk = {},{}'.format(path,obj_not_junk))
                    if obj_not_junk != 1:
                        continue
                except KeyError as err:
                    pass

                np_obj_ffi = np.array(obj[ffi])
                np_obj_ffi_shape = np.shape(np_obj_ffi)
                n_dims = len(np_obj_ffi_shape)
                self.logger.debug('path,ffi,n_dims = {},{},{}'.format(path,ffi,n_dims))
                if n_dims == 2:       # Check if valid data extension
                    keep_ffi = 1
                    filenames_kept_list.append(all_dark_files[i])
                    frames_data.append(obj[ffi])
                    frames_data_exptimes.append(exp_time)
                    frames_data_mjdobs.append(mjd_obs)
                    frames_data_path.append(path)

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

            self.logger.debug('Subtracted master bias from dark data...')

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

                # Separately normalize debiased images by EXPTIME.

                exp_time = frames_data_exptimes[i]
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

        # Remove confusing or non-relevant keywords, if existing.

        try:
            del master_holder.header['GREEN_CCD']['OSCANV1']
            del master_holder.header['GREEN_CCD']['OSCANV2']
            del master_holder.header['GREEN_CCD']['OSCANV3']
            del master_holder.header['GREEN_CCD']['OSCANV4']
            del master_holder.header['RED_CCD']['OSCANV1']
            del master_holder.header['RED_CCD']['OSCANV2']
            del master_holder.header['RED_CCD']['OSCANV3']
            del master_holder.header['RED_CCD']['OSCANV4']

        except KeyError as err:
            pass

        for ffi in self.lev0_ffi_exts:
            if ffi in del_ext_list: continue
            master_holder.header[ffi]['BUNIT'] = ('electrons/sec','Units of master dark')
            master_holder.header[ffi]['NFRAMES'] = (n_frames_kept[ffi],'Number of frames in input stack')
            master_holder.header[ffi]['MINEXPTM'] = (self.exptime_minimum,'Minimum exposure time of input darks (seconds)')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (mjd_obs_min[ffi],'Minimum MJD of dark observations')
            master_holder.header[ffi]['MAXMJD'] = (mjd_obs_max[ffi],'Maximum MJD of dark observations')

            mjd_obs_mid = (mjd_obs_min[ffi] + mjd_obs_max[ffi]) * 0.5
            master_holder.header[ffi]['MIDMJD'] = (mjd_obs_mid,'Middle MJD of dark observations')
            t_object = Time(mjd_obs_mid,format='mjd')
            t_iso_string = str(t_object.iso)
            t_iso_string += "Z"
            t_iso_for_hdr = t_iso_string.replace(" ","T")
            master_holder.header[ffi]['DATE-MID'] = (t_iso_for_hdr,'Middle timestamp of dark observations')

            filename_match_bias = re.match(r".+/(kpf_.+\.fits)", self.masterbias_path)
            try:
                masterbias_path_filename_only = filename_match_bias.group(1)
            except:
                masterbias_path_filename_only = self.masterbias_path

            master_holder.header[ffi]['INPBIAS'] = masterbias_path_filename_only

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

        master_holder.header['PRIMARY']['IMTYPE'] = ('Dark','Master dark')

        master_holder.to_fits(self.masterdark_path)


        # Overwrite the newly created FITS file with one having a cleaned-up primary header.

        new_primary_hdr = fits.Header()
        new_primary_hdr['EXTNAME'] = 'PRIMARY'
        new_primary_hdr['DATE-OBS'] = date_obs
        new_primary_hdr['IMTYPE'] = ('Dark','Master dark')
        new_primary_hdr['TARGOBJ'] = (self.dark_object,'Target object of stacking')
        new_primary_hdr['INSTRUME'] = ('KPF','Doppler Spectrometer')
        new_primary_hdr['OBSERVAT'] = ('KECK','Observatory name')
        new_primary_hdr['TELESCOP'] = ('Keck I','Telescope')

        #FitsHeaders.cleanup_primary_header(self.masterdark_path,self.masterdark_path,new_primary_hdr)


        # Return list of values.

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_dark_exit_code,master_dark_infobits]

        return Arguments(exit_list)
