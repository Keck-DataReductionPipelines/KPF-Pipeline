from os.path import exists
import numpy as np
import configparser as cp
from datetime import datetime, timezone
import re
from astropy.io import fits
from astropy.time import Time

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

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/master_arclamp/configs/default.cfg'

class MasterArclampFramework(KPF0_Primitive):

    """
    Description:
        This class works within the Keck pipeline framework to compute the master arclamp,
        for a specific type of illumination given by OBJECT in the primary FITS header,
        by stacking input images for exposures with IMTYPE.lower() == 'arclamp'
        and specified input arclamp_object.  The input FITS files are selected from the
        given path that can include many kinds of FITS files, not just arclamp files.
        Subtract master bias from each input flat 2D raw image.  Separately normalize
        debiased images by EXPTIME, and then subtract master dark.  Optionally divide by
        the master flat.  Stack all normalized debiased, flattened images.
        Set appropriate infobit if number of pixels with less than 5 samples
        is greater than 1% of total number of image pixels.
        Currently makes master arclamps for GREEN_CCD and RED_CCD only.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        arclamp_object (str): Desired kind of stack (e.g.,  autocal-une-sky).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_master_bias.fits).
        masterdark_path (str): Pathname of input master dark (e.g., /testdata/kpf_master_dark.fits).
        masterflat_path (str): Pathname of output master flat (e.g., /testdata/kpf_master_flat.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        arclamp_object (str): Desired kind of stack (e.g.,  autocal-une-sky).
        masterbias_path (str): Pathname of input master bias (e.g., /testdata/kpf_green_red_bias.fits).
        masterdark_path (str): Pathname of output master dark (e.g., /testdata/kpf_green_red_dark.fits).
        masterflat_path (str): Pathname of output master flat (e.g., /testdata/kpf_master_flat.fits).
        imtype_keywords (str): FITS keyword for filtering input flat files (fixed as 'IMTYPE').
        imtype_values_str (str): Value of FITS keyword (fixed as 'Flatlamp'), to be converted to lowercase for test.
        module_config_path (str): Location of default config file (modules/master_flat/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
        skip_flattening (int): Set to 1 to skip flattening of the inputs; otherwise zero.
        max_num_frames_to_stack (int): Maximum number of frames allowed in the stack.
        min_num_frames_to_stack_with_outlier_rejection (int): Minimum number of frames to stack_with_outlier_rejection
                                                              (if less than this minimum and >=2, then use simple median)

    Outputs:
        Full-frame-image FITS extensions in output master arclamp:
        EXTNAME = 'GREEN_CCD'          / GREEN flat-field corrections
        EXTNAME = 'RED_CCD '           / RED flat-field corrections
        EXTNAME = 'GREEN_CCD_UNC'      / GREEN flat-field uncertainties
        EXTNAME = 'GREEN_CCD_CNT'      / GREEN stack sample numbers (after data-clipping)
        EXTNAME = 'RED_CCD_UNC'        / RED flat-field uncertainties
        EXTNAME = 'RED_CCD_CNT'        / RED stack sample numbers (after data-clipping)

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.n_sigma = self.action.args[1]
        self.all_fits_files_path = self.action.args[2]
        self.lev0_ffi_exts = self.action.args[3]
        self.arclamp_object = self.action.args[4]
        self.masterbias_path = self.action.args[5]
        self.masterdark_path = self.action.args[6]
        self.masterflat_path = self.action.args[7]
        self.masterarclamp_path = self.action.args[8]

        self.imtype_keywords = ['IMTYPE','OBJECT']       # Unlikely to be changed.
        self.imtype_values_str = ['Arclamp',self.arclamp_object]

        try:
            self.module_config_path = context.config_path['master_arclamp']
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

        self.skip_flattening = int(module_param_cfg.get('skip_flattening', 0))
        self.max_num_frames_to_stack = int(module_param_cfg.get('max_num_frames_to_stack', 50))
        self.min_num_frames_to_stack_with_outlier_rejection = int(module_param_cfg.get('min_num_frames_to_stack_with_outlier_rejection', 4))

        self.logger.info('self.skip_flattening = {}'.format(self.skip_flattening))
        self.logger.info('self.max_num_frames_to_stack = {}'.format(self.max_num_frames_to_stack))
        self.logger.info('self.min_num_frames_to_stack_with_outlier_rejection = {}'.format(self.min_num_frames_to_stack_with_outlier_rejection))


    def _perform(self):

        """
        Returns [exitcode, infobits] after computing and writing master-arclamp FITS file.  See below for possible exitcodes.

        Exitcode      Comment
        ----------------------------------------
          0           Normal execution
          5           Could not query database for fallback input master file
          6           tester_object != self.arclamp_object (unlikely with pre-checking added later)
          7           Number of input files < 2 for given valid FITS extension (ffi == 'GREEN_CCD' or ffi == 'RED_CCD')
          8           Number of input files = 0
         10           Observation date not available to query database for fallback input master file
        """


        # Initialization.

        master_arclamp_exit_code = 0
        master_arclamp_infobits = 0
        input_master_type = 'all'


        # Filter arclamp files with IMTYPE=‘arclamp’ and that match the input object specification with OBJECT.
        # Parse obsdate

        self.logger.info('self.arclamp_object = {}'.format(self.arclamp_object))

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_arclamp_files = fh.match_headers_string_lower()
        n_all_arclamp_files = len(all_arclamp_files)

        if n_all_arclamp_files == 0:
            self.logger.info('n_all_arclamp_files = {}'.format(n_all_arclamp_files))
            master_arclamp_exit_code = 8
            exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
            return Arguments(exit_list)

        obsdate_match = re.match(r".*(\d\d\d\d\d\d\d\d).*", all_arclamp_files[0])
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
                     master_arclamp_exit_code = 5
                     exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
                     return Arguments(exit_list)
            else:
                self.logger.info('Observation date not available so master bias file cannot be queried from database; returning...')
                master_arclamp_exit_code = 10
                exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
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
                     master_arclamp_exit_code = 5
                     exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
                     return Arguments(exit_list)
            else:
                self.logger.info('Observation date not available so master dark file cannot be queried from database; returning...')
                master_arclamp_exit_code = 10
                exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
                return Arguments(exit_list)

        self.logger.info('self.masterdark_path = {}'.format(self.masterdark_path))

        if self.skip_flattening == 0:
            masterflat_path_exists = exists(self.masterflat_path)
            self.logger.info('masterflat_path_exists = {}'.format(masterflat_path_exists))

            if not masterflat_path_exists:
                if obsdate != None:
                    cal_type_pair = ['flat','autocal-flat-all']                    # Query database for fallback master flat.
                    dbh.get_nearest_master_file(obsdate,cal_file_level,contentbitmask,cal_type_pair)
                    self.logger.info('database-query exit_code = {}'.format(dbh.exit_code))
                    self.logger.info('Master dark database-query filename = {}'.format(dbh.filename))
                    if dbh.exit_code == 0:
                        self.masterflat_path = dbh.filename
                    else:
                        self.logger.info('Master flat file cannot be queried from database; returning...')
                        master_arclamp_exit_code = 5
                        exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
                        return Arguments(exit_list)
                else:
                    self.logger.info('Observation date not available so master flat file cannot be queried from database; returning...')
                    master_arclamp_exit_code = 10
                    exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
                    return Arguments(exit_list)

            self.logger.info('self.masterflat_path = {}'.format(self.masterflat_path))

        dbh.close()      # Close database connection.

        master_bias_data = KPF0.from_fits(self.masterbias_path,self.data_type)
        master_dark_data = KPF0.from_fits(self.masterdark_path,self.data_type)

        if self.skip_flattening == 0:
            master_flat_data = KPF0.from_fits(self.masterflat_path,self.data_type)

        mjd_obs_list = []
        exp_time_list = []
        arclamp_object_list = []
        for arclamp_file_path in (all_arclamp_files):
            arclamp_file = KPF0.from_fits(arclamp_file_path,self.data_type)
            mjd_obs = float(arclamp_file.header['PRIMARY']['MJD-OBS'])
            mjd_obs_list.append(mjd_obs)
            exp_time = float(arclamp_file.header['PRIMARY']['ELAPSED'])
            exp_time_list.append(exp_time)
            header_object = arclamp_file.header['PRIMARY']['OBJECT']
            arclamp_object_list.append(header_object)
            #self.logger.debug('arclamp_file_path,exp_time,header_object = {},{},{}'.format(arclamp_file_path,exp_time,header_object))


        # Ensure prototype FITS header for product file has matching OBJECT and contains both
        # GRNAMPS and REDAMPS keywords (indicating that the data exist).

        for arclamp_file_path in (all_arclamp_files):

            tester = KPF0.from_fits(arclamp_file_path)
            tester_object = tester.header['PRIMARY']['OBJECT']

            if tester_object == self.arclamp_object:

                try:
                    tester_grnamps = tester.header['PRIMARY']['GRNAMPS']
                except KeyError as err:
                    continue

                try:
                    tester_redamps = tester.header['PRIMARY']['REDAMPS']
                except KeyError as err:
                    continue

                self.logger.info('Prototype FITS header from {}'.format(arclamp_file_path))

                date_obs = tester.header['PRIMARY']['DATE-OBS']

                break

            else:

                tester = None

        if tester is None:
            master_arclamp_exit_code = 6
            exit_list = [master_arclamp_exit_code,master_arclamp_infobits]
            return Arguments(exit_list)


        del_ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                del_ext_list.append(i)
        master_holder = tester

        filenames_kept = {}
        n_frames_kept = {}
        mjd_obs_min = {}
        mjd_obs_max = {}
        for ffi in self.lev0_ffi_exts:

            self.logger.debug('Loading arclamp data, ffi = {}'.format(ffi))
            keep_ffi = 0

            filenames_kept_list = []
            frames_data = []
            frames_data_exptimes = []
            frames_data_mjdobs = []
            frames_data_path = []
            n = 0
            for i in range(0, n_all_arclamp_files):
                exp_time = exp_time_list[i]
                mjd_obs = mjd_obs_list[i]
                header_object = arclamp_object_list[i]

                #self.logger.debug('i,fitsfile,ffi,exp_time,arclamp_object_list[i] = {},{},{},{},{}'.format(i,all_arclamp_files[i],ffi,exp_time,arclamp_object_list[i]))

                if not (ffi == 'GREEN_CCD' or ffi == 'RED_CCD'):
                    raise NameError('FITS extension {} not supported; check recipe config file.'.format(ffi))

                if header_object != self.arclamp_object:
                    #self.logger.debug('---->ffi,header_object,self.arclamp_object = {},{},{}'.format(ffi,header_object,self.arclamp_object))
                    continue

                path = all_arclamp_files[i]
                obj = KPF0.from_fits(path)


                # Check QC keywords and skip image if it does not pass QC checking.

                skip = qc.check_all_qc_keywords(obj,path,input_master_type,self.logger)
                self.logger.debug('After calling qc.check_all_qc_keywords: i,path,skip = {},{},{}'.format(i,path,skip))
                if skip:
                    continue


                np_obj_ffi = np.array(obj[ffi])
                np_obj_ffi_shape = np.shape(np_obj_ffi)
                n_dims = len(np_obj_ffi_shape)
                #self.logger.debug('path,ffi,n_dims = {},{},{}'.format(path,ffi,n_dims))
                if n_dims == 2:       # Check if valid data extension
                    keep_ffi = 1
                    filenames_kept_list.append(all_arclamp_files[i])
                    frames_data.append(obj[ffi])
                    frames_data_exptimes.append(exp_time)
                    frames_data_mjdobs.append(mjd_obs)
                    frames_data_path.append(path)
                    #self.logger.debug('Keeping arclamp image: i,fitsfile,ffi,mjd_obs,exp_time = {},{},{},{},{}'.format(i,all_arclamp_files[i],ffi,mjd_obs,exp_time))
                    if n >= self.max_num_frames_to_stack:
                        continue
                    n += 1

            n_frames = (np.shape(frames_data))[0]
            self.logger.debug('Number of frames in stack = {}'.format(n_frames))

            # Skip extension if number of frames to stack is less than 2.

            if n_frames < 2:
                self.logger.debug('n_frames < 2 for ffi,n_frames = {},{}'.format(ffi,n_frames))
                del_ext_list.append(ffi)
                continue

            if keep_ffi == 0:
                #self.logger.debug('ffi,keep_ffi = {},{}'.format(ffi,keep_ffi))
                del_ext_list.append(ffi)
                continue

            # Subtract master bias.

            frames_data = np.array(frames_data) - np.array(master_bias_data[ffi])

            self.logger.debug('Subtracting master bias from arclamp data...')

            normalized_frames_data = []

            filenames_kept[ffi] = filenames_kept_list
            n_frames_kept[ffi] = n_frames
            mjd_obs_min[ffi] = min(frames_data_mjdobs)
            mjd_obs_max[ffi] = max(frames_data_mjdobs)

            for i in range(0, n_frames):
                single_frame_data = frames_data[i]
                exp_time = frames_data_exptimes[i]

                # Subtract off exposure time times master-dark-current rate.

                try:
                    single_normalized_frame_data = single_frame_data - exp_time * np.array(master_dark_data[ffi])
                except:
                    single_normalized_frame_data = single_frame_data
                    self.logger.debug('Could not subtract dark: np.shape(np.array(master_dark_data[ffi])) = {},{},{}'.format(i,ffi,np.shape(np.array(master_dark_data[ffi]))))

                if self.skip_flattening == 0:
                    #self.logger.debug('Flattening arclamp image: i,fitsfile,ffi,exp_time = {},{},{},{}'.format(i,frames_data_path[i],ffi,exp_time))
                    single_normalized_frame_data /= np.array(master_flat_data[ffi])   # Optionally divide master-flat.

                normalized_frames_data.append(single_normalized_frame_data)

            #
            # Stack the frames.
            #

            normalized_frames_data = np.array(normalized_frames_data)

            fs = FrameStacker(normalized_frames_data,self.n_sigma,self.logger)
            if n_frames >= self.min_num_frames_to_stack_with_outlier_rejection:
                self.logger.debug('Computing stack average with outlier rejection...')
                stack_avg,stack_var,cnt,stack_unc = fs.compute()
            else:
                self.logger.debug('Computing stack median (no outlier rejection)...')
                stack_avg,stack_var,cnt,stack_unc = fs.compute_stack_median()

            arclamp = stack_avg
            arclamp_unc = stack_unc

            ### kpf master file creation ###
            master_holder[ffi] = arclamp

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.create_extension(ffi_unc_ext_name,ext_type=np.array)
            master_holder[ffi_unc_ext_name] = arclamp_unc.astype(np.float32)

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.create_extension(ffi_cnt_ext_name,ext_type=np.array)
            master_holder[ffi_cnt_ext_name] = cnt.astype(np.int32)

            n_samples_lt_5 = (cnt < 5).sum()
            rows = np.shape(master_holder[ffi])[0]
            cols = np.shape(master_holder[ffi])[1]
            n_pixels = rows * cols
            pcent_diff = 100 * n_samples_lt_5 / n_pixels

            # Set appropriate infobit if number of pixels with less than 5 samples in
            # current FITS extension is greater than 1% of total number of pixels in image.

            if pcent_diff > 1.0:
                self.logger.info('ffi,n_samples_lt_5 = {},{}'.format(ffi,n_samples_lt_5))
                if "GREEN_CCD" in (ffi).upper():
                   master_arclamp_infobits |= 2**0
                elif "RED_CCD" in (ffi).upper():
                   master_arclamp_infobits |= 2**1

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
            master_holder.header[ffi]['BUNIT'] = ('electrons','Units of master arclamp')
            master_holder.header[ffi]['NFRAMES'] = (n_frames_kept[ffi],'Number of frames in input stack')
            master_holder.header[ffi]['SKIPFLAT'] = (self.skip_flattening,'Flag to skip flat-field calibration')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (mjd_obs_min[ffi],'Minimum MJD of arclamp observations')
            master_holder.header[ffi]['MAXMJD'] = (mjd_obs_max[ffi],'Maximum MJD of arclamp observations')

            mjd_obs_mid = (mjd_obs_min[ffi] + mjd_obs_max[ffi]) * 0.5
            master_holder.header[ffi]['MIDMJD'] = (mjd_obs_mid,'Middle MJD of arclamp observations')
            t_object = Time(mjd_obs_mid,format='mjd')
            t_iso_string = str(t_object.iso)
            t_iso_string += "Z"
            t_iso_for_hdr = t_iso_string.replace(" ","T")
            master_holder.header[ffi]['DATE-MID'] = (t_iso_for_hdr,'Middle timestamp of arclamp observations')

            master_holder.header[ffi]['TARGOBJ'] = (self.arclamp_object,'Target object of stacking')

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

            if self.skip_flattening == 0:

                filename_match_flat = re.match(r".+/(kpf_.+\.fits)", self.masterflat_path)
                try:
                    masterflat_path_filename_only = filename_match_flat.group(1)
                except:
                    masterflat_path_filename_only = self.masterflat_path

                master_holder.header[ffi]['INPFLAT'] = masterflat_path_filename_only

            datetimenow = datetime.now(timezone.utc)
            createdutc = datetimenow.strftime("%Y-%m-%dT%H:%M:%SZ")
            master_holder.header[ffi]['CREATED'] = (createdutc,'UTC of master-arclamp creation')
            master_holder.header[ffi]['INFOBITS'] = (master_arclamp_infobits,'Bit-wise flags defined below')

            master_holder.header[ffi]['BIT00'] = ('2**0 = 1', 'GREEN_CCD has gt 1% pixels with lt 5 samples')
            master_holder.header[ffi]['BIT01'] = ('2**1 = 2', 'RED_CCD has gt 1% pixels with lt 5 samples')

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.header[ffi_unc_ext_name]['BUNIT'] = ('electrons','Units of master-arclamp uncertainty')

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

        master_holder.header['PRIMARY']['IMTYPE'] = ('Arclamp','Master arclamp')

        master_holder.to_fits(self.masterarclamp_path)


        # Overwrite the newly created FITS file with one having a cleaned-up primary header.

        new_primary_hdr = fits.Header()
        new_primary_hdr['EXTNAME'] = 'PRIMARY'
        new_primary_hdr['DATE-OBS'] = date_obs
        new_primary_hdr['IMTYPE'] = ('Arclamp','Master arclamp')
        new_primary_hdr['TARGOBJ'] = (self.arclamp_object,'Target object of stacking')
        new_primary_hdr['INSTRUME'] = ('KPF','Doppler Spectrometer')
        new_primary_hdr['OBSERVAT'] = ('KECK','Observatory name')
        new_primary_hdr['TELESCOP'] = ('Keck I','Telescope')

        #FitsHeaders.cleanup_primary_header(self.masterarclamp_path,self.masterarclamp_path,new_primary_hdr)


        # Return list of values.

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_arclamp_exit_code,master_arclamp_infobits]

        return Arguments(exit_list)
