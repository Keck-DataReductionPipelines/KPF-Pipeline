import numpy as np
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
DEFAULT_CFG_PATH = 'modules/master_lfc/configs/default.cfg'

class MasterLFCFramework(KPF0_Primitive):

    """
    Description:
        This class works within the Keck pipeline framework to compute
        the master lfc by stacking input images for exposures with 
        SCI-OBJ=LFCFiber selected from the given path, which can include
        many kinds of FITS files, not just lfces.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterlfc_path (str): Pathname of output master lfc (e.g., /testdata/kpf_green_red_lfc.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterlfc_path (str): Pathname of output master lfc (e.g., /testdata/kpf_green_red_lfc.fits).
        exptime_keyword (str): FITS keyword for filtering input lfc files (fixed as 'EXPTIME').
        exptime__value_str (str): Maximum value of FITS keyword (fixed as <= '0.0').
        config_path (str): Location of default config file (modules/master_lfc/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.n_sigma = self.action.args[1]
        self.all_fits_files_path = self.action.args[2]
        self.lev0_ffi_exts = self.action.args[3]
        self.masterlfc_path = self.action.args[4]

        self.match_keyword = 'SCI-OBJ'   # Unlikely to be changed.
        self.match_value_str = 'lfcfiber'

        try:
            self.config_path = context.config_path['master_lfc']
            print("--->MasterLFCFramework class: self.config_path =",self.config_path)
        except:
            self.config_path = DEFAULT_CFG_PATH

        print("{} class: self.config_path = {}".format(self.__class__.__name__,self.config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('config_path = {}'.format(self.config_path))


    def _perform(self):

        """
        Returns [exitcode, infobits] after computing and writing master-lfc FITS file.

        """

        master_lfc_exit_code = 0
        master_lfc_infobits = 0

        fh = FitsHeaders(self.all_fits_files_path,self.match_keyword,self.match_value_str,self.logger)
        all_lfc_files = fh.match_headers_string_lower()

        mjd_obs_list = []
        for lfc_file_path in (all_lfc_files):
            lfc_file = KPF0.from_fits(lfc_file_path,self.data_type)
            mjd_obs = lfc_file.header['PRIMARY']['MJD-OBS']
            mjd_obs_list.append(mjd_obs)

        tester = KPF0.from_fits(all_lfc_files[0])
        ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                ext_list.append(i)
        master_holder = tester
        for ffi in self.lev0_ffi_exts:
            frames_data=[]
            for path in all_lfc_files:
                obj = KPF0.from_fits(path)
                exp = obj.header['PRIMARY']['EXPTIME']
                frames_data.append(obj[ffi] / exp)
            frames_data = np.array(frames_data)

            fs = FrameStacker(frames_data,self.n_sigma)
            avg,var,cnt,unc = fs.compute()

            ### kpf master file creation ###
            master_holder[ffi] = avg

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.create_extension(ffi_unc_ext_name,ext_type=np.array)
            master_holder[ffi_unc_ext_name] = unc.astype(np.float32)

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
                   master_lfc_infobits |= 2**0
                elif "RED_CCD" in (ffi).upper():
                   master_lfc_infobits |= 2**1
                elif "CA_HK" in (ffi).upper():
                   master_lfc_infobits |= 2**2
            
        for ext in ext_list:
            master_holder.del_extension(ext)

        # Add informational keywords to FITS header.

        for ffi in self.lev0_ffi_exts:
            master_holder.header[ffi]['NFRAMES'] = (len(all_lfc_files),'Number of frames in stack')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (min(mjd_obs_list),'Minimum MJD of lfc observations')
            master_holder.header[ffi]['MAXMJD'] = (max(mjd_obs_list),'Maximum MJD of lfc observations')
            datetimenow = datetime.now(timezone.utc)
            createdutc = datetimenow.strftime("%Y-%m-%dT%H:%M:%SZ")
            master_holder.header[ffi]['CREATED'] = (createdutc,'UTC of master-lfc creation')
            master_holder.header[ffi]['INFOBITS'] = (master_lfc_infobits,'Bit-wise flags defined below')

            master_holder.header[ffi]['BIT00'] = ('2**0 = 1', 'GREEN_CCD has >1% pixels with <10 samples')
            master_holder.header[ffi]['BIT01'] = ('2**1 = 2', 'RED_CCD has >1% pixels with <10 samples')
            master_holder.header[ffi]['BIT02'] = ('2**2 = 4', 'CA_HK" has >1% pixels with <10 samples')
            
        master_holder.to_fits(self.masterlfc_path)

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_lfc_exit_code,master_lfc_infobits]

        return Arguments(exit_list)
