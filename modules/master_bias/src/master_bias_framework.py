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
DEFAULT_CFG_PATH = 'modules/master_bias/configs/default.cfg'

class MasterBiasFramework(KPF0_Primitive):

    """
    Description:
        This class works within the Keck pipeline framework to compute
        the master bias by stacking input images for exposures with
        EXPTIME <= 0.0 selected from the given path, which can include
        many kinds of FITS files, not just biases.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of output master bias (e.g., /testdata/kpf_green_red_bias.fits).

    Attributes:
        data_type (str): Type of data (e.g., KPF).
        n_sigma (float): Number of sigmas for data-clipping (e.g., 2.1).
        all_fits_files_path (str , which can include file glob): Location of inputs (e.g., /data/KP*.fits).
        lev0_ffi_exts (list of str): FITS extensions to stack (e.g., ['GREEN_CCD','RED_CCD']).
        masterbias_path (str): Pathname of output master bias (e.g., /testdata/kpf_green_red_bias.fits).
        imtype_keywords (str): FITS keyword for filtering input bias files (fixed as ['IMTYPE','OBJECT']).
        imtype_values_str (str): Values of FITS keyword (fixed as ['Bias','autocal-bias']).
        config_path (str): Location of default config file (modules/master_bias/configs/default.cfg)
        logger (object): Log messages written to log_path specified in default config file.
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.n_sigma = self.action.args[1]
        self.all_fits_files_path = self.action.args[2]
        self.lev0_ffi_exts = self.action.args[3]
        self.masterbias_path = self.action.args[4]

        self.imtype_keywords = ['IMTYPE','OBJECT']       # Unlikely to be changed.
        self.imtype_values_str = ['Bias','autocal-bias']

        try:
            self.config_path = context.config_path['master_bias']
            print("--->MasterBiasFramework class: self.config_path =",self.config_path)
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
        Returns [exitcode, infobits] after computing and writing master-bias FITS file.

        """

        master_bias_exit_code = 0
        master_bias_infobits = 0

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.imtype_values_str,self.logger)
        all_bias_files = fh.get_good_biases()

        mjd_obs_list = []
        for bias_file_path in (all_bias_files):
            bias_file = KPF0.from_fits(bias_file_path,self.data_type)
            mjd_obs = float(bias_file.header['PRIMARY']['MJD-OBS'])
            mjd_obs_list.append(mjd_obs)

        tester = KPF0.from_fits(all_bias_files[0])
        del_ext_list = []
        for i in tester.extensions.keys():
            if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'CA_HK' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                del_ext_list.append(i)
        master_holder = tester
        for ffi in self.lev0_ffi_exts:
            keep_ffi = 0
            frames_data=[]
            for path in all_bias_files:
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
                   master_bias_infobits |= 2**0
                elif "RED_CCD" in (ffi).upper():
                   master_bias_infobits |= 2**1
                elif "CA_HK" in (ffi).upper():
                   master_bias_infobits |= 2**2

        for ext in del_ext_list:
            master_holder.del_extension(ext)

        # Add informational keywords to FITS header.

        master_holder.header['PRIMARY']['IMTYPE'] = ('Bias','Master bias')

        for ffi in self.lev0_ffi_exts:
            if ffi in del_ext_list: continue
            master_holder.header[ffi]['BUNIT'] = ('DN','Units of master bias')
            master_holder.header[ffi]['NFRAMES'] = (len(all_bias_files),'Number of frames in stack')
            master_holder.header[ffi]['NSIGMA'] = (self.n_sigma,'Number of sigmas for data-clipping')
            master_holder.header[ffi]['MINMJD'] = (min(mjd_obs_list),'Minimum MJD of bias observations')
            master_holder.header[ffi]['MAXMJD'] = (max(mjd_obs_list),'Maximum MJD of bias observations')
            datetimenow = datetime.now(timezone.utc)
            createdutc = datetimenow.strftime("%Y-%m-%dT%H:%M:%SZ")
            master_holder.header[ffi]['CREATED'] = (createdutc,'UTC of master-bias creation')
            master_holder.header[ffi]['INFOBITS'] = (master_bias_infobits,'Bit-wise flags defined below')

            master_holder.header[ffi]['BIT00'] = ('2**0 = 1', 'GREEN_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT01'] = ('2**1 = 2', 'RED_CCD has gt 1% pixels with lt 10 samples')
            master_holder.header[ffi]['BIT02'] = ('2**2 = 4', 'CA_HK" has gt 1% pixels with lt 10 samples')

            ffi_unc_ext_name = ffi + '_UNC'
            master_holder.header[ffi_unc_ext_name]['BUNIT'] = ('DN','Units of master-bias uncertainty')

            ffi_cnt_ext_name = ffi + '_CNT'
            master_holder.header[ffi_cnt_ext_name]['BUNIT'] = ('Count','Number of stack samples')

        master_holder.to_fits(self.masterbias_path)

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [master_bias_exit_code,master_bias_infobits]

        return Arguments(exit_list)
