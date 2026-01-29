import re

import database.modules.utils.kpf_db as db
from modules.Utils.kpf_fits import FitsHeaders

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/Utils/pick_inputs_masters_drp.cfg'
CONTEXT_CFG_PATH = 'pick_inputs_masters_drp'

class PickInputsMastersDRP(KPF0_Primitive):

    """
    Description:
        This class picks all input FITS files in a given directory
        that are required as inputs for all master-calibration pipelines.
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.all_fits_files_path = self.action.args[1]
        self.exptime_minimum = self.action.args[2]
        self.flat_object = self.action.args[3]

        self.imtype_keywords = 'IMTYPE'       # Unlikely to be changed.
        self.bias_imtype_values_str = 'Bias'
        self.dark_imtype_values_str = 'Dark'
        self.arclamp_imtype_values_str = 'Arclamp'

        self.flat_imtype_keywords = ['IMTYPE','OBJECT']
        #self.flat_imtype_values_str = ['Flatlamp','autocal-flat-all']
        #self.flat_imtype_values_str = ['Flatlamp','test-flat-all']
        self.flat_imtype_values_str = ['Flatlamp',self.flat_object]

        try:
            self.config_path = context.config_path[CONTEXT_CFG_PATH]
            print("--->PickInputsMastersDRP class: self.config_path =",self.config_path)
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
        Returns list of input L0 FITS files for all master-calibration pipelines.
        Filter out any FITS files that have a dash in the filename (which have been
        replaced by a redelivery from the mountain).
        """

        # Filter bias files with IMTYPE='Bias' and EXPTIME= 0.0.

        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.bias_imtype_values_str,self.logger)
        all_bias_files,all_bias_objects = fh.get_good_biases()

        ret_bias_files = []
        for bias_file in all_bias_files:
            if "-" in bias_file:
                continue
            ret_bias_files.append(bias_file)

        # Filter dark files with IMTYPE=‘Dark’ and the specified minimum exposure time.

        fh2 = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.dark_imtype_values_str,self.logger)
        all_dark_files,all_dark_objects = fh2.get_good_darks(self.exptime_minimum)

        ret_dark_files = []
        for dark_file in all_dark_files:
            if "-" in dark_file:
                continue
            ret_dark_files.append(dark_file)

        # Filter flat files with IMTYPE=‘flatlamp’ and specified OBJECT.
        # Try to extract the observation date from self.all_fits_files_path
        # (e.g., '/data/kpf/20260125/*.fits') and perform a database query to
        # get files on either side of the UT boundary; if the observation date
        # not found, then fall back on the original method to obtain the files.

        filename_match = re.match(r".+/(\d\d\d\d\d\d\d\d)/.+", self.all_fits_files_path)

        try:

            dateobs_match = filename_match.group(1)

            self.logger.info(f"dateobs_match = {dateobs_match}")


            # Open database connection.

            dbh = db.KPFDB(verbose=True)


            # Parameters for querying all L0 FITS file records associated with the desired calibration product.

            dateobs = dateobs_match
            imtype = self.flat_imtype_values_str[0]
            object = self.flat_imtype_values_str[1]
            contentbitmask = 3
            hoursbeforemidnight = 0                    # Until kpf_masters_2D.recipe is repaired, keep this zero.
            hoursaftermidnight = 6


            # Query database for L0 FITS files for flatfield generation.

            records = dbh.get_l0_calibration_fits_files(dateobs,
                                                        imtype,
                                                        object,
                                                        contentbitmask,
                                                        hoursbeforemidnight,
                                                        hoursaftermidnight)

            self.logger.info('database-query exit_code = {}'.format(dbh.exit_code))

            nrecs = len(records)
            self.logger.info(f'nrecs= {nrecs}')

            self.logger.info(f'records= {records}')


            # Close database connection.

            dbh.close()


            # Parse filename from records.
            # Replace outside-container path with inside.

            all_flat_files = []
            for record in records:
                filename = record[1].replace("/data/kpf","/data")
                self.logger.info(f'filename= {filename}')
                all_flat_files.append(filename)

        except:

            self.logger.info(f"No dateobs match found (self.all_fits_files_path = {self.all_fits_files_path}); " +\
                             "continuing with original method...")

            fh3 = FitsHeaders(self.all_fits_files_path,self.flat_imtype_keywords,self.flat_imtype_values_str,self.logger)
            all_flat_files = fh3.match_headers_string_lower()


        # Filter out filenames containing a dash (means it was redelivered).

        ret_flat_files = []
        for flat_file in all_flat_files:
            if "-" in flat_file:
                continue
            ret_flat_files.append(flat_file)

        # Filter arclamp files with IMTYPE=‘arclamp’.

        fh4 = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.arclamp_imtype_values_str,self.logger)
        all_arclamp_files,all_arclamp_objects = fh4.get_good_arclamps()

        ret_arclamp_files = []
        for arclamp_file in all_arclamp_files:
            if "-" in arclamp_file:
                continue
            ret_arclamp_files.append(arclamp_file)

        return Arguments(ret_bias_files,ret_dark_files,ret_flat_files,ret_arclamp_files,all_bias_objects,all_dark_objects,all_arclamp_objects)
