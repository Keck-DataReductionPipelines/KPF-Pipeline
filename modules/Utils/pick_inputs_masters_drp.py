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

        self.imtype_keywords = 'IMTYPE'       # Unlikely to be changed.
        self.bias_imtype_values_str = 'Bias'
        self.dark_imtype_values_str = 'Dark'
        self.arclamp_imtype_values_str = 'Arclamp'

        self.flat_imtype_keywords = ['IMTYPE','OBJECT']
        self.flat_imtype_values_str = ['Flatlamp','autocal-flat-all']

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
        Returns list of input FITS files for all master-calibration pipelines.

        """

        # Filter bias files with IMTYPE='Bias' and EXPTIME= 0.0.
        
        fh = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.bias_imtype_values_str,self.logger)
        all_bias_files,all_bias_objects = fh.get_good_biases()

        # Filter dark files with IMTYPE=‘Dark’ and the specified minimum exposure time.

        fh2 = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.dark_imtype_values_str,self.logger)
        all_dark_files,all_dark_objects = fh2.get_good_darks(self.exptime_minimum)

        # Filter flat files with IMTYPE=‘flatlamp’, but exclude those that either don't have
        # SCI-OBJ == CAL-OBJ and SKY-OBJ == CALOBJ or those with SCI-OBJ == "" or SCI-OBJ == "None".

        fh3 = FitsHeaders(self.all_fits_files_path,self.flat_imtype_keywords,self.flat_imtype_values_str,self.logger)
        all_flat_files = fh3.match_headers_string_lower()

        # Filter arclamp files with IMTYPE=‘arclamp’. 

        fh4 = FitsHeaders(self.all_fits_files_path,self.imtype_keywords,self.arclamp_imtype_values_str,self.logger)
        all_arclamp_files,all_arclamp_objects = fh4.get_good_arclamps()

        return Arguments(all_bias_files,all_dark_files,all_flat_files,all_arclamp_files,all_bias_objects,all_dark_objects,all_arclamp_objects)
