# Standard dependencies 
"""
    This module defines class `CaHKExtraction` which inherits from `KPF0_Primitive` and provides methods to perform the
    event on CA H&K extraction in the recipe.

    Description:
        * Method `__init__`:

            CaHKExtraction constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderTrace` event issued in the recipe:
                    - `action.args[0] (kpfpipe.models.level0.KPF0|str)`: Instance of `KPF0` or the path of a fits file
                      containing image data for H&K extraction.
                    - `action.args[1] (str)`: Path to a file defining the fiber and order location
                    - `action.args[2] (list)`: List containing the fiber names.
                    - `action.args[3] (kpfpipe.models.level1.KPF1)`:  Instance of `KPF1` containing spectral
                      extraction results. If not existing, it is None.
                    - `action.args['output_exts'] (str)`: Extension names of the extensions to contain
                      the extraction result for each fiber, optional, Defaults to fiber list.
                    - `action.args['output_wave_exts'] (str)`: Extension names of the extensions to contain
                      the wavelength solution for each fiber, optional, Defaults to fiber list prefixed with '_wave'.
                    - `action.args['dark'] (KPF0, optional)`: dark master file for dark subtraction. Defaults to None.
                    - `action.args['bias'] (KPF0, optional)`: bias master file for bias subtraction. Defaults to None.
                    - `action.args['wave_files'] (list, optional)`: Wavelength solution files per fiber list.
                      Defaults to None.
                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of hk_spectral_extraction  in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,
                - `input_img (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`,  assigned by `actions.args[0]`.
                - `fibers (list)`: The interested fibers to be processed, assigned by `actions.args[1]`.
                - `fiber_loc (dict)`: An dictionary instance contains the order location for each fiber.
                - `output_exts (list)`: output extension names.

        * Method `__perform`:

            CaHKExtraction returns the result in `Arguments` object which contains a level 1 data object (`KPF1`)
            with the extraction results.
    Usage:
        For the recipe, the order trace event is issued like::

            :
            lev0_data = kpf0_from_fits(input_lev0_file)
            output_data = kpf1_from_fits(output_lev1_file, data_type = data_type)
            :
            output_data = CaHKExtraction(lev0_data,
                    hk_trace_table,                     # ex. /data/masters/kpfMaster_HKOrderBounds20220909.csv
                    hk_fiber_list,                      # ex. ['sci', 'sky']
                    output_data,                        # lev1 containing Ca H&K extraction
                    output_exts=hk_spec_ext,            # ex. ['CA_HK_SCI', 'CA_HK_SKY']
                    dark=hk_dark_data,                  # lev0 containing dark image
                    wave_files=hk_wavelength_tables)    # ex. ['/data/masters/kpfMaster_HKwave20220909_sci.csv',
                                                        #      '/data/masters/kpfMaster_HKwave20220909_sky.csv']
            :

        where `output_data` is KPF1 object wrapped in `Arguments` class object.
"""

import configparser

# Pipeline dependencies
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.ca_hk.src.alg import CaHKAlg
import numpy as np
from os.path import exists
from astropy.io import fits

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/ca_hk/configs/default_hk.cfg'
CAHK_EXT = 'CA_HK'


class CaHKExtraction(KPF0_Primitive):
    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        # CA_HK data from level 0 data
        if isinstance(action.args[0], str):
            img = KPF0.from_fits(action.args[0])
        elif isinstance(action.args[0], KPF0):
            img = action.args[0]
        else:
            img = None
        self.input_img = img[CAHK_EXT] if hasattr(img, CAHK_EXT) else None

        # trace path
        self.trace_path = action.args[1]
        # fiber list
        if action.args[2] is not None and isinstance(action.args[2], list):
            self.fibers = action.args[2]
        elif action.args[2] is not None and isinstance(action.args[2], str):
            self.fibers = [action.args[2]]
        else:
            self.fibers = []

        # level 1 data instance for output, existing or not
        self.output_level1 = action.args[3]

        if "dark" in args_keys:
            if isinstance(action.args['dark'], str):
                img = KPF0.from_fits(action.args['dark'])
            elif isinstance(action.args['dark'], KPF0):
                img = action.args['dark']
            else:
                img = None
        else:
            img = None
        self.dark_img = img[CAHK_EXT] if hasattr(img, CAHK_EXT) else None

        if "bias" in args_keys:
            if isinstance(action.args['bias'], str):
                img = KPF0.from_fits(action.args['bias'])
            elif isinstance(action.args['bias'], KPF0):
                img = action.args['gias']
            else:
                img = None
        else:
            img = None
        self.bias_img = img[CAHK_EXT] if hasattr(img, CAHK_EXT) else None

        self.wave_table_files = []
        if 'wave_files' in args_keys:
            if isinstance(action.args['wave_files'], list):
                self.wave_table_files.extend(action.args['wave_files'])
            else:
                self.wave_table_files.append(action.args['wave_files'])

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['hk_spectral_extraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        self.output_exts = []
        if 'output_exts' in args_keys:
            if isinstance(action.args['output_exts'], list):
                self.output_exts.extend(action.args['output_exts'])
            elif isinstance(action.args['output_exts'], str):
                self.output_exts.append(action.args['output_exts'])

        self.output_wave_exts = []
        if 'output_wave_exts' in args_keys:
            if isinstance(action.args['output_wave_exts'], list):
                self.output_wave_exts.extend(action.args['output_wave_exts'])
            elif isinstance(action.args['output_wave_exts'], str):
                self.output_wave_exts.append(action.args['output_wave_exts'])

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        try:
            # no trace_path, no ca_hk data, dark image and bias image size not matches that of ca_hk data
            if (self.trace_path is None) or not exists(self.trace_path) or \
                (self.input_img is not None and self.input_img.size == 0) or \
                    ((self.dark_img is not None) and
                     (self.input_img is not None) and
                     (np.shape(self.dark_img) != np.shape(self.input_img))) or \
                    ((self.bias_img is not None) and
                     (self.input_img is not None) and
                     (np.shape(self.bias_img) != np.shape(self.input_img))):
                self.alg = None
            else:
                self.alg = CaHKAlg(self.input_img, self.fibers,
                           output_exts =  self.output_exts,
                           output_wl_exts = self.output_wave_exts,
                           config=self.config, logger=self.logger)
        except KeyError:
            self.alg = None

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0

        return True

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform spectral extraction from Ca H&K image based on the trace location information

        Returns:
            KPF1 instance
        """
        if self.alg is None:
            self.logger.warning("CaHKExtraction: no CA_HK data or trace data or wrong dark/bg size")
            return Arguments(None)

        # load trace location data
        self.alg.load_trace_location(self.trace_path)

        fibers = self.alg.get_fibers()

        self.output_exts = self.alg.get_output_exts()
        self.output_wave_exts = self.alg.get_wavelength_exts()

        result, msg = self.alg.img_subtraction(self.dark_img, self.bias_img)
        if not result and self.logger:
            self.logger.warning("CaHKExtraction: dark/bias subtraction error: "+msg)
            return Arguments(None)

        self.alg.img_scaling()

        for idx, fiber in enumerate(fibers):
            df_ext_result = self.alg.extract_spectrum(fiber)

            data_df = df_ext_result['spectral_extraction_result']
            assert data_df is not None, df_ext_result['message']

            self.output_level1 = self.construct_level1_data(data_df, self.output_exts[idx], self.output_level1)

            if len(self.wave_table_files) > idx:
                self.build_wavelength_ext(self.wave_table_files[idx], fiber,
                                          self.output_wave_exts[idx], self.output_level1)

        self.output_level1.receipt_add_entry('CaHkExtraction', self.__module__,
                                             f'CA_HK extraction to extensions {" ".join(self.output_exts)}', 'PASS')

        if self.logger:
            self.logger.warning("CaHkExtraction: Receipt written")

        if self.logger:
            self.logger.warning("CaHkExtraction: Done!")

        return Arguments(self.output_level1)

    @staticmethod
    def construct_level1_data(ext_result, ext_name, output_level1):
        if output_level1 is not None:
            kpf1_obj = output_level1
        else:
            kpf1_obj = KPF1()

        if ext_result is not None:
            total_order, width = np.shape(ext_result.values)
        else:
            total_order = 0

        # if no data in ext_result, not build data extension and the associated header
        if total_order > 0:
            data_ext_name = ext_name.upper()

            # data = op_result.values
            kpf1_obj[data_ext_name] = ext_result.values

            for att in ext_result.attrs:
                kpf1_obj.header[data_ext_name][att] = ext_result.attrs[att]

        return kpf1_obj

    def build_wavelength_ext(self, wave_file, fiber, wave_ext, out_lev1):
        wave_table = self.alg.load_wavelength_table(wave_file, fiber)
        if wave_table is not None:
            out_lev1[wave_ext] = wave_table
            if self.logger:
                self.logger.warning("CaHkExtraction: write wls to "+wave_ext)
