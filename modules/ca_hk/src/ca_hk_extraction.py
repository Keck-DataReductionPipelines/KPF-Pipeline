# Standard dependencies 
"""
    This module defines class `OrderTrace` which inherits from `KPF0_Primitive` and provides methods to perform the
    event on order trace calculation in the recipe.

    Attributes:
        CaHKExtraction

    Description:
        * Method `__init__`:

            CaHKExtraction constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderTrace` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0|str)`: Instance of `KPF0` or the path of a fits file
                      containing image data for H&K extraction.
                    - `action.args[1] (str)`: Path to a file defining the fiber and order location
                    - `action.args[2] (list)`: List contaiing the fiber names.
                    - `action.args[3] (kpfpipe.models.level1.KPF1)`:  Instance of `KPF1` containing spectral
                      extraction results. If not existing, it is None.
                    - `action.args['output_exts'] (str)`: Extension names of the extensions to contain
                      the extraction result for each fiber, optional, Defaults to the fiber list.
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
            hk_data = kpf0_from_fits(hk_file)
            output_lev1_hk = CaHKExtraction(hk_data, input_trace_file, fiber_list, None, output_exts=output_exts)
            :

            where 'hk_data' is level 0 data (`KPF0`) object containing 'CA_HK' extension,
            'input_trace_file' is the path of the csv file containing the location defintion of the order trace,
            'fiber_list' is a list containing the fibers to be processed, and
            'output_exts' is a list containing the names of the extensions to contain the output.

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


class CaHKExtraction(KPF0_Primitive):
    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        if isinstance(action.args[0], KPF0):
            self.input_img = action.args[0]['CA_HK']
        else:
            hdus = fits.open(action.args[0])
            self.input_img = hdus[0].data

        self.trace_path = action.args[1]
        if action.args[2] is not None and isinstance(action.args[2], list):
            self.fibers = action.args[2]
        elif self.fibers is not None and isinstance(action.args[2], str):
            self.fibers = [action.args[2]]
        else:
            self.fibers = []
        self.output_level1 = action.args[3]  # kpf1 instance already exist or None

        self.total_fibers = len(self.fibers)
        self.output_exts = []
        if 'output_exts' not in args_keys:
            self.output_exts.extend(self.fibers)
        elif isinstance(action.args['output_exts'], list):
            self.output_exts.extend(action.args['output_exts'])
        else:
            self.output_exts.append(action.args['output_exts'])

        if len(self.output_exts) < len(self.fibers):
            for idx in range(len(self.output_exts), len(self.fibers)):
                self.output_exts.append(self.fibers[idx])
        self.fiber_loc = None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['hk_spectral_extraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        self.alg = CaHKAlg(self.input_img,  self.fibers, config=self.config, logger=self.logger)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input_img, np.ndarray) and self.trace_path is not None and exists(self.trace_path)

        return success

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

        if self.logger:
            self.logger.info("HKExtraction: define the trace location")

        # load trace location data
        trace_def = self.alg.load_trace_location(self.trace_path)

        for idx, fiber in enumerate(self.fibers):
            df_ext_result = self.alg.extract_spectrum(fiber)

            data_df = df_ext_result['spectral_extraction_result']
            assert data_df is not None, df_ext_result['message']

            self.output_level1 = self.construct_level1_data(data_df, self.output_exts[idx], self.output_level1)

        self.output_level1.receipt_add_entry('CaHkExtraction', self.__module__,
                                             f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("CaHkExtraction: Receipt written")

        if self.logger:
            self.logger.info("CaHkExtraction: Done!")

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
