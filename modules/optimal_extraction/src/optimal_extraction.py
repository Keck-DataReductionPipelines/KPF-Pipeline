# Standard dependencies
"""
    This module defines class OptimalExtraction which inherits from `KPF0_Primitive` and provides methods to perform
    the event on optimal extraction in the recipe.

    Attributes:
        OptimalExtraction

    Description:
        * Method `__init__`:

            OptimalExtraction constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OptimalExtraction` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing spectrum data for
                      optimal extraction.
                    - `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing flat data and order
                      trace result.
                    - `action.args[2] (str)`:  File path for the file containing optimal extraction results in the
                      terms of level 1 data.
                    - `action.args['order_name'] (str, optional)`: Name of the order to be processed.
                      Defaults to 'SCI1'.
                    - `action.args['start_order'] (int, optional)`: Index of the first order to be processed.
                      Defaults to 0.
                    - `action.args['max_result_order']: (int, optional)`: Total orders to be processed, Defaults to -1.
                    - `action.args['rectification_method']: (str, optional)`: Rectification method, '`norect`',
                      '`vertial`', or '`normal`', to rectify the curved order trace. Defaults to '`norect`',
                      meaning no rectification.
                    - `action.args['wavecal_fits']: (str, optional)`: Path of the fits file containing wavelength
                      calibration data. Defaults to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of optimal extraction in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input_spectrum (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`, assigned by `actions.args[0]`.
                - `input_flat (kpfpipe.models.level0.KPF0):  Instance of `KPF0`, assigned by `actions.args[1]`.
                - `order_name (str)`: Name of the order to be processed.
                - `start_order (int)`: Index of the first order to be processed.
                - `max_result_order (int)`: Total orders to be processed.
                - `rectification_method (int)`: Rectification method code as defined in `OptimalExtractionAlg`.
                - `wavecal_fits (str)`: Path of the fits file with wavelength calibration data.
                - `config_path (str)`: Path of config file for optimal extraction.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg (modules.order_trace.src.alg.OptimalExtractionAlg)`: Instance of `OptimalExtractionAlg` which
                  has operation codes for the computation of optimal extraction.


        * Method `__perform`:

            OptimalExtraction returns the result in `Arguments` object which contains a level 1 data object (`KPF1`)
            with the optimal extraction results and the wavelength data tentatively transported from
            `action.args['wavecal_fits']` if there is.

    Usage:
        For the recipe, the optimal extraction event is issued like::

            :
            lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
            op_data = OptimalExtraction(lev0_data, lev0_flat_data,
                                        output_lev1_file, order_name=order_name,
                                        rectification_method=rect_method,
                                        wavecal_fits=input_lev1_file)
            :
"""


import configparser
import pandas as pd
import numpy as np

# Pipeline dependencies
# from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.optimal_extraction.src.alg import OptimalExtractionAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class OptimalExtraction(KPF0_Primitive):

    OrderMap = {
                    'PARAS': {
                            'total_order_name': 1, 'first_order': {'SCI1': 0}
                    },
                    'NEID': {
                            'total_order_name': 2, 'first_order': {'SCI1': 0}, 'wave_start_order': {'SCI1': 7}
                    }
                }
    default_agrs_val = {
                    'order_name': 'SCI1',
                    'max_result_order': -1,
                    'start_order': 0,
                    'rectification_method': 'norect',  # 'norect', 'normal', 'vertical'
                    'wavecal_fits': None
                }

    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        # action.args[0] is for level 0 fits
        # action.args[1] is for level 0 flat with order trace result extension
        self.input_spectrum = action.args[0]
        self.input_flat = action.args[1]
        self.output_data_file = action.args[2]

        self.order_name = self.get_args_value('order_name', action.args, args_keys)
        self.max_result_order = self.get_args_value("max_result_order", action.args, args_keys)
        self.start_order = self.get_args_value("start_order", action.args, args_keys)
        self.rectification_method = self.get_args_value("rectification_method", action.args, args_keys)
        self.wavecal_fits = self.get_args_value('wavecal_fits', action.args, args_keys)

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['optimal_extraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        self.alg = OptimalExtractionAlg(self.input_flat.data, self.input_spectrum.data,
                                        self.input_spectrum.header['DATA'],
                                        self.input_flat.extension['ORDER_TRACE_RESULT'],
                                        self.input_flat.header['ORDER_TRACE_RESULT'],
                                        config=self.config, logger=self.logger)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input_flat, KPF0) and isinstance(self.input_spectrum, KPF0) and \
            'ORDER_TRACE_RESULT' in self.input_flat.extension

        return success

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform optimal extraction by calling method `extract_spectrum` from OptimalExtractionAlg and create an instance
        of level 1 data (KPF1) to contain the analysis result.

        Returns:
            Level 1 data containing optimal extraction result.

        """
        # rectification_method: OptimalExtractAlg.NoRECT(fastest) OptimalExtractAlg.VERTICAL, OptimalExtractAlg.NORMAL
        # extraction_method: 'optimal' (default), 'sum'

        if self.logger:
            self.logger.info("OptimalExtraction: rectifying and extracting order...")

        ins = self.alg.get_instrument().upper()
        assert(ins in self.OrderMap.keys())

        order_info = self.OrderMap[ins]
        total_order_name = order_info['total_order_name'] if 'total_order_name' in order_info else 1
        total_spectrum_order = self.alg.get_spectrum_order()
        assert(self.order_name in order_info['first_order'])

        o_set = np.arange(order_info['first_order'][self.order_name], total_spectrum_order, total_order_name, dtype=int)
        s_order = self.start_order if self.start_order is not None else 0
        e_order = min((s_order + self.max_result_order), len(o_set)) \
            if (self.max_result_order is not None and self.max_result_order > 0) else len(o_set)
        o_set = o_set[s_order:e_order]
        opt_ext_result = self.alg.extract_spectrum(rectification_method=self.rectification_method, order_set=o_set)

        assert('optimal_extraction_result' in opt_ext_result and
               isinstance(opt_ext_result['optimal_extraction_result'], pd.DataFrame))

        data_df = opt_ext_result['optimal_extraction_result']

        kpf1_sample = KPF1.from_fits(self.wavecal_fits, ins) if self.wavecal_fits is not None else None
        kpf1_obj = self.construct_level1_data(data_df, ins, kpf1_sample)
        self.add_wavecal_to_level1_data(kpf1_obj, ins, kpf1_sample)

        kpf1_obj.receipt_add_entry('OptimalExtraction', self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("OptimalExtraction: Receipt written")

        if self.logger:
            self.logger.info("OptimalExtraction: Done!")

        return Arguments(kpf1_obj)

    def add_wavecal_to_level1_data(self, level1_obj: KPF1, ins: str, level1_sample: KPF1):
        if level1_sample is None or not self.order_name in level1_obj.data:
            return False

        s, total_order, width = np.shape(level1_obj.data[self.order_name])
        if s != 3:
            return False

        wave_data = None
        wave_header = None

        kpf1_wave = KPF1.from_fits(self.wavecal_fits, ins)
        if self.order_name in kpf1_wave.data.keys() and level1_sample.data[self.order_name] is not None:
            wave_data = kpf1_wave.data[self.order_name][1, :, :]
            wave_header = kpf1_wave.header[self.order_name + '_WAVE']

        level1_obj.header[self.order_name+'_WAVE'] = {}

        if wave_data is not None and wave_header is not None:
            # wavecal header is empty or data is empty
            order_info = self.OrderMap[ins]
            wave_start = order_info['wave_start_order'][self.order_name] \
                if self.order_name in order_info['wave_start_order'] else 0
            wave_end = min(wave_start + total_order, np.shape(wave_data)[0])
            level1_obj.data[self.order_name][1, 0:(wave_end-wave_start), :] = wave_data[wave_start:wave_end, :]
            level1_obj.header[self.order_name+'_WAVE'] = wave_header
            return True
        else:
            return False

    def construct_level1_data(self, op_result, ins, level1_sample: KPF1):
        total_order, width = np.shape(op_result.values)

        sample_primary_header = None
        if level1_sample is not None:
            sample_primary_header = level1_sample.header['PRIMARY']

        # if os.path.isfile(self.output_data_file):       # filename of level 1 fits
        #    kpf1_obj = KPF1.from_fits(self.output_data_file, 'KPF')
        # else:
        # make brand new file
        kpf1_obj = KPF1()

        kpf1_obj.data[self.order_name] = np.zeros((3, total_order, width))
        kpf1_obj.header[self.order_name+'_FLUX'] = {att: op_result.attrs[att] for att in op_result.attrs}
        kpf1_obj.header[self.order_name+'_VARIANCE'] = {}
        kpf1_obj.header[self.order_name+'_WAVE'] = {}

        kpf1_obj.data[self.order_name][0, :, :] = op_result.values
        if ins == 'NEID' and sample_primary_header is not None:
            for h_key in ['SSBZ100', 'SSBJD100']:
                kpf1_obj.header[self.order_name + '_FLUX'][h_key] = sample_primary_header[h_key]

        return kpf1_obj

    def get_args_value(self, key: str, args: Arguments, args_keys: list):
        v = None
        if key in args_keys and args[key] is not None:
            v = args[key]
        else:
            v = self.default_agrs_val[key]
        if key != 'rectification_method':
            return v

        method = OptimalExtractionAlg.NoRECT

        if v is not None and isinstance(v, str):
            if v.lower() == 'normal':
                method = OptimalExtractionAlg.NORMAL
            elif v.lower() == 'vertical':
                method = OptimalExtractionAlg.VERTICAL

        return method
