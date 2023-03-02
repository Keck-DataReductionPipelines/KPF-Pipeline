# Standard dependencies
"""
    This module defines class BarycentricCorrection which inherits from `KPF_Primitive` and provides methods to perform
    the event on barycentric correction in the recipe.

    Description:
        * Method `__init__`:

            BarycentricCorrection constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `BarycentricCorrection` event issued in the recipe:

                    - `action.args['bc_config'] (dict)`: Instance of dict which contains the observation configuration
                      for Barycentric correction.
                    - `action.args['start_time'] (str | float)`: Starting time in yyyy-mm-dd or Julian Data format.
                      Defaults to None.
                    - `action.args['period'] (str | int)`: A period of days for Barycentric correction computation.
                      Default to None.
                    - `action.args['bc_corr_path'] (str)`: Path of file, a csv file storing a list of redshift values
                      from Barycentric correction computation over a period of time. Default to None.
                    - `actions.args['bc_corr_output'] (str)`: Path of output file, a csv file. Default to None.
                    - `actions.args['bc_result'] (dict)`: Result of redshift values from module of BC correction
                      Default to None. This contains the maximum and minimum redshift values to be set to the header of
                      level 1 data if there is.
                    - `actions.args['dataset'] (list)`: List of data in level 1 data model. Default to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of Barycentric correction.

            and the following attributes are defined to initialize the object,

                -  `bc_config (dict)`:  Instance of dict which contains the observation configuration
                   for Barycentric correction, assigned by `action.args['bc_config']`.
                - `start_mjd (float)`: Starting day for Barycentric correction computation, assigned by
                  `action.args['start_mjd']`.
                - `period (int)`:  Period of days for Barycentric correction computation, assigned by
                  `actions.args['period']`.
                - `data_path(str)`: Path of the csv file storing redshift numbers, assigned by
                  `actions.args['bc_corr_path']`.
                - `data_output_path(str)`: Path of output csv file, assigned by `actions.args['bc_corr_output']`.
                - `config_path (str)`: Path of config file for Barycentric correction.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `bc_data (dict)`: Result of redshift values from Barycentric correction computation,
                  assigned by `actions.args['bc_result']`.
                - `dataset (list)`: List of KPF1 data, assigned by actions.args['dataset'].
                - `alg (modules.barycentric_correction.src.alg_barycentric_corr.BarycentricCorrectionAlg)`:
                  Instance of `BarycentricCorrectionAlg` which has operation codes for Barycentric correction
                  computation.

                The observation configuration could be accessed from either `bc_config` or `config`.


        * Method `__perform`:

            BarycentricCorrection returns the result in `Arguments` object which contains a list of redshift values,
            maximum redshift value, minimum redshift value over a period of days.

    Usage:
        For the recipe, the Barycentric correction event is issued like::

            :
            op_data = BarycentricCorrection(start_time='2458591.5', period=380, bc_corr_path=KPF_TEST_DATA)
            :
"""

import configparser

# Pipeline dependencies
# from kpfpipe.logger import start_logger
from kpfpipe.primitives.core import KPF_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from astropy.time import Time
# Local dependencies
from modules.barycentric_correction.src.alg_barycentric_corr import BarycentricCorrectionAlg

# Global read-only variables
DEFAULT_CFG_PATH ='modules/barycentric_correction/configs/default.cfg'
MAXBC = 'maxbc'
MINBC = 'minbc'
BCList = 'bc_list'


class BarycentricCorrection(KPF_Primitive):
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        # action.args['start_jd'] is for start time in julian data format
        # action.args['period'] is the period to compute
        # action.args['bc_corr_path'] is the path of the file storing barycentric correction
        self.bc_config = action.args['bc_config'] if 'bc_config' in args_keys else None
        st = action.args['start_time'] if 'start_time' in args_keys else None
        if st is not None:
            if isinstance(st, int) or isinstance(st, float):
                st = float(st)
            else:
                try:
                    st = Time(st).jd
                except:
                    st = None
        self.start_jd = st

        pd = action.args['period'] if 'period' in args_keys else None
        if pd is not None:
            try:
                pd = int(float(pd))
            except:
                pd = None

        self.period = pd if pd is not None else 1
        self.data_path = action.args['bc_corr_path'] if 'bc_corr_path' in args_keys else None
        self.data_output_path = action.args['bc_corr_output'] if 'bc_corr_output' in args_keys else None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['barycentric_correction']
        except:
            self.config_path = DEFAULT_CFG_PATH

        if self.config_path:
            self.config.read(self.config_path)
        else:
            self.config = None

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        self.kpf_dataset = None
        dataset = action.args['dataset'] if 'dataset' in args_keys else None

        if dataset is not None:
            if not isinstance(dataset, list):
                dataset = [dataset]
            if all([isinstance(d, KPF1) for d in dataset]):
                self.kpf_dataset = dataset

        self.bc_data = action.args['bc_result'] if 'bc_result' in args_keys else None
        # Order trace algorithm setup

        if self.bc_data is None or all([(k in self.bc_data and self.bc_data[k] is not None) for k in [MAXBC, MINBC]]):
            self.alg = BarycentricCorrectionAlg(self.bc_config,
                                            config=self.config, logger=self.logger)
        else:
            self.alg = None

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = self.bc_config is None or isinstance(self.bc_config, dict)
        return success

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform barycentric correction by BarycentricCorrectionAlg
        Returns:
            redshift values, maximum redshift and minimum redshift over the period

        """

        if self.logger:
            self.logger.info("BarycentricCorrectionAlg: starting...")

        if self.alg is not None:
            bc_min_max = self.alg.get_zb_long(self.start_jd, self.period, data_path=self.data_path,
                                       save_to_path = self.data_output_path)
            zb_list = self.alg.get_zb_list(self.start_jd, self.period, data_path=self.data_path,
                                       save_to_path = self.data_output_path)

            if self.logger:
                self.logger.info("BarycentricCorrection: Done!")

            bc_data = {"bc_list": zb_list,  "maxbc": bc_min_max[1], "minbc": bc_min_max[0]}
        else:
            bc_data = self.bc_data
            if self.logger:
                self.logger.info("BarycentricCorrection: get data calculated earlier!")

        if bc_data is not None and self.kpf_dataset is not None:
            for d in self.kpf_dataset:
                d.header['PRIMARY']['MAXBC'] = bc_data[MAXBC]
                d.header['PRIMARY']['MINBC'] = bc_data[MINBC]
            if self.logger:
                self.logger.info("BarycentricCorrection: Set max and min BC to header!")

        return Arguments(bc_data)

