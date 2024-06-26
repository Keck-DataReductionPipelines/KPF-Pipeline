# Standard dependencies
"""
    This module defines class `RadialVelocityInit` which inherits from `KPF_Primitive` and provides methods
    to perform the event on radial velocity initial setting in the recipe.

    Description:
        * Method `__init__`:

            RadialVelocityInit constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocityInit` event issued in the recipe:

                    - `action.args['start_time'] (str | float)`: Starting time in yyyy-mm-dd or Julian Data format.
                      Defaults to None.
                    - `action.args['period'] (str | int)`: A period of days for Barycentric correction computation.
                      Default to None.
                    - `action.args['l1_data'] (KPF1)`: KPF1 instance. The primary header of this L1 object provides
                      star configuration information for the initial setting.
                    - `action.args['bc_corr_path'] (str)`: Path of the csv file storing a list of Barycentric
                      correction related data over a period of time. Default to None.
                    - `actions.args['test_data_path'] (str)`: Path of the star config file. Default to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and following attributes are defined to initialize the object,

                - `bc_period (float)`: Period for Barycentric velocity correction calculation.
                - `bc_start_jd (float)`: Start time in Julian data format for Barycentric velocity correction calculation.
                - `bc_data (str)`: Path of csv file storing barycentric correction related data for a period of time.
                - `test_data (str)`: Path of the star config file.
                - `bc_output_data (str)`: Path of csv output file storing the result from barycentric correction computation.
                - `l1_data (KPF1)`: L1 instance containing star configuration information.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg_rv_init (RadialVelocityAlgInit)`: Instance of `RadialVelocityAlgInit` which has operation codes
                  for radial velocity initial setting.

        * Method `__perform`:

            RadialVelocityInit returns the result in `Arguments` object which contains the initialization result
            including status, error message, and the data. Please refer to `Returns` section
            in :func:`modules.radial_velocity.src.alg_rv_init.RadialVelocityAlgInit.start()`
            for the detail of the result.

    Usage:
        For the recipe, the optimal extraction init event is issued like::

            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_init = RadialVelocityInit(start_time="2021-03-01",
                    l1_data=lev1_data,
                    bc_corr_path='/data/bary/',
                    test_data_path='/data/masters/'
                    )
            :

            rv_data = RadialVelocity(lev1_data, rv_init, ...)
            :

        where `rv_data` is dict object wrapped in `Arguments` class object.
"""

import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.core import KPF_Primitive
from astropy.time import Time

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
import os

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/radial_velocity/configs/default.cfg'


class RadialVelocityInit(KPF_Primitive):

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        """
        Example KPF module
        """
        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        st = action.args['start_time'] if 'start_time' in args_keys else None
        if st is not None:
            if isinstance(st, int) or isinstance(st, float):
                st = float(st)
            else:
                try:
                    st = Time(st).jd
                except:
                    st = None
        self.bc_start_jd = st if st is not None else Time("2019-04-18").jd

        pd = action.args['period'] if 'period' in args_keys else None
        if pd is not None:
            try:
                pd = int(float(pd))
            except:
                pd = None

        self.bc_period = pd if pd is not None else 380
        self.test_data = action.args['test_data_path'] if 'test_data_path' in args_keys \
            else (os.getenv('KPFPIPE_TEST_DATA') + '/')
        # barycentric correction default period: 380 day, start date: apr-18-2019
        self.bc_data = action.args['bc_corr_path'] if 'bc_corr_path' in args_keys else None
        self.bc_output_data = None
        self.l1_data = action.args['l1_data'] if 'l1_data' in args_keys else None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['radial_velocity']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config form: {}'.format(self.config_path))
        # Order trace algorithm setup

        self.alg_rv_init = RadialVelocityAlgInit(self.config, self.logger,
                                                 l1_data = self.l1_data,
                                                 bc_time=self.bc_start_jd, bc_period=self.bc_period,
                                                 bc_corr_path = self.bc_data,
                                                 test_data=self.test_data)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0

        return True

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform radial velocity init by call method 'start' from RadialVelocityAlgInit.

        Returns:
            Init result including status, error message if the status is false and the data from init. Please refer to
            `Returns` section of :func:`~alg_rv_init.RadialVelocityAlgInit.start()`
        """

        if self.logger:
            self.logger.info("RadialVelocityInit: Start RV init ")

        init_result = self.alg_rv_init.start()

        if init_result['status'] and 'data' in init_result:
            if self.logger:
                self.logger.info("RadialVelocityInit: Init for radial velocity is done")

            return Arguments(init_result)
        else:
            if self.logger:
                self.logger.info("RadialVelocityInit: Init for radial velocity fails - " + init_result['msg'])

            return Arguments(None)



