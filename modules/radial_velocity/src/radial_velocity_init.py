# Standard dependencies
"""
    This module defines class `RadialVelocityInit` which inherits from `KPF_Primitive` and provides methods
    to perform the event on radial velocity initial setting in the recipe.

    Attributes:
        RadialVelocityInit


    Description:
        * Method `__init__`:

            RadialVelocityInit constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: no arguments are passed from
                  `RadialVelocityInit` event issued in the recipe.
                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and following attributes are defined to initialize the object,

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

            rv_init = RadialVelocityInit()
            :
            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_data = RadialVelocity(lev1_data, rv_init, order_name=order_name)
            :

    """

import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/radial_velocity/configs/default.cfg'


class RadialVelocityInit(KPF_Primitive):
    """ Radial velocity init primitive

    This module defines class RadialVelocityInit and methods to perform the initial setting for the radial velocity
    computation.

    Attributes:
        config_path (str): Path of config file for order trace.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.
        alg_rv_init (RadialVelocityAlgInit): Instance of RadialVelocityAlgInit.

    """
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        """
        Example KPF module
        """
        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        # start a logger
        # self.logger = start_logger(self.__class__.__name__, None)

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path('radial_velocity_init')
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config form: {}'.format(self.config_path))
        # Order trace algorithm setup
        self.alg_rv_init = RadialVelocityAlgInit(self.config, self.logger)

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
        init_result = self.alg_rv_init.start()

        assert(init_result['status'] and 'data' in init_result)

        if self.logger:
            self.logger.info("RadialVelocityInit: Init for radial velocity is done")

        return Arguments(init_result)


