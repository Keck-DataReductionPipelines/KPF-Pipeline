import configparser
from modules.Utils.config_parser import ConfigHandler
from modules.Utils.alg_base import ModuleAlgBase


class RadialVelocityBase(ModuleAlgBase):
    """Base class for Radial Velocity related classes.

    This module defines class 'RadialVelocityBase' and methods to do basic work for all Radial Velocity associated
    classes.

    Args:
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        instrument (str): Instrument name.
        config_ins (ConfigHandler): Related to 'PARAM' section or section associated with the instrument
            if it is defined in the config file.
    """

    name = "RadialVelocity"
    def __init__(self, config=None, logger=None):
        ModuleAlgBase.__init__(self, self.name,  config, logger)

        ins = self.config_param.get_config_value('instrument', '') if self.config_param is not None else ''
        self.instrument = ins.upper()
        self.config_ins = ConfigHandler(config, ins, self.config_param)  # section of instrument or 'PARAM'

    def get_instrument(self):
        return self.instrument.upper()
