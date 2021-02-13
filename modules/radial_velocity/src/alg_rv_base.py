import configparser
from modules.Utils.config_parser import ConfigHandler


class RadialVelocityBase:
    """Base class for Radial Velocity related classes.

    This module defines class 'RadialVelocityBase' and methods to do basic work for all Radial Velocity associated
    classes.

    Args:
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        logger (logging.Logger): From parameter `logger`.
        instrument (str): Instrument name.
        config_param (ConfigHandler): Instance representing the section defined in the configuration file.
        is_debug (bool): A flag indicating if doing the logging or displaying the debugging information.
        debug_output (str): File path of the file that the debug information is printed to. The printing goes to
            standard output if it is an  empty string or no printing is made if it is None.

    """

    def __init__(self, config=None, logger=None):
        self.logger = logger
        p_config = ConfigHandler(config, 'PARAM')
        self.instrument = p_config.get_config_value('instrument', '')
        ins = self.instrument.upper()
        self.config_param = ConfigHandler(config, ins, p_config)  # section of instrument or 'PARAM'

        self.is_debug = bool(self.logger)
        self.debug_output = None

    def enable_debug_print(self, to_print=True):
        """Enable or disable debug printing.

        Args:
            to_print (bool, optional): Enable or disable. Defaults to True.

        """
        self.is_debug = to_print or bool(self.logger)

    def d_print(self, *args, end='\n', info=False):
        """Print information to a logger or a file.

         Args:
             *args: Various length argument list. Content to be printed as the argument list passed to ``print()``.
             end (str): Ending text attached to the string formed by `*args`.
             info (bool): True if the printing is for execution information and False if the printing is for
                debug information.
                The execution information is logged to logger if there is.
                The debug information is logged to logger if there is and written to the file specified by
                attribute `debug_output` if there is.

         """
        if self.is_debug:
            out_str = ' '.join([str(item) for item in args])
            if self.logger:
                if info:
                    self.logger.info(out_str)
                else:
                    self.logger.debug(out_str)
            if self.debug_output is not None and not info:
                if self.debug_output:
                    with open(self.debug_output, 'a') as f:
                        f.write(' '.join([str(item) for item in args]) + end)
                        f.close()
                else:
                    print(out_str, end=end)

    def add_file_logger(self, filename=None):
        """Set file to log the debug information.

        The setting is mainly for printing out the debug information for the development.

        Args:
            filename (str): File path of the file.

        """
        self.enable_debug_print(filename is not None)
        self.debug_output = filename

    def get_instrument(self):
        return self.instrument.upper()
