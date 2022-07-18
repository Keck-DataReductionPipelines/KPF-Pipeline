from modules.Utils.config_parser import ConfigHandler
import time
import logging
from kpfpipe.logger import *


class ModuleAlgBase:
    """Order trace extraction.

    This module defines class 'ModuleAlgBase' and methods to output the message including log the running status,
    debug information, or time profiling.
    The message could be output to
        - external logger, (self.logger)
        - the logger defined in module config, (self.config_logger), or
        - a text file defined by function add_file_logger (self.debug_output).

    Args:
        logger_name (str): Name for an instance logging.Logger created for the instance of the derived class with
            the methods related to some algorithm topic.
        config (configparser.ConfigParser): config context.
        logger (logging.Logger): Instance of logging.Logger, may be passed from the external application.

    Attributes:
        logger (logging.Logger): Instance of logging.Logger, assigned by argument logger.
        config_param (ConfigHandler): Related to 'PARAM' section define in config file.
        config_logger (logging.Logger): Related to 'LOGGER' section defined in the config file.

        debug_output (str): File path for the file that the debug information is printed to. The printing goes to
            standard output if it is an empty string or no printing is made if it is None.
        is_time_profile (bool): Print out the time profiled message to the external logger in terms of info level
            logging message.
        is_debug (bool): Print out the message to the provided external logger, config defined logger or text logging
            file in term of either info or debug level logging message.
    """

    def __init__(self, logger_name, config=None, logger=None):
        self.config_param = ConfigHandler(config, 'PARAM')      # per section of 'PARAM' in config
        self.config_logger = self.get_config_logger(config, logger_name)  # section 'LOGGER' in config or logger_name
        self.logger = logger                                    # external logger
        self.debug_output = None                                # if sending debug printing to a file or a terminal
        self.is_time_profile = False                            # if time profiling to config_logger or debug_output
        # if outputting the message to logger or config_logger
        self.is_debug = True if self.logger or self.config_logger else False

    def enable_debug_print(self, to_print=True):
        """ Enable or disable debug printing.

        Args:
            to_print (bool, optional): Print out the debug information of the execution
                per to_print or the existence of the external logger or the module defined logger.  Defaults to False.

        Returns:
            None.

        """
        self.is_debug = to_print or bool(self.logger) or bool(self.config_logger)

    def enable_time_profile(self, is_time=False):
        """Enable or disable time profiling printing.

        Args:
            is_time (bool, optional): Print out the time information while running. Defaults to False.

        Return:
            None.

        """

        self.is_time_profile = is_time

    def add_file_logger(self, filename=None):
        """ Add file to log debug information during the running.

        Print debug information
        1. to stdout if it is provided as empty string, or
        2. to a file with path `print_debug` if it is non empty string, or
        3. no print if it is None.  Defaults to None.
        This is used to provide a file to output the debug information mainly for testing and debugging purpose.
        It is different from the specified external logger, i.e. an instance of logging.Logger, which may be
        passed from the recipe primitive to log the running status.

        Args:
            filename (str, optional): Filename of the log file. Defaults to None.

        Returns:
            None.

        """
        self.enable_debug_print(filename is not None)
        self.debug_output = filename

    def time_check(self, t_start, step_msg):
        """Count and display the execution time.

        Args:
            t_start (float): Start time to count.
            step_msg (str): Message to print.

        Returns:
            float: End of time.

        """
        t_end = time.time()
        self.t_print(step_msg, (t_end - t_start), 'sec.')
        return t_end

    def d_print(self, *args, end='\n', info=False):
        """Print out running status or debug information to logger and/or debug information to a file.

        Args:
            *args: Variable length argument list to print.
            end (str, optional): Specify what to print at the end of *arg.
            info (bool): Print-out is for information level, not for debug level.

        Returns:
            None.

        Notes:
            This function handles the print-out to the logger defined in the config file or other file as specified in
            :func:`~alg.OrderTraceAlg.add_file_logger()`.

        """
        if self.is_debug:
            out_str = ' '.join([str(item) for item in args])
            if self.logger:
                if info:
                    self.logger.info(out_str)
                else:
                    self.logger.debug(out_str)

            if self.config_logger:
                if info:
                    self.config_logger.info(out_str)
                else:
                    self.config_logger.debug(out_str)

            if self.debug_output is not None and not info:
                if self.debug_output:
                    with open(self.debug_output, 'a') as f:
                        f.write(out_str + end)
                        f.close()
                else:
                    print(out_str, end=end)

    def t_print(self, *args):
        """Print time profiling information to config logger as info level message or debug output file.

        Args:
             *args: Variable length argument list to print.

        Returns:
            None.

        """

        if self.is_time_profile:
            out_str = ' '.join([str(item) for item in args])
            if self.config_logger:
                self.config_logger.info(out_str)
            if self.debug_output is not None:
                if self.debug_output:
                    with open(self.debug_output, 'a') as f:
                        f.write(out_str + '\n')
                        f.close()
                else:
                    print(out_str+'\n')

    @staticmethod
    def get_config_logger(config, logger_name):
        """Instance of logging.Logger associated with a config instance or a logger name.

        The creation of the Logger instance is based on config if config exists. If config is none, this call tries
        to return a reference to a Logger instance which may exist already based on the logger name.

        Args:
            config (configparser.ConfigParser): Config context.
            logger_name (str): logger name.

        Returns:
            logging.Logger: An instance of logging.Logger

        Notes: Using the same logger_name could make multiple class instances to share the same Logger and dump the
        message to the same file or stream handler.

        """
        logger_section = config['LOGGER'] if config and config.has_section('LOGGER') else None

        logger = None

        if logger_section is not None:
            log_start = logger_section.getboolean('start_log', False)

            if log_start:
                log_path = logger_section.get('log_path', None)
                log_mode = 'a' if logger_section.get('log_append', False) else 'w'
                log_lvl = logger_section.get('log_level', 'debug')
                log_verbose = logger_section.getboolean('log_verbose', True)

                logger = logging.getLogger(logger_name)
                logger.setLevel(get_level(log_lvl))
                f_handle = None
                s_handle = None
                if log_path or log_verbose:
                    for handler in logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            f_handle = handler
                        elif isinstance(handler, logging.StreamHandler):
                            s_handle = handler

                    formatter = logging.Formatter('[%(name)s][%(levelname)s]:%(message)s')

                    # set up a file handler
                    if log_path and f_handle is None:
                        f_handle = logging.FileHandler(log_path, mode=log_mode)  # logging to file
                        f_handle.setLevel(get_level(log_lvl))
                        f_handle.setFormatter(formatter)
                        logger.addHandler(f_handle)
                    elif not log_path and f_handle:
                        logger.removeHandler(f_handle)

                    # set up stream handler
                    if log_verbose and s_handle is None:
                        s_handle = logging.StreamHandler()
                        s_handle.setLevel(get_level(log_lvl))
                        s_handle.setFormatter(formatter)
                        logger.addHandler(s_handle)
                    elif not log_verbose and s_handle:
                        logger.removeHandler(s_handle)
                logger.propagate = False
        elif logger_name is not None:
            logger = logging.getLogger(logger_name)
        return logger

    @staticmethod
    def start_time():
        """Get current time for profiling.

        Returns:
            None.
        """
        return time.time()

    @staticmethod
    def get_level(lvl: str):
        """Convert logging level from string to numeric value.

        Read the logging level (string) from config file and return the corresponding logging level
        (technically of type int)

        Args:
            lvl (str): Level string, 'debug', 'info', 'warning', 'error', 'critical' or others.

        Returns:
            int: a logging level in numeric value.

        """

        if lvl == 'debug':
            return logging.DEBUG
        elif lvl == 'info':
            return logging.INFO
        elif lvl == 'warning':
            return logging.WARNING
        elif lvl == 'error':
            return logging.ERROR
        elif lvl == 'critical':
            return logging.CRITICAL
        else:
            return logging.NOTSET



