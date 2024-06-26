
import os

import logging
import configparser as cp

def get_level(lvl:str) -> int:
    '''
    read the logging level (string) from config file and return 
    the corresponding logging level (technically of type int)
    '''
    if lvl == 'debug': return logging.DEBUG
    elif lvl == 'info': return logging.INFO
    elif lvl == 'warning': return logging.WARNING
    elif lvl == 'error': return logging.ERROR
    elif lvl == 'critical': return logging.CRITICAL
    else: return logging.NOTSET

def start_logger(logger_name: str, config: str, log_path=None) -> logging.Logger:
    '''

    Args:
        logger_name (str): name of primitive, which will be shown in each log msg. 
        config (str): path to configuration file
        log_path (str): (optional) path to the logfile if different than whats defined in the configuration file
    '''
    # start a logger instance:
    # if logger_name in logging.Logger.manager.loggerDict.keys():
    #     logger =  logging.Logger.manager.loggerDict[logger_name]
    #     print(logger_name)
    # else: 
    #     logger = logging.getLogger(logger_name)

    if config is None: 
        # a config file is not provided, so don't start logger
        print('[{}] missing log configuration...not starting a new logger'.format(
            logger_name))
        return None
    config_obj = cp.ConfigParser()
    res = config_obj.read(config)
    if res == []:
        # thsi will occur if configuration failed to read
        raise IOError('failed to read {}'.format(config))

    log_cfg = config_obj['LOGGER']

    log_start = log_cfg.get('start_log', False)
    if log_path is None:
        log_path = log_cfg.get('log_path', 'log')
        
    log_lvl = log_cfg.get('log_level', 'warning')
    log_verbose = log_cfg.getboolean('log_verbose', True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(get_level(log_lvl))
    logger.propagate = False

    formatter = logging.Formatter('[%(name)s][%(levelname)s]:%(message)s')
    if log_start:
        if log_path:
            f_handle = logging.FileHandler(log_path, mode='w')
            f_handle.setLevel(get_level(log_lvl))
            f_handle.setFormatter(formatter)
            logger.addHandler(f_handle)
        if log_verbose:
            s_handle = logging.StreamHandler()
            s_handle.setLevel(get_level(log_lvl))
            s_handle.setFormatter(formatter)
            logger.addHandler(s_handle)

    return logger