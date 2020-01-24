import logging



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

def start_logger(logger_name: str, log_config: dict) -> logging.Logger:

    log_path = log_config.get('log_path', None)
    log_lvl = log_config.get('log_level', None)
    log_verbose = log_config.getboolean('log_verbose', None)

    # basic logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(get_level(log_lvl))

    formatter = logging.Formatter('[%(name)s][%(levelname)s]:%(message)s')
    f_handle = logging.FileHandler(log_path, mode='w') # logging to file
    f_handle.setLevel(get_level(log_lvl))
    f_handle.setFormatter(formatter)
    logger.addHandler(f_handle)

    if log_verbose: 
        # also print to terminal 
        s_handle = logging.StreamHandler()
        s_handle.setLevel(get_level(log_lvl))
        s_handle.setFormatter(formatter)
        logger.addHandler(s_handle)
    return logger