import configparser
import sys
import os
import time
import signal
import subprocess

import logging
from datetime import datetime


def get_pgpass(dbname, dbuser):
    """
    Read the credentials for the psql database from the .pgpass file

    Args:
        dbname (str): the name of the database to match
        dbuser (str): the name of the user to match

    Returns: <str, str> port, password
    """
    pgpass_path = os.path.expanduser('~/.pgpass')
    try:
        with open(pgpass_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"The .pgpass file does not exist at {pgpass_path}")
        return None, None

    for line in lines:
        # hostname:port:database:username:password
        parts = line.strip().split(':')
        if len(parts) != 5:
            continue

        hostname, port, database, username, dbpass = parts

        if database == dbname and username == dbuser:
            return port, dbpass

    return None, None


def chk_rm_docker_container(container_name, log):
    """
    Check if a docker container with the same name is running,  if so,  stop
    and remove it.

    Args:
        container_name (str): the string name of the container.
        log (obj): the log object

    Returns:
    """
    result = subprocess.run(f"docker ps -a -q -f name={container_name}",
                            shell=True, capture_output=True, text=True)
    container_id = result.stdout.strip()

    if container_id:
        log.info(f"Stop / remove container '{container_name}' (ID: {container_id}).")
        subprocess.run(f"docker stop {container_id}", shell=True)
        subprocess.run(f"docker rm {container_id}", shell=True)
        log.info(f"Container '{container_name}' has been stopped / removed.")


def get_log(log_name):
    """
    Access to the log object.

    Args:
        log_name (str): the name of the log to return

    Returns: <obj) the log object if found.

    """

    # get the log if already exists
    log = logging.getLogger(log_name)
    if not log.handlers:
        print('an error occurred while getting the log')
        return None

    return log


class MessageFormatter(logging.Formatter):
    """
    Custom formatter that shows `funcName` only if the log level is higher than INFO.
    """

    def format(self, record):
        # Set format based on log level
        if record.levelno > logging.INFO:
            self._fmt = '%(asctime)s [%(levelname)s] %(funcName)s - %(message)s'
        else:
            self._fmt = '%(asctime)s [%(levelname)s] - %(message)s'

        # Initialize the style with the updated format
        self._style = logging.PercentStyle(self._fmt)

        return super().format(record)


def configure_logger(log_dir, log_name):
    """
    Set up the logger.

    Args:
        log_dir (str): The directory of the log.
        log_name (str): The name of the log.

    Returns:
        logging.Logger: Configured logger object.
    """

    # Get the log if it already exists
    log = logging.getLogger(log_name)
    if log.handlers:
        return log

    # Set up the logger
    log_path = f'{log_dir}/{log_name}.log'
    log.setLevel(logging.INFO)

    formatter = MessageFormatter()

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    log.info("Starting the KPF DRP logger.")

    return log


def cfg_init(app_path, config_name):
    """
    The API start up configuration.

    :param app_path: <str> the path that the api is running
    :param config_name: <str> name of the config file.

    :return: config as an object, and the initialized log.
    """
    config_filename = config_name

    config_file = f'{app_path}/{config_filename}'
    print(f"Reading configuration from: {config_file}")

    config = configparser.ConfigParser()
    config.read(config_file)

    return config


def get_cfg(config, section, param_name):
    """
    Function used to read the config file,  and exit if key or value does not
    exist.

    :param config: <class 'configparser.ConfigParser'> the config file parser.
    :param section: <str> the section name in the config file.
    :param param_name: <str> the 'key' of the parameter within the section.
    :return: <str> the config file value for the parameter.
    """
    try:
        param_val = config[section][param_name]
    except KeyError:
        err_msg = f"Check Config file, there is no parameter name - "
        err_msg += f"section: {section} parameter name: {param_name}"
        sys.exit(err_msg)

    if not param_val:
        err_msg = f"Check Config file, there is no value for "
        err_msg += f"section: {section} parameter name: {param_name}"
        sys.exit(err_msg)

    return param_val


def is_log_file_done(log_file, timeout=120):
    """
    Check if the log file has been updated recently.

    Args:
        log_file (str): the path (outside docker) to the log file.
        timeout (int): the number of seconds to allow the log to be idle.

    Returns: <bool> True if the log file has not bee updated in timeframe.

    """
    last_mod_time = os.path.getmtime(log_file)
    current_time = time.time()
    return (current_time - last_mod_time) > timeout


def sync_files(data_workspace, procdate, log):
    """
    Sync the files from koadata to the workspace used for processing the
    master files.

    Args:
        data_workspace (str): the directory to copy the files to.
        procdate (str): the processing date,  YYYYMMDD.
        log (obj): the log object.
    """
    try:
        log.info(f'rsyncing over files from /koadata/KPF/{procdate}/lev0/*.fits '
                 f'to {data_workspace}/L0/{procdate}/')
        subprocess.run(['rsync', '-av', f'/koadata/KPF/{procdate}/lev0/*.fits',
                        f'{data_workspace}/L0/{procdate}/'])
        log.info('rsync complete')
    except subprocess.CalledProcessError as err:
        log.error(f'Error rsyncing files: {err}')


def mk_output_dirs(data_workspace, masters_perm_dir, procdate):
    """
    Create the output directories for the processing date.

    Args:
        data_workspace (str): the directory to copy the files to.
        masters_perm_dir (str): the masters final resting place.
        procdate (str): the processing date,  YYYYMMDD.
    """
    os.makedirs(f"{masters_perm_dir}/{procdate}", exist_ok=True)
    os.makedirs(f"{data_workspace}/L0/{procdate}", exist_ok=True)
    os.makedirs(f"{data_workspace}/2D/{procdate}", exist_ok=True)
    os.makedirs(f"{data_workspace}/masters/{procdate}", exist_ok=True)
    os.makedirs(f"{data_workspace}/logs/{procdate}", exist_ok=True)


def log_stub(prefix, proc_type, procdate, log):
    """
    Write a log stub for starting and ending of the log.

    Args:
        log (obj): the logger object/
        proc_type (str): the processing type that is running.
        procdate (str): the processing date.

    Returns:

    """
    msg = f"{prefix} Keck {proc_type} reduction for {procdate}"
    msg_wrap = "=" * len(msg)
    log.info(msg_wrap)
    log.info(msg)
    log.info(msg_wrap)


def get_dated_cfg(procdate, cfg_dir, cfg_prefix):
    """
    Find the dated configuration files that are closed in the past,  any
    configurations with a later date will be newer than the data and be
    used when there are instrument changes,  ie service or shifts in the
    calibrations.

    The configuration file date is the most recent FITS (or other calibration)
    file used within the file.

    Args:
        cfg_dir (str): The directory to search for the configuration files.
        cfg_prefix (str): the file prefix,  ie keck_kpf_drp_ would be used to
                          find keck_kpf_drp_20241120.cfg

    Returns (str): The filename (including cfg_dir) that is the closest match.
    """
    cfg_suffix = ".cfg"

    # add the '_' between prefix and date
    if cfg_prefix[-1] != '_':
        cfg_prefix += '_'

    len_prefix = len(cfg_prefix)
    len_suffix = len(cfg_suffix)
    today_dt = datetime.strptime(procdate, "%Y%m%d")

    # find the file is closest but in the past
    closest_file = None
    closest_date = None

    # Iterate over all files in the directory
    for file in os.listdir(cfg_dir):
        
        # ignore files that don't match
        if not file.startswith(cfg_prefix) or not file.endswith(cfg_suffix):
            continue
            
        # get the date from the filename, fmt: 20241120
        date_str = file[len_prefix:-len_suffix]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < today_dt and (
                    closest_date is None or file_date > closest_date):
                closest_file = os.path.join(cfg_dir, file)
                closest_date = file_date
        except ValueError:
            # invalid date formats
            continue

    return closest_file


def kill_exist_proc(script_name, log=None):
    """
    Kill an existing process by the name of the script.

    Args:
        script_name ():
        log (): the logger object

    Returns:

    """
    def get_cmdline_info(pid):
        try:
            with open(f"/proc/{pid}/cmdline", "r") as f:
                return f.read().split("\0")
        except Exception as e:
            return []

    current_pid = os.getpid()
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                # don't commit Hari-Kari
                if int(pid) == current_pid:
                    continue

                cmdline = get_cmdline_info(pid)
                if cmdline:
                    if any(script_name in arg for arg in cmdline):
                        msg = f"Killing process '{script_name}' with PID {pid}"
                        if log:
                            log.info(msg)
                        else:
                            print(msg)

                        os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as err:
                pass
