"""
The base used for scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os
import sys
import time
import subprocess

import keck_utils as utils

from datetime import datetime, timezone

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeCronBase:
    """
    The base for KPF pipeline cronjobs
    """
    def __init__(self, procname):

        self.procname = procname
        self.default_ncpu = 232

        # will be defined later
        self.pid = None
        self.ncpu = None
        self.recipe = None
        self.config = None
        self.procdate = None
        self.logs_root = None
        self.dockerruncmd = None
        self.containerimage = None
        self.dockercmdscript = None
        self.logs_root_docker = None
        self.docker_bash_script = None

        # cfg file parameters
        self.dbuser = None
        self.dbname = None
        self.data_drp = None
        self.data_root = None
        self.logs_base = None
        self.kpfdrp_dir = None
        self.containername = None
        self.data_workspace = None
        self.masters_perm_dir = None
        self.reference_fits_dir = None

        # read the command line arguments
        self.read_cmd_line()

        # read the configuration
        self.read_cron_cfg()

        # initialize to make the container exit after X seconds.
        self.exit_timer = None
        self.wait_interval = 300

        # used to stop the docker container if the log is idle
        self.log_chk = None

        # create the output directories
        utils.mk_output_dirs(self.data_workspace, self.masters_perm_dir, self.procdate)

        # override to move the location elsewhere
        self.set_log_dir()
        self.log_name = f'keck_kpf_{procname}_{self.procdate}'
        os.makedirs(self.logs_root, exist_ok=True)

        self.log = utils.configure_logger(self.logs_root, f"{self.log_name}")

        self.dbport, self.dbpass = utils.get_pgpass(self.dbname, self.dbuser)

        # change to the code directory
        os.chdir(self.kpfdrp_dir)

        # prepare the std out file
        self.set_stdout_log()

        # log start
        utils.log_stub('Starting', f'{procname.title()}-Processing', self.procdate, self.log)

    def run(self):
        """
        Create the docker script,  container,  run it,  exit
        """
        self.set_recipe()
        self.log.info(f'using configuration file: {self.config}')
        self.log.info(f'using configuration recipe: {self.recipe}')

        self.define_docker_script()
        self.log.info(f'Docker Bash Script: {self.docker_bash_script}')

        self.define_docker_cmd()
        self.log.info(f'Docker Command: {self.dockerruncmd}')

        self.start_docker()
        self.wait_to_complete()

    def read_cmd_line(self):
        args = utils.cmd_line_args(f"Start the KPF DRP {self.procname.title()} Reduction.")

        # get the date to process and the unique string for docker and logs
        if not args.date:
            self.procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
        else:
            self.procdate = args.date

        # the number of cores to use
        if not args.ncpu:
            self.ncpu = self.default_ncpu
        else:
            self.ncpu = args.ncpu

    def read_cron_cfg(self):
        """
        Read the configuration for this base class and crons that extend it.
        """
        cfg_name = 'keck_kpfcron.cfg'
        cfg = utils.cfg_init(APP_PATH, cfg_name)

        # work directories
        self.data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
        self.data_drp = utils.get_cfg(cfg, 'dirs', 'data_drp')
        self.data_root = utils.get_cfg(cfg, 'dirs', 'data_root')

        # master permanent location
        self.masters_perm_dir = utils.get_cfg(cfg, 'dirs', 'masters_perm_dir')

        # final location of all logs + procdate
        self.logs_base = f"{utils.get_cfg(cfg, 'dirs', 'logs_base')}"

        # /code/KPF-Pipeline/default
        self.kpfdrp_dir = utils.get_cfg(cfg, 'dirs', 'kpfdrp_dir')

        # The path to the reference fits files
        self.reference_fits_dir = utils.get_cfg(cfg, 'dirs', 'reference_fits_dir')

        # The docker container name (only unique to the date,
        # kill any containers running with same date)
        container_base = utils.get_cfg(
            cfg, 'docker_names', f'{self.procname}_container_base'
        )
        self.containername = f"{container_base}_{self.procdate}"

        # database info,  config and ~/.pgpass file
        self.dbuser = utils.get_cfg(cfg, 'db', 'dbuser')
        self.dbname = utils.get_cfg(cfg, 'db', 'dbname')

    def set_log_dir(self):
        """
        Set the location to write the logs
        """
        self.logs_root = f"{self.data_workspace}/logs/{self.procdate}"
        self.logs_root_docker = f"/data/logs/{self.procdate}"

    def set_stdout_log(self):
        """
        Define the standard out log,  primarily needed in testing.
        """
        self.stdout_log = f"{self.logs_root_docker}/{self.log_name}.stdout"

        # this is done outside of docker
        self.reset_log(f"{self.logs_root}/{self.log_name}.stdout")

        self.log.info(f'Docker log location: {self.stdout_log}')

    def start_docker(self):
        """
        Write the docker Bash script,  start the docker container,  run the
        Bash script.
        """
        uniq_str = f"{self.procdate}-{datetime.now().strftime('%s')}"
        self.dockercmdscript = f'jobs/kpf_{self.procname}_{uniq_str}'
        self.containerimage = 'kpf-drp:latest'

        # write the docker script to be read by the docker container
        with open(self.dockercmdscript, "w") as file:
            file.write(self.docker_bash_script)

        # check if by chance another container exists with same name
        utils.chk_rm_docker_container(self.containername, self.log)

        # start the docker process
        self.pid = self.run_docker()
        self.log.info(f'Docker Process ID: {self.pid}')

    def run_docker(self):
        """
        Run the docker container.
        """
        self.log.info(f"docker run command: {self.dockerruncmd}")

        # start the docker process
        container_id = subprocess.check_output(self.dockerruncmd, shell=True).decode().strip()

        # Get the PID of the running container
        get_pid_cmd = f"docker inspect --format '{{{{.State.Pid}}}}' {container_id}"
        pid = subprocess.check_output(get_pid_cmd, shell=True).decode().strip()

        return pid

    def wait_to_complete(self):
        """
        Wait for the docker container to complete.  Three ways to notify
        complete (1) the bash script ends (2) watch an output log to stop
        updating (3) add an exit timer.

        The watch mode of the pipeline watches forever so it needs to be stopped
        manually.
        """
        # wait for the bash script to complete in the container
        completed = self.wait_container_complete()
        if not completed:
            self.log.error('Issue starting the docker process!')

        success = utils.chk_rm_docker_container(self.containername, self.log)
        if not success:
            self.log.warning(f'could not remove the Docker Container: {self.containername}')

        utils.log_stub('Ending', f'{self.procname.title()}-Processing', self.procdate, self.log)

    def reset_log(self, filename):
        """
        Remove the file if it exists and initialize it for writing.

        Args:
            filename (str): The name of the file to reset.
        """
        # Remove the file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        # Open the file in write mode to initialize it
        with open(filename, 'w') as file:
            file.write(f"Standard Output for {self.procname.title()}\n")

    def wait_container_complete(self):
        """
        Wait for the docker container to complete.

        Args:
            pid (str): The process ID of the docker container
            containername (str): the name of the container
            log (obj): the log file object
            chk_log_name (str): optional,  the log file to check if it updated recently.
            wait_time (int): optional, the time frequency to poll at

        Returns (bool): True when the container process has exited cleanly.

        """
        if self.pid == 0:
            self.log.warning(f'Docker Process never started,  PID: {self.pid}')
            return False

        # start a timer for the 'max_wait_time'
        start_time = time.time()
        iter = 0

        # Monitor the container's PID
        while True:
            elapsed_time = time.time() - start_time
            if self.exit_timer and elapsed_time > self.exit_timer:
                self.log.info(f"Time over, {self.exit_timer // 3600} hours passed. Exiting...")
                return True

            # Check if the process is still running,  if not it is complete
            try:
                subprocess.check_output(f"ps -p {self.pid}", shell=True)
                self.log.info(f"Container {self.containername} with PID {self.pid} is running.")
            except subprocess.CalledProcessError as err:
                self.log.info(f'Exception: {err}')
                utc_time = datetime.now(timezone.utc).strftime('%H:%M:%S')
                self.log.info(f"{utc_time} {self.containername} is complete!")
                break

            if iter != 0 and self.log_chk and utils.is_log_file_done(self.log_chk):
                self.log.info(f"Log file {self.log_chk} has been idle,  stopping pipeline.")
                stop_command = f"docker exec {self.containername} pkill -f kpf"
                subprocess.run(stop_command, shell=True)
                time.sleep(120)
                continue

            # Add to log and sleep
            iter += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log.info(f"[{timestamp}] Sleeping {self.wait_interval} seconds...")
            time.sleep(self.wait_interval)

        return True

    def set_recipe(self):
        """
        Need to define the recipe (self.recipe) and the recipe configuration
        file (self.config).
        """
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} has not been implemented!")

    def define_docker_script(self):
        """
        Need to define the bash script run by the docker container.
        """
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} has not been implemented!")

    def define_docker_cmd(self):
        """
        Need to define the bash script run by the docker container.
        """
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} has not been implemented!")

