"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os
import sys
import signal
import keck_utils as utils

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeNightly(KPFPipeCronBase):
    """
    The Nightly (morning-after) Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeNightly, self).__init__(procname)

        # exit after 12 hours (12 hrs * 60 minutes * 60 seconds)
        if not self.exit_timer:
            self.exit_timer = 6 * 60 * 60

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.
        """
        self.recipe = 'recipes/kpf_drp.recipe'
        cfg_dir = 'configs'
        self.config = utils.get_dated_cfg(self.procdate, cfg_dir, 'keck_kpf_drp')
        if not self.config:
            self.log.error(f'config not found for {self.procdate}, {cfg_dir}, keck_kpf_drp')
            exit()

    def set_log_dir(self):
        """
        Set the location to write the logs
        """
        self.logs_root = f"{self.data_drp}/logs/{self.procdate}"
        self.logs_root_docker = f"/data/logs/{self.procdate}"

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #!/bin/bash

            # mkdirs if they don't exist
            {self.make_directories_str()}

            # make the symlinks
            {self.link_wrkspace_drp()}

            # set-up the pipeline
            make init >> {self.stdout_log} 2>&1;

            # touch the files that currently exist in the directory so they are recognized as new
            python /code/KPF-Pipeline/cronjobs/keck_slow_touch.py --date {self.procdate} --fits /data_workspace/L0 --log /data/logs/ >> {self.stdout_log} &

            # run the pipeline for all data in the directory
            kpf --watch /data/L0/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;
            """

    def define_docker_cmd(self):
        """
        The command used to start the docker container and run the bash script.
        """
        super().define_docker_cmd()

        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.kpfdrp_dir}:/code/KPF-Pipeline -v {self.data_drp}:/data "
            f"-v {self.logs_base}:/logs -v {self.masters_perm_dir}:/masters "
            f"-v {self.data_workspace}:/data_workspace "
            f"-v {self.data_root}:/data_root --network=host "
            f"-e DBPASS={self.dbpass} -e DBPORT={self.dbport} "
            f"-e DBNAME={self.dbname} -e DBUSER={self.dbuser} "
            f"-e DBSERVER=127.0.0.1 "
            f"{self.containerimage} bash ./{self.dockercmdscript}"
        )


def main():
    cron_obj = KPFPipeNightly('nightly')

    def exit_cleanly(signum, frame):
        print(f"Received signal {signum}.")
        cron_obj.clean_up()
        sys.exit(0)

    signal.signal(signal.SIGTERM, exit_cleanly)
    signal.signal(signal.SIGINT, exit_cleanly)

    try:
        cron_obj.run()
    except Exception as e:
        print(f"Exception occurred: {e}")
        cron_obj.clean_up()
        sys.exit(1)
    else:
        cron_obj.clean_up()


if __name__ == '__main__':
    main()



