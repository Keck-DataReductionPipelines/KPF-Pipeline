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
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeNightly, self).__init__(procname)

        # dial back the ncpu since it is running at night with the QLP
        self.ncpu = 120

        # exit after 14 hours (6pm to 8am) (18 hrs * 60 minutes * 60 seconds)
        self.exit_timer = 14 * 60 * 60

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.
        """
        self.recipe = 'recipes/kpf_drp.recipe'
        cfg_dir = 'configs'
        self.config = utils.get_dated_cfg(self.procdate, cfg_dir, 'keck_kpf_drp_watch')
        if not self.config:
            self.log.error(f'config not found for {self.procdate}, {cfg_dir}, keck_kpf_drp')
            exit()

    def set_log_dir(self):
        """
        Set the location to write the logs
        """
        self.logs_root = f"{self.data_drp}/logs/watch/{self.procdate}"
        self.logs_root_docker = f"/data/logs/watch/{self.procdate}"

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #!/bin/bash

            # mkdirs if they don't exist
            mkdir -p /data/logs/{self.procdate}; 
            mkdir -p /data/L1/{self.procdate}; 
            mkdir -p /data/L2/{self.procdate}; 

            # make the symlinks
            ln -fs /data_workspace/L0/{self.procdate} /data/L0/{self.procdate};
            ln -fs /data_workspace/2D/{self.procdate} /data/2D/{self.procdate};
            ln -fs /masters /data/masters;
            ln -fs /data_root/reference_fits /data/reference_fits;

            # set-up the pipeline
            make init >> {self.stdout_log} 2>&1;

            # run the pipeline for all data in the directory
            kpf --reprocess --watch /data/L0/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;
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
    cron_obj = KPFPipeNightly('nightly_watch')

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



