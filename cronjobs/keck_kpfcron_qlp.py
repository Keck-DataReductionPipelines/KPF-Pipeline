"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os
import sys
import signal

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))

class KPFPipeQuickLook(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeQuickLook, self).__init__(procname)

        # dial back the ncpu since it is running at night with the
        # Nightly Watch Pipeline
        self.ncpu = 4
        # self.ncpu = 1

        # exit after 23.5 hours,  start 1pm UT to 00:30 UT (next day)
        if not self.exit_timer:
            self.exit_timer = 23.6 * 60 * 60

    def set_recipe(self):

        # set the recipe and config file
        # self.recipe = 'recipes/keck_quicklook_watch.recipe'
        self.recipe = 'recipes/quicklook_watch.recipe'
        self.config = 'configs/keck_quicklook_watch.cfg'

    def set_log_dir(self):
        """
        Set the location to write the logs
        """
        self.logs_root = f"{self.data_drp}/logs/QLP/{self.procdate}"
        self.logs_root_docker = f"/data/logs/QLP/{self.procdate}"

    def define_docker_script(self):
        self.docker_bash_script = f"""
            #!/bin/bash
    
            # mkdirs if they don't exist
            {self.make_directories_str()}
    
            # make the symlinks
            {self.link_wrkspace_drp()}

            # set-up the pipeline
            make init >> {self.stdout_log} 2>&1;
            
            python /code/KPF-Pipeline/cronjobs/keck_slow_touch.py --date {self.procdate} --fits /data/{self.level}/ --log /data/logs/ >> {self.stdout_log} &

            # run the pipeline for all data in the directory
            kpf --watch /data/{self.level}/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;

            """

    def define_docker_cmd(self):
        super().define_docker_cmd()

        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.kpfdrp_dir}:/code/KPF-Pipeline -v {self.data_drp}:/data "
            f"-v {self.logs_base}:/logs -v {self.masters_perm_dir}:/masters "
            f"-v {self.data_workspace}:/data_workspace -v {self.data_root}:/data_root "
            f"--network=host -e DBPASS={self.dbpass} -e DBPORT={self.dbport} "
            f"-e DBNAME={self.dbname} -e DBUSER={self.dbuser} -e DBSERVER=127.0.0.1 "
            f"{self.containerimage} bash ./{self.dockercmdscript}"
        )


def main():
    cron_obj = KPFPipeQuickLook('quicklook')

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





