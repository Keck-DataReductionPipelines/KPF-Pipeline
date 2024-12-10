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

        # dial back the ncpu since it is running at night with the Quicklook Watch Pipeline
        self.ncpu = 120

        # exit after 12 hours
        self.exit_timer = 14 * 60 * 60

    def set_recipe(self):

        # set the recipe and config file
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
            mkdir -p /data/logs/QLP/{self.procdate};
            mkdir -p /data/L1/{self.procdate};
            mkdir -p /data/L2/{self.procdate};
    
            # make the symlinks
            ln -fs /data_workspace/L0/{self.procdate} /data/L0/{self.procdate};
            ln -fs /data_workspace/2D/{self.procdate} /data/2D/{self.procdate};
            ln -fs /masters /data/masters;
            ln -fs /data_root/reference_fits /data/reference_fits;
    
            # set-up the pipeline
            make init >> {self.stdout_log} 2>&1;
    
            # touch the files so the pipe recognized them as new
            python /code/KPF-Pipeline/cronjobs/keck_slow_touch.py --date {self.procdate} --fits /data/L0 --log /data/logs/QLP/ &
    
            # run the pipeline for all data in the directory
            kpf --reprocess --watch /data/L0/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;
            
            # once it catches up,  if it exits,  run again without the reprocess to avoid exiting early
            kpf --watch /data/L0/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;
    
            # remove the symlinks
            rm -f /data/masters;
            rm -f /data/reference_fits;
            rm -f /data/L0/{self.procdate};
            rm -f /data/2D/{self.procdate};
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





