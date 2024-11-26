"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeQuickLook(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeQuickLook, self).__init__(procname)

        # exit after 12 hours
        self.stop_tmr = 12 * 60 * 60

    def set_recipe(self):

        # set the recipe and config file
        self.recipe = 'recipes/quicklook_watch.recipe'
        self.config = 'configs/keck_quicklook_watch.cfg'

    # def set_stdout_log(self):
    #     logs_root_docker = self.logs_root.replace(self.data_drp, '/data/')
    #     self.stdout_log = f"{logs_root_docker}/{self.log_name}.stdout"
    #     self.reset_log(self.stdout_log)
    #
    #     self.log.info(f'External log location: {self.logs_root}')
    #     self.log.info(f'Docker log location: {self.stdout_log}')

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
            python /code/KPF-Pipeline/cronjobs/keck_slow_touch.py --date {self.procdate} --fits /data/L0 --log /data/logs/{self.procdate} &
    
            # run the pipeline for all data in the directory
            kpf --reprocess --watch /data/L0/{self.procdate}/ --ncpus={self.ncpu} -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1;
    
            # keep the log
            mkdir -p /logs/{self.procdate} 2>&1; 
            cp -p /code/KPF-Pipeline/logs/pipeline_{self.procdate}.log /logs/{self.procdate}/kpf_pipeline_nightly_{self.procdate}.log >> {self.stdout_log} 2>&1;
    
            # remove the symlinks
            rm -f /data/masters;
            rm -f /data/reference_fits;
            rm -f /data/L0/{self.procdate};
            rm -f /data/2D/{self.procdate};
            """

    def define_docker_cmd(self):
        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.kpfdrp_dir}:/code/KPF-Pipeline -v {self.data_drp}:/data "
            f"-v {self.logs_base}:/logs -v {self.masters_perm_dir}:/masters "
            f"-v {self.data_workspace}:/data_workspace -v {self.data_root}:/data_root "
            f"--network=host -e DBPASS={self.dbpass} -e DBPORT={self.dbport} "
            f"-e DBNAME={self.dbname} -e DBUSER={self.dbuser} -e DBSERVER=127.0.0.1 "
            f"{self.containerimage} bash ./{self.dockercmdscript}"
        )


if __name__ == '__main__':

    cron_obj = KPFPipeQuickLook('quicklook')
    cron_obj.run()




