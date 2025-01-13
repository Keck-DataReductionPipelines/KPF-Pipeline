"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os
import keck_utils as utils

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeMastersLevel1(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeMastersLevel1, self).__init__(procname)

        # set to monitor the log file for being idle (location is real
        # location, outside the container)
        self.log_chk = self.stdout_log.replace('/data', self.data_workspace)

        # set the masters working directory
        self.masters_work_dir = f"{self.data_workspace}/masters/"

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.
        """
        self.recipe = 'recipes/kpf_drp.recipe'
        cfg_dir = 'configs'
        self.config = utils.get_dated_cfg(self.procdate, cfg_dir, 'keck_kpf_masters_l1')
        if not self.config:
            self.log.error(f'config not found for {self.procdate}, '
                           f'{cfg_dir}, keck_kpf_masters_l1')
            exit()

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #! /bin/bash 

            # initialized the pipeline and env
            make init >> {self.stdout_log} 2>&1; 
            export PYTHONUNBUFFERED=1; 
            pip install psycopg2-binary >> {self.stdout_log} 2>&1; 
            git config --global --add safe.directory /code/KPF-Pipeline >> {self.stdout_log} 2>&1; 

            # create the directories and links in the workspace
            mkdir -p /masters/{self.procdate} >> {self.stdout_log} 2>&1; 
            cp -pr /masters_permanent/{self.procdate}/kpf_{self.procdate}*.fits /masters/{self.procdate} >> {self.stdout_log} 2>&1;  
            ln -s /reference_fits /data/reference_fits; 
            ln -s /masters_permanent/reference_masters  /masters/reference_masters;  
            rm /masters/{self.procdate}/kpf_{self.procdate}_smooth_lamp.fits >> {self.stdout_log} 2>&1; 

            # run the pipeline
            kpf --ncpus {self.ncpu} --watch /masters/{self.procdate}/ --reprocess --masters -r {self.recipe} -c {self.config} >> {self.stdout_log} 2>&1; 

            # copy the results to the permanent directory
            cp -p /masters/{self.procdate}/* /masters_permanent/{self.procdate} >> {self.stdout_log} 2>&1; 
            mkdir -p /logs/{self.procdate} >> {self.stdout_log} 2>&1; 
            mkdir -p /masters_permanent/{self.procdate}/logs >> {self.stdout_log} 2>&1; 

            cp -p /data/logs/{self.procdate}/pipeline_{self.procdate}.log /logs/{self.procdate}/kpf_pipeline_masters_drp_lev1_{self.procdate}.log >> {self.stdout_log} 2>&1;  
            cp -p /data/logs/{self.procdate}/*kpf*.log /masters_permanent/{self.procdate}/logs/ >> {self.stdout_log} 2>&1;  
            cp -p /data/logs/{self.procdate}/*kpf*.log /logs/{self.procdate}/ >> {self.stdout_log} 2>&1;  

            exit;
        """

    def define_docker_cmd(self):
        """
        The command used to start the docker container and run the bash script.
        """
        super().define_docker_cmd()

        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.kpfdrp_dir}:/code/KPF-Pipeline "
            f"-v {self.reference_fits_dir}:/reference_fits "
            f"-v {self.logs_base}:/logs -v {self.data_workspace}:/data "
            f"-v {self.masters_work_dir}:/masters "
            f"-v {self.masters_perm_dir}:/masters_permanent --network=host "
            f"-e DBPASS={self.dbpass} -e DBPORT={self.dbport} "
            f"-e DBNAME={self.dbname} -e DBUSER={self.dbuser} "
            f"-e DBSERVER=127.0.0.1 {self.containerimage} "
            f"bash ./{self.dockercmdscript}"
        )


if __name__ == '__main__':

    cron_obj = KPFPipeMastersLevel1('masters_level1')
    cron_obj.run()




