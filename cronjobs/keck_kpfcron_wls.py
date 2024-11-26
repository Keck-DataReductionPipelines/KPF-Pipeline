"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeMastersWLS(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeMastersWLS, self).__init__(procname)

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.
        """
        self.recipe = 'recipes/wls_auto.recipe'
        self.config = 'configs/keck_wls_auto.cfg'

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #! /bin/bash 
    
            # set-up the full log
            rm {self.stdout_log}; 
            touch  {self.stdout_log}; 
    
            # initialized the pipeline and env
            make init >> {self.stdout_log} 2>&1; 
            export PYTHONUNBUFFERED=1; 
            git config --global --add safe.directory /code/KPF-Pipeline; 
    
            # create the directories and links in the workspace
            mkdir -p /data/masters/{self.procdate} >> {self.stdout_log} 2>&1; 
            ln -s /reference_fits /data/reference_fits; 
            cp -pr /masters/{self.procdate}/kpf_{self.procdate}*L1.fits /data/masters/{self.procdate} >> {self.stdout_log} 2>&1; 
    
            # run the pipeline
            kpf -r {self.recipe} -c {self.config} --date {self.procdate} >> {self.stdout_log} 2>&1; 
    
            # copy the files created to the permanent masters location
            # rm /masters/{self.procdate}/*master_WLS* >> {self.stdout_log} 2>&1; 
            cp -p /data/masters/{self.procdate}/*master_WLS* /masters/{self.procdate} >> {self.stdout_log} 2>&1; 
            mkdir -p /masters/{self.procdate}/wlpixelfiles >> {self.stdout_log} 2>&1; 
            cp -p /data/masters/wlpixelfiles/*kpf_{self.procdate}* /masters/{self.procdate}/wlpixelfiles >> {self.stdout_log} 2>&1; 
            cp -p /code/KPF-Pipeline/pipeline_{self.procdate}.log /masters/{self.procdate}/pipeline_wls_auto_{self.procdate}.log >> {self.stdout_log} 2>&1; 
            mkdir -p /logs/{self.procdate} 2>&1; 
            cp -p /masters/{self.procdate}/*logs /logs/{self.procdate}/ >> {self.stdout_log} 2>&1;
            rm /code/KPF-Pipeline/pipeline_{self.procdate}.log; 
            exit; 
        """


    def define_docker_cmd(self):
        """
        The command used to start the docker container and run the bash script.
        """
        super().define_docker_cmd()

        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.logs_base}:/logs -v {self.kpfdrp_dir}:/code/KPF-Pipeline "
            f"-v {self.masters_perm_dir}:/masters -v {self.data_workspace}:/data "
            f"-v {self.reference_fits_dir}:/reference_fits "
            f"--network=host -e DBPORT={self.dbport} -e DBNAME={self.dbname} "
            f"-e DBUSER={self.dbuser} -e DBPASS={self.dbpass} "
            f"-e DBSERVER=127.0.0.1 "
            f"{self.containerimage} bash ./{self.dockercmdscript}"
        )


if __name__ == '__main__':

    cron_obj = KPFPipeMastersWLS('masters_wls')
    cron_obj.run()




