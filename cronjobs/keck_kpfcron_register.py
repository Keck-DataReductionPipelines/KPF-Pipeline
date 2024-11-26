"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeMastersRegister(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeMastersRegister, self).__init__(procname)

        # poll for the script to complete every 60s
        self.wait_interval = 60

        self.reg_cals = 'database/scripts/registerCalFilesForDate.py'
        self.reg_cals_log = f'{self.logs_root_docker}/registerCalFilesForDate_{self.procdate}.out'

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.

        The register does not run the pipeline so no recipe / config required.
        """
        return

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #! /bin/bash; 
            rm {self.stdout_log};  
            touch  {self.stdout_log};  
            make init >> {self.stdout_log} 2>&1; 
            export PYTHONUNBUFFERED=1; 
            pip install psycopg2-binary; 
            git config --global --add safe.directory /code/KPF-Pipeline; 
            python {self.reg_cals} {self.procdate} >& {self.reg_cals_log};  
            cp -p  {self.reg_cals_log} /masters/{self.procdate}/ >> {self.stdout_log} 2>&1;  
            mkdir -p /logs/{self.procdate} 2>&1; 
            cp -p  {self.reg_cals_log} /logs/{self.procdate}/ >> {self.stdout_log} 2>&1;  
            rm {self.reg_cals_log}; 
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
            f"--network=host -e DBPORT={self.dbport} -e DBNAME={self.dbname} "
            f"-e DBUSER={self.dbuser} -e DBPASS={self.dbpass} "
            f"-e DBSERVER=127.0.0.1 "
            f"{self.containerimage} bash ./{self.dockercmdscript}"
        )


if __name__ == '__main__':

    cron_obj = KPFPipeMastersRegister('masters_register')
    cron_obj.run()




