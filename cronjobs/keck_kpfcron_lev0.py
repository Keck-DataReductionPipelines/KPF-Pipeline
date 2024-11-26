"""
The QuickLook process scripts that run the KPF Pipeline.  The format
is meant to be used to run the pipeline in-house and Keck.

-LFuhrman 2024Nov21

"""
import os
import keck_utils as utils

from datetime import datetime, timezone

from keck_kpfcron_base import KPFPipeCronBase

APP_PATH = os.path.abspath(os.path.dirname(__file__))


class KPFPipeMastersLevel0(KPFPipeCronBase):
    """
    The QuickLook Processing Cronjob
    """
    def __init__(self, procname):
        super(KPFPipeMastersLevel0, self).__init__(procname)

        self.smooth_lamp_script = 'scripts/make_smooth_lamp_pattern_new.py'
        self.smooth_lamp_log = f'make_smooth_lamp_pattern_new_{self.procdate}.log'

    def set_recipe(self):
        """
        The quicklook recipe and recipe configuration file.
        """
        self.recipe = 'recipes/kpf_masters_drp.recipe'
        cfg_dir = 'configs'
        self.config = utils.get_dated_cfg(self.procdate, cfg_dir, 'keck_kpf_masters_drp')
        if not self.config:
            self.log.error(f'config not found for {self.procdate}, '
                           f'{cfg_dir}, keck_kpf_masters_drp')
            exit()

    def define_docker_script(self):
        """
        The script to run the pipeline from within the docker container.
        """
        self.docker_bash_script = f"""
            #! /bin/bash 
    
            # setup the full log
            mkdir -p {self.logs_root};  
            rm -f {self.stdout_log}; 
            touch  {self.stdout_log};
    
            # initialized the pipeline and env
            make init >> {self.stdout_log} 2>&1; 
            export PYTHONUNBUFFERED=1; 
            git config --global --add safe.directory /code/KPF-Pipeline >> {self.stdout_log} 2>&1; 
    
            # remove previously generated masters
            rm -rf /data/masters/{self.procdate}/*fits >> {self.stdout_log} 2>&1;
    
            # link the reference FITS files to the workspace
            ln -fs /reference_fits /data/reference_fits;
    
            # remove old masters from the pool
            find /data/masters/pool/kpf_????????_master_*fits -mtime +7 -exec rm {{}} + >> {self.stdout_log} 2>&1;
    
            # run the pipeline
            kpf -r {self.recipe} -c {self.config} --date {self.procdate} >> {self.stdout_log} 2>&1; 
            python {self.smooth_lamp_script} /data/masters/pool/kpf_{self.procdate}_master_flat.fits /data/masters/pool/kpf_{self.procdate}_smooth_lamp.fits >& {self.smooth_lamp_log} >> {self.stdout_log} 2>&1;
            sleep 3; 
    
            # copy the files created to the permanent masters location
            cp -p /data/masters/pool/kpf_{self.procdate}* /masters/{self.procdate} >> {self.stdout_log} 2>&1; 
            cp -p /data/masters/pool/kpf_{self.procdate}* /masters/pool/ >> {self.stdout_log} 2>&1; 
            chmod a+wrx /masters/{self.procdate}/* >> {self.stdout_log} 2>&1; 
            cp -p /data/logs/{self.procdate}/pipeline_{self.procdate}.log /masters/{self.procdate}/kpf_pipeline_masters_drp_lev0_{self.procdate}.log 2>&1; 
            cp -p /code/KPF-Pipeline/{self.smooth_lamp_log} /masters/{self.procdate}/ >> {self.stdout_log} 2>&1;
            rm /code/KPF-Pipeline/{self.smooth_lamp_log} 2>&1; 
            mkdir -p /logs/{self.procdate} 2>&1; 
            cp -p /data/logs/{self.procdate}/pipeline_{self.procdate}.log /logs/{self.procdate}/kpf_pipeline_masters_drp_lev0_{self.procdate}.log >> {self.stdout_log} 2>&1;  
            cp -p /data/logs/{self.procdate}/*kpf*.log /logs/{self.procdate}/ >> {self.stdout_log} 2>&1;  
            cp -p /masters/{self.procdate}/*.log /logs/{self.procdate}/ >> {self.stdout_log} 2>&1;  
            exit
        """

    def define_docker_cmd(self):
        """
        The command used to start the docker container and run the bash script.
        """
        super().define_docker_cmd()

        self.dockerruncmd = (
            f"docker run -d --name {self.containername} "
            f"-v {self.kpfdrp_dir}:/code/KPF-Pipeline "
            f"-v {self.masters_perm_dir}:/masters "
            f"-v {self.data_workspace}:/data --network=host "
            f"-v {self.reference_fits_dir}:/reference_fits "
            f"-v {self.logs_base}:/logs "
            f"-e DBPORT={self.dbport} -e DBNAME={self.dbname} "
            f"-e DBUSER={self.dbuser} -e DBPASS={self.dbpass} "
            f"-e DBSERVER=127.0.0.1 {self.containerimage} "
            f"bash ./{self.dockercmdscript}"
        )


if __name__ == '__main__':

    cron_obj = KPFPipeMastersLevel0('masters_level0')
    cron_obj.run()




