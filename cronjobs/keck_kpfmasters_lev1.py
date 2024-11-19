"""
Run the Level 1 Masters reduction for a night.
"""
import os
import subprocess
import keck_utils as utils

from datetime import datetime, timezone

APP_PATH = os.path.abspath(os.path.dirname(__file__))


def mk_docker_script():
    docker_bash_script = f"""
        #! /bin/bash 

        # set-up the full log
        rm {stdout_log};  
        touch  {stdout_log}; 

        # initialized the pipeline and env
        make init >> {stdout_log} 2>&1; 
        export PYTHONUNBUFFERED=1; 
        pip install psycopg2-binary >> {stdout_log} 2>&1; 
        git config --global --add safe.directory /code/KPF-Pipeline >> {stdout_log} 2>&1; 

        # create the directories and links in the workspace
        mkdir -p /masters/{procdate} >> {stdout_log} 2>&1; 
        cp -pr /masters_permanent/{procdate}/kpf_{procdate}*.fits /masters/{procdate} >> {stdout_log} 2>&1;  
        ln -s /reference_fits /data/reference_fits; 
        ln -s /masters_permanent/reference_masters  /masters/reference_masters;  
        rm /masters/{procdate}/kpf_{procdate}_smooth_lamp.fits >> {stdout_log} 2>&1; 

        # run the pipeline
        kpf --ncpus {ncpu} --watch /masters/{procdate}/ --reprocess --masters -r {recipe} -c {config} >> {stdout_log} 2>&1; 

        # copy the results to the permanent directory
        cp -p /masters/{procdate}/* /masters_permanent/{procdate} >> {stdout_log} 2>&1; 
        mkdir -p /logs/{procdate} >> {stdout_log} 2>&1; 
        mkdir -p /masters_permanent/{procdate}/logs >> {stdout_log} 2>&1; 

        cp -p /data/logs/{procdate}/pipeline_{procdate}.log /logs/{procdate}/kpf_pipeline_masters_drp_lev1_{procdate}.log >> {stdout_log} 2>&1;  
        cp -p /data/logs/{procdate}/*kpf*.log /masters_permanent/{procdate}/logs/ >> {stdout_log} 2>&1;  
        cp -p /data/logs/{procdate}/*kpf*.log /logs/{procdate}/ >> {stdout_log} 2>&1;  
        
        exit;
    """

    with open(dockercmdscript, "w") as file:
        file.write(docker_bash_script)

    os.chmod(dockercmdscript, 0o755)

    return docker_bash_script


def run_docker():
    dockerruncmd = (
        f"docker run -d --name {containername} "
        f"-v {kpfdrp_dir}:/code/KPF-Pipeline "
        f"-v {reference_fits_dir}:/reference_fits -v {logs_base}:/logs "
        f"-v {data_workspace}:/data -v {masters_work_dir}:/masters "
        f"-v {masters_perm_dir}:/masters_permanent --network=host "
        f"-e DBPASS={dbpass} -e DBPORT={dbport} -e DBNAME={dbname} "
        f"-e DBUSER={dbuser} -e DBSERVER=127.0.0.1 {containerimage} "
        f"bash ./{dockercmdscript}"
    )

    log.info(f"docker run command: {dockerruncmd}")

    # start the docker process
    container_id = subprocess.check_output(dockerruncmd, shell=True).decode().strip()

    # Get the PID of the running container
    get_pid_cmd = f"docker inspect --format '{{{{.State.Pid}}}}' {container_id}"
    pid = subprocess.check_output(get_pid_cmd, shell=True).decode().strip()

    return pid


if __name__ == '__main__':

    args = utils.cmd_line_args("Start the KPF Masters Reduction Level 1 for night of data.")

    # get the date to process and the unique string for docker and logs
    if not args.date:
        procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
    else:
        procdate = args.date

    # the number of cores to run with
    ncpu = 232

    tm = procdate + datetime.now().strftime('%s')
    uniq_str = f"{procdate}-{tm}"

    cfg = utils.cfg_init(APP_PATH, 'keck_kpfcron.cfg')

    # final location of all logs + procdate
    logs_base = f"{utils.get_cfg(cfg, 'dirs', 'logs_base')}"

    # /kpfdata/data_workspace/logs
    logs_root = f"{utils.get_cfg(cfg, 'dirs', 'logs_root')}/{procdate}"
    log_name = f'keck_kpfmasters_lev1_{procdate}'

    log = utils.configure_logger(logs_root, f"{log_name}")

    # work directories
    data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
    masters_work_dir = f"{data_workspace}/masters/"

    # master permanent location
    masters_perm_dir = utils.get_cfg(cfg, 'dirs', 'masters_perm_dir')

    # /code/KPF-Pipeline/default
    kpfdrp_dir = utils.get_cfg(cfg, 'dirs', 'kpfdrp_dir')

    # The path to the reference fits files
    reference_fits_dir = utils.get_cfg(cfg, 'dirs', 'reference_fits_dir')

    # The docker container name
    containername = f"{utils.get_cfg(cfg, 'docker_names', 'lev1_container_base')}_{tm}"

    # database info,  config and ~/.pgpass file
    dbuser = utils.get_cfg(cfg, 'db', 'dbuser')
    dbname = utils.get_cfg(cfg, 'db', 'dbname')
    dbport, dbpass = utils.get_pgpass(dbname, dbuser)

    dockercmdscript = f'jobs/kpfmasterscmd_lev1_{uniq_str}'
    containerimage = 'kpf-drp:latest'
    recipe = 'recipes/kpf_drp.recipe'
    config = 'configs/keck_kpf_masters_l1.cfg'

    logs_root_docker = logs_root.replace(data_workspace, '/data/')
    stdout_log = f"{logs_root_docker}/{log_name}.stdout"

    # change to the code directory
    os.chdir(kpfdrp_dir)

    # log start
    utils.log_stub('Starting', 'Level 1', procdate, log)

    # write the docker script to be read by the docker container
    docker_bash_script = mk_docker_script()

    # check if by chance another container exists with same name
    utils.chk_rm_docker_container(containername, log)

    # start the docker process
    pid = run_docker()

    # wait for the bash script to complete in the container
    log_chk = f"{logs_root}/{log_name}.stdout"
    utils.wait_container_complete(pid, containername, log, chk_log_name=log_chk)

    success = utils.chk_rm_docker_container(containername, log)

    utils.log_stub('Ending', 'Level 1', procdate, log)

