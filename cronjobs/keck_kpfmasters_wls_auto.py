from datetime import datetime, timezone
import os
import subprocess
import keck_utils as utils

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
        git config --global --add safe.directory /code/KPF-Pipeline; 

        # create the directories and links in the workspace
        mkdir -p /data/masters/{procdate} >> {stdout_log} 2>&1; 
        ln -s /reference_fits /data/reference_fits; 
        cp -pr /masters/{procdate}/kpf_{procdate}*L1.fits /data/masters/{procdate} >> {stdout_log} 2>&1; 

        # run the pipeline
        kpf -r {recipe} -c {config} --date {procdate} >> {stdout_log} 2>&1; 

        # copy the files created to the permanent masters location
        # rm /masters/{procdate}/*master_WLS* >> {stdout_log} 2>&1; 
        cp -p /data/masters/{procdate}/*master_WLS* /masters/{procdate} >> {stdout_log} 2>&1; 
        mkdir -p /masters/{procdate}/wlpixelfiles >> {stdout_log} 2>&1; 
        cp -p /data/masters/wlpixelfiles/*kpf_{procdate}* /masters/{procdate}/wlpixelfiles >> {stdout_log} 2>&1; 
        cp -p /code/KPF-Pipeline/pipeline_{procdate}.log /masters/{procdate}/pipeline_wls_auto_{procdate}.log >> {stdout_log} 2>&1; 
        mkdir -p /logs/{procdate} 2>&1; 
        cp -p /masters/{procdate}/*logs /logs/{procdate}/ >> {stdout_log} 2>&1;
        rm /code/KPF-Pipeline/pipeline_{procdate}.log; 
        exit; 
    """

    with open(dockercmdscript, "w") as file:
        file.write(docker_bash_script)

    os.chmod(dockercmdscript, 0o755)

    return docker_bash_script


def run_docker():
    dockerruncmd = (
        f"docker run -d --name {containername} "
        f"-v {logs_base}:/logs -v {kpfdrp_dir}:/code/KPF-Pipeline "
        f"-v {masters_perm_dir}:/masters -v {data_workspace}:/data "
        f"-v {reference_fits_dir}:/reference_fits "
        f"--network=host -e DBPORT={dbport} -e DBNAME={dbname} "
        f"-e DBUSER={dbuser} -e DBPASS={dbpass} -e DBSERVER=127.0.0.1 "
        f"{containerimage} bash ./{dockercmdscript}"
    )

    log.info(f"docker run command: {dockerruncmd}")

    # start the docker process
    container_id = subprocess.check_output(dockerruncmd, shell=True).decode().strip()

    # Get the PID of the running container
    get_pid_cmd = f"docker inspect --format '{{{{.State.Pid}}}}' {container_id}"
    pid = subprocess.check_output(get_pid_cmd, shell=True).decode().strip()

    return pid


if __name__ == '__main__':

    args = utils.cmd_line_args("Start the KPF WLS Masters Reduction.")

    # get the date to process and the unique string for docker and logs
    if not args.date:
        procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
    else:
        procdate = args.date

    # get the config file
    cfg = utils.cfg_init(APP_PATH, 'keck_kpfcron.cfg')

    # work directories
    data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
    masters_work_dir = f"{data_workspace}/masters/"

    # master permanent location
    masters_perm_dir = utils.get_cfg(cfg, 'dirs', 'masters_perm_dir')

    # create the output directories
    utils.mk_output_dirs(data_workspace, masters_perm_dir, procdate)

    tm = procdate + datetime.now().strftime('%s')
    uniq_str = f"{procdate}-{tm}"

    # final location of all logs + procdate
    logs_base = f"{utils.get_cfg(cfg, 'dirs', 'logs_base')}"

    # /kpfdata/data_workspace/logs
    logs_root = f"{utils.get_cfg(cfg, 'dirs', 'logs_root')}/{procdate}"
    log_name = f'keck_kpfmasters_wls_auto_{procdate}'

    log = utils.configure_logger(logs_root, f"{log_name}")

    # /code/KPF-Pipeline/default
    kpfdrp_dir = utils.get_cfg(cfg, 'dirs', 'kpfdrp_dir')

    # The path to the reference fits files
    reference_fits_dir = utils.get_cfg(cfg, 'dirs', 'reference_fits_dir')

    # The docker container name
    containername = f"{utils.get_cfg(cfg, 'docker_names', 'wls_container_base')}_{tm}"

    # database info,  config and ~/.pgpass file
    dbuser = utils.get_cfg(cfg, 'db', 'dbuser')
    dbname = utils.get_cfg(cfg, 'db', 'dbname')
    dbport, dbpass = utils.get_pgpass(dbname, dbuser)

    dockercmdscript = f'jobs/kpfmasterscmd_wls_{uniq_str}'
    containerimage = 'kpf-drp:latest'
    recipe = 'recipes/wls_auto.recipe'
    config = 'configs/keck_wls_auto.cfg'

    logs_root_docker = logs_root.replace(data_workspace, '/data/')
    stdout_log = f"{logs_root_docker}/{log_name}.stdout"

    # change to the code directory
    os.chdir(kpfdrp_dir)

    # log start
    utils.log_stub('Starting', 'WLS', procdate, log)

    # write the docker script to be read by the docker container
    docker_bash_script = mk_docker_script()

    # check if by chance another container exists with same name
    utils.chk_rm_docker_container(containername, log)

    # start the docker process
    pid = run_docker()

    # wait for the bash script to complete in the container
    utils.wait_container_complete(pid, containername, log)

    success = utils.chk_rm_docker_container(containername, log)

    utils.log_stub('Ending', 'WLS', procdate, log)

