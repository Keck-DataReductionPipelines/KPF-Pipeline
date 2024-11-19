from datetime import datetime, timezone
import os
import subprocess
import keck_utils as utils

APP_PATH = os.path.abspath(os.path.dirname(__file__))


def mk_docker_script():
    docker_bash_script = f"""
        #! /bin/bash 

        # setup the full log
        mkdir -p {logs_root};  
        rm -f {stdout_log}; 
        touch  {stdout_log};

        # initialized the pipeline and env
        make init >> {stdout_log} 2>&1; 
        export PYTHONUNBUFFERED=1; 
        git config --global --add safe.directory /code/KPF-Pipeline >> {stdout_log} 2>&1; 

        # remove previously generated masters
        rm -rf /data/masters/{procdate}/*fits >> {stdout_log} 2>&1;

        # link the reference FITS files to the workspace
        ln -fs /reference_fits /data/reference_fits;

        # remove old masters from the pool
        find /data/masters/pool/kpf_????????_master_*fits -mtime +7 -exec rm {{}} + >> {stdout_log} 2>&1;

        # run the pipeline
        kpf -r {recipe} -c {config} --date {procdate} >> {stdout_log} 2>&1; 
        python {smooth_lamp_script} /data/masters/pool/kpf_{procdate}_master_flat.fits /data/masters/pool/kpf_{procdate}_smooth_lamp.fits >& {smooth_lamp_log} >> {stdout_log} 2>&1;
        sleep 3; 

        # copy the files created to the permanent masters location
        cp -p /data/masters/pool/kpf_{procdate}* /masters/{procdate} >> {stdout_log} 2>&1; 
        cp -p /data/masters/pool/kpf_{procdate}* /masters/pool/ >> {stdout_log} 2>&1; 
        chmod a+wrx /masters/{procdate}/* >> {stdout_log} 2>&1; 
        cp -p /data/logs/{procdate}/pipeline_{procdate}.log /masters/{procdate}/kpf_pipeline_masters_drp_lev0_{procdate}.log 2>&1; 
        cp -p /code/KPF-Pipeline/{smooth_lamp_log} /masters/{procdate}/ >> {stdout_log} 2>&1;
        rm /code/KPF-Pipeline/{smooth_lamp_log} 2>&1; 
        mkdir -p /logs/{procdate} 2>&1; 
        cp -p /data/logs/{procdate}/pipeline_{procdate}.log /logs/{procdate}/kpf_pipeline_masters_drp_lev0_{procdate}.log >> {stdout_log} 2>&1;  
        cp -p /data/logs/{procdate}/*kpf*.log /logs/{procdate}/ >> {stdout_log} 2>&1;  
        cp -p /masters/{procdate}/*.log /logs/{procdate}/ >> {stdout_log} 2>&1;  
        exit
    """

    with open(dockercmdscript, "w") as file:
        file.write(docker_bash_script)

    os.chmod(dockercmdscript, 0o755)

    return docker_bash_script


def run_docker():
    dockerruncmd = (f"docker run -d --name {containername} -v {kpfdrp_dir}:/code/KPF-Pipeline "
                    f"-v {masters_perm_dir}:/masters -v {data_workspace}:/data --network=host "
                    f"-v {reference_fits_dir}:/reference_fits "
                    f"-v {logs_base}:/logs "
                    f"-e DBPORT={dbport} -e DBNAME={dbname} -e DBUSER={dbuser} "
                    f"-e DBPASS={dbpass} -e DBSERVER=127.0.0.1 {containerimage} "
                    f"bash ./{dockercmdscript}")

    log.info(f"docker run command: {dockerruncmd}")

    # start the docker process
    container_id = subprocess.check_output(dockerruncmd, shell=True).decode().strip()

    # Get the PID of the running container
    get_pid_cmd = f"docker inspect --format '{{{{.State.Pid}}}}' {container_id}"
    pid = subprocess.check_output(get_pid_cmd, shell=True).decode().strip()

    return pid


if __name__ == '__main__':

    args = utils.cmd_line_args("Start the KPF Masters Reduction Level 0 for night of data.")

    # get the date to process and the unique string for docker and logs
    if not args.date:
        procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
    else:
        procdate = args.date

    #get the config file
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

    # final location of all logs
    logs_base = f"{utils.get_cfg(cfg, 'dirs', 'logs_base')}"

    # /kpfdata/data_workspace/logs
    logs_root = f"{utils.get_cfg(cfg, 'dirs', 'logs_root')}/{procdate}"
    log_name = f'keck_kpfmasters_lev0_{procdate}'

    log = utils.configure_logger(logs_root, f"{log_name}")

    # /code/KPF-Pipeline/default
    kpfdrp_dir = utils.get_cfg(cfg, 'dirs', 'kpfdrp_dir')

    # The path to the reference fits files
    reference_fits_dir = utils.get_cfg(cfg, 'dirs', 'reference_fits_dir')

    # The docker container name
    containername = f"{utils.get_cfg(cfg, 'docker_names', 'lev0_container_base')}_{tm}"

    # database info,  config and ~/.pgpass file
    dbuser = utils.get_cfg(cfg, 'db', 'dbuser')
    dbname = utils.get_cfg(cfg, 'db', 'dbname')
    dbport, dbpass = utils.get_pgpass(dbname, dbuser)

    dockercmdscript = f'jobs/kpfmasterscmd_lev0_{uniq_str}'
    containerimage = 'kpf-drp:latest'
    recipe = 'recipes/kpf_masters_drp.recipe'
    config = 'configs/keck_kpf_masters_drp.cfg'

    logs_root_docker = logs_root.replace(data_workspace, '/data/')
    stdout_log = f"{logs_root_docker}/{log_name}.stdout"

    # smooth lamp script
    smooth_lamp_script = 'scripts/make_smooth_lamp_pattern_new.py'
    smooth_lamp_log = f'make_smooth_lamp_pattern_new_{procdate}.log'

    # change to the code directory
    os.chdir(kpfdrp_dir)

    # log start
    utils.log_stub('Starting', 'Level 0', procdate, log)

    # write the docker script to be read by the docker container
    docker_bash_script = mk_docker_script()

    # sync the files from the
    utils.sync_files(data_workspace, procdate, log)

    # check if by chance another container exists with same name
    utils.chk_rm_docker_container(containername, log)

    # start the docker process
    pid = run_docker()

    # wait for the bash script to complete in the container
    utils.wait_container_complete(pid, containername, log)

    success = utils.chk_rm_docker_container(containername, log)

    utils.log_stub('Ending', 'Level 0', procdate, log)


