from datetime import datetime, timezone
import os
import subprocess
import keck_utils as utils

APP_PATH = os.path.abspath(os.path.dirname(__file__))


def mk_docker_script():

    docker_bash_script = f"""
        #!/bin/bash

        source /home/kpfdrprun/.bash_profile;  

        # mkdirs if they don't exist
        mkdir -p /data/logs/{procdate};
        mkdir -p /data/L1/{procdate};
        mkdir -p /data/L2/{procdate};

        # make the symlinks
        ln -fs /data_workspace/L0/{procdate} /data/L0/{procdate};
        ln -fs /data_workspace/2D/{procdate} /data/2D/{procdate};
        ln -fs /masters /data/masters;
        ln -fs /data_root/reference_fits /data/reference_fits;

        # set-up the pipeline
        make init >> {stdout_log} 2>&1;

        # touch the files so the pipe recognized them as new
        python /code/KPF-Pipeline/cronjobs/keck_slow_touch.py --date {procdate} --fits /data/L0 --log /data/logs/{procdate} &

        # run the pipeline for all data in the directory
        kpf --reprocess --watch /data/L0/{procdate}/ --ncpus={ncpu} -r {recipe} -c {config} >> {stdout_log} 2>&1;

        # keep the log
        mkdir -p /logs/{procdate} 2>&1; 
        cp -p /code/KPF-Pipeline/logs/pipeline_{procdate}.log /logs/{procdate}/kpf_pipeline_nightly_{procdate}.log >> {stdout_log} 2>&1;

        # remove the symlinks
        rm -f /data/masters;
        rm -f /data/reference_fits;
        rm -f /data/L0/{procdate};
        rm -f /data/2D/{procdate};
        """

    with open(dockercmdscript, "w") as file:
        file.write(docker_bash_script)

    return docker_bash_script


def run_docker():
    dockerruncmd = (
        f"docker run -d --name {containername} "
        f"-v {kpfdrp_dir}:/code/KPF-Pipeline -v {data_drp}:/data "
        f"-v {logs_base}:/logs -v {masters_perm_dir}:/masters "
        f"-v {data_workspace}:/data_workspace -v {data_root}:/data_root "
        f"--network=host -e DBPASS={dbpass} -e DBPORT={dbport} "
        f"-e DBNAME={dbname} -e DBUSER={dbuser} -e DBSERVER=127.0.0.1 "
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

    args = utils.cmd_line_args("Start the KPF DRP Nightly Reduction.")

    # get the date to process and the unique string for docker and logs
    if not args.date:
        procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
    else:
        procdate = args.date

    # the number of cores to use
    ncpu = 232

    #get the config file
    cfg = utils.cfg_init(APP_PATH, 'keck_kpfcron.cfg')

    # work directories
    data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
    data_drp = utils.get_cfg(cfg, 'dirs', 'data_drp')
    data_root = utils.get_cfg(cfg, 'dirs', 'data_root')

    # master permanent location
    masters_perm_dir = utils.get_cfg(cfg, 'dirs', 'masters_perm_dir')

    # create the output directories
    utils.mk_output_dirs(data_workspace, masters_perm_dir, procdate)

    tm = procdate + datetime.now().strftime('%s')
    uniq_str = f"{procdate}-{tm}"

    # final location of all logs + procdate
    logs_base = f"{utils.get_cfg(cfg, 'dirs', 'logs_base')}"

    # /kpfdata/data_workspace/logs
    logs_root = f"{data_drp}/logs/{procdate}"
    log_name = f'keck_kpfnightly_drp_{procdate}'
    os.makedirs(logs_root, exist_ok=True)

    log = utils.configure_logger(logs_root, f"{log_name}")

    # /code/KPF-Pipeline/default
    kpfdrp_dir = utils.get_cfg(cfg, 'dirs', 'kpfdrp_dir')

    # The path to the reference fits files
    reference_fits_dir = utils.get_cfg(cfg, 'dirs', 'reference_fits_dir')

    # The docker container name
    containername = f"{utils.get_cfg(cfg, 'docker_names', 'nightly_container_base')}_{tm}"

    # database info,  config and ~/.pgpass file
    dbuser = utils.get_cfg(cfg, 'db', 'dbuser')
    dbname = utils.get_cfg(cfg, 'db', 'dbname')
    dbport, dbpass = utils.get_pgpass(dbname, dbuser)

    dockercmdscript = f'jobs/kpfnightly_drp_{uniq_str}'
    containerimage = 'kpf-drp:latest'
    recipe = 'recipes/kpf_drp.recipe'
    config = 'configs/keck_kpf_drp.cfg'

    logs_root_docker = logs_root.replace(data_drp, '/data/')
    stdout_log = f"{logs_root_docker}/{log_name}.stdout"

    # change to the code directory
    os.chdir(kpfdrp_dir)

    # log start
    utils.log_stub('Starting', 'Nightly-Processing', procdate, log)

    # write the docker script to be read by the docker container
    docker_bash_script = mk_docker_script()

    # check if by chance another container exists with same name
    utils.chk_rm_docker_container(containername, log)

    # start the docker process
    pid = run_docker()
    log.info(f'Docker Process ID: {pid}')

    # wait for the bash script to complete in the container
    log_chk = f"{logs_root}/{log_name}.stdout"
    utils.wait_container_complete(pid, containername, log, chk_log_name=log_chk)

    success = utils.chk_rm_docker_container(containername, log)

    utils.log_stub('Ending', 'Nightly-Processing', procdate, log)

