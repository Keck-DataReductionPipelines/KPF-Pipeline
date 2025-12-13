 #!/usr/bin/env python3

########################################################################
# Python script to run the complete masters pipeline or specified parts
# inside container, with options that are lacking in the original
# cronjobs/kpf_masters_pipeline.py for first-time masters processing.
# Input start and end observation-date range of the data to reprocess
# on the command line as positional arguments (YYYYMMDD format).
########################################################################

import sys
import os
import re
import pytz
import subprocess
import signal
import psutil
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from kpfpipe.tools.git_tools import get_git_tag, get_git_branch
import time


iam = "reprocess_masters.py"
iam_version = "1.0"

print("iam =",iam)
print("iam_version =",iam_version)

iam_pid = os.getpid()

print("iam_pid =",iam_pid)


########################################################################################################
# Methods.
########################################################################################################

def parse_args(STEPS):

    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Run complete masters pipeline or specified parts inside container.')
    parser.add_argument('startdate', type=str, help='Start observation date in YYYYMMDD format')
    parser.add_argument('enddate', type=str, help='End observation date in YYYYMMDD format (same as startdate for single date)')
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=STEPS,
        help="Which steps to run (default: all)",
    )
    parser.add_argument('--force', action='store_true', help='Process even if datecode/version are listed in the logfile')
    parser.add_argument('--dry-run',action='store_true',help='Dry run mode: print commands without executing them')
    parser.add_argument('--logfile', type=str, default='reprocess_masters.log', help='Log file path')
    parser.add_argument('--ncpu',type=int,default=1,help='Number of parallel observation-date processes (default = 1)')
    parser.add_argument('--forward', action='store_true', help='Process dates in chronological order (reverse is default)')
    parser.add_argument('--not-nice', action='store_true', help='Do not apply standard nice (=15) deprioritization')
    parser.add_argument('--local-tz', type=str, default='America/Los_Angeles',
                        help='Local timezone for logfile lines (default: America/Los_Angeles)')
    parser.add_argument('-v','--verbose', action='store_true', help='Print detailed messages during execution')
    args = parser.parse_args()\

    return args


def load_processed_dates(logfile, version):
    processed_dates = set()
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            for line in f:
                if line.startswith("Datecode") or "FAILED" in line:
                    continue
                datecode = line[:10].strip()
                log_version = line[67:77].strip()
                if log_version == version:
                    processed_dates.add(datecode)
    return processed_dates


def is_only_numbers(input_string):
    """Checks if a string contains only digits."""
    return bool(re.match(r'^\d+$', input_string))


def get_dates_in_range(start_date_str, end_date_str, date_format="%Y%m%d"):

    """
    Generates a list of dates within a specified range.

    Args:
        start_date_str (str): The start date string.
        end_date_str (str): The end date string.
        date_format (str): The format of the date strings (default: "%Y%m%d").

    Returns:
        list: A list of datetime.date objects representing the dates in the range.
    """
    start_date = datetime.strptime(start_date_str, date_format).date()
    end_date = datetime.strptime(end_date_str, date_format).date()

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    return date_list


def form_output_file_for_python_script(pythonscript,procdate,verbose=False):

    filename_match = re.match(r".+/(.+)\.py", pythonscript)

    output_file = None

    try:
        output_file = filename_match.group(1) + f"_{procdate}.out"

        if verbose:
            print(f"output_file = {output_file}")

    except:
        print(f"*** Error: could not form output file from python script {pythonscript}; continuing...")

    return output_file


def execute_command(cmd,procdate,dryrun,fh,nice_prefix):

    '''
    Execute command.
    '''

    if dryrun:
        fh.write(f"Dryrun: would execute: {cmd}\n")
        return 0

    if cmd[0][0] == "#":
        fh.write("Skipping commented-out line: {cmd}...\n")
        return 0

    split_cmd = cmd.split()

    if len(nice_prefix) > 0:
        idx_cmd = 3
        code_to_execute_args = nice_prefix + split_cmd
    else:
        code_to_execute_args = split_cmd
        idx_cmd = 0

    fh.write(f"execute_command: code_to_execute_args = {code_to_execute_args}\n")


    # Execute code_to_execute.

    pid = None

    if code_to_execute_args[idx_cmd] == "kpf":

        fh.write("Executing kpf command...\n")

        code_to_execute_object = subprocess.Popen(code_to_execute_args,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL,
                                                  text=True)

        pid = code_to_execute_object.pid

        fh.write(f"pid = {pid}\n")

        code_to_execute_object.wait()

    elif code_to_execute_args[idx_cmd] == "cp" or code_to_execute_args[idx_cmd] == "rm":

        fh.write("Executing with shell=True...\n")

        code_to_execute_object = subprocess.run(cmd,
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL,
                                                shell=True,
                                                text=True)

    else:

        fh.write("Capturing output...\n")

        code_to_execute_object = subprocess.run(code_to_execute_args,
                                                text=True,
                                                capture_output=True)

        code_to_execute_stdout = code_to_execute_object.stdout

        code_to_execute_stderr = code_to_execute_object.stderr

        fh.write(f"code_to_execute_args[idx_cmd]={code_to_execute_args[idx_cmd]}\n")

        if code_to_execute_args[idx_cmd] == "python":

            pythonscript = code_to_execute_args[idx_cmd + 1]

            fh.write(f"pythonscript={pythonscript}\n")

            output_file = form_output_file_for_python_script(pythonscript,procdate)

            if output_file is not None:

                with open(output_file, "w") as f:
                    f.write("STDOUT:\n")
                    f.write(code_to_execute_stdout)
                    f.write("STDERR:\n")
                    if code_to_execute_stderr == "" or code_to_execute_stderr is None:
                        f.write("None\n")
                    else:
                        f.write(code_to_execute_stderr)

        else:
            fh.write(f"code_to_execute_stdout = {code_to_execute_stdout}\n")
            fh.write(f"code_to_execute_stderr = {code_to_execute_stderr}\n")


    # Print return code.

    returncode = code_to_execute_object.returncode

    fh.write(f"returncode = {returncode}\n")


    # Handle sleeping kpf process that is subsequently detached from this python code
    # and attached to parent init process (PPID=1).

    if pid is not None:

        init_pid = 1
        init_process = psutil.Process(init_pid)
        recursive=False
        children = init_process.children(recursive=recursive)
        for child in children:

            try:
                child_pid = child.pid
                child_name = child.name()
                child_status = child.status()

                if child_pid != iam_pid and child_name == 'kpf' and child_status == 'sleeping':

                    try:
                        # Send SIGKILL signal (9) for forceful termination
                        os.kill(child_pid, signal.SIGKILL)
                        fh.write(f"Process with PID {child_pid} killed successfully.\n")
                    except ProcessLookupError:
                        fh.write(f"Process with PID {child_pid} not found.\n")
                    except Exception as e:
                        fh.write(f"Error killing process: {e}\n")

            except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
                fh.write(f"Could not extract child_pid,child_name,child_status from child={child}; continuing...\n")
            except:
                fh.write(f"Could not extract child_pid,child_name,child_status from child={child}; continuing...\n")

    return returncode


#-------------------------------------------------------------------------------------------------------------
# Custom methods for parallel processing, taking advantage of multiple cores on machine.
#-------------------------------------------------------------------------------------------------------------

def run_single_core_job(procdates,bash_cmds_for_nights,dryrun,nparallel,local_tz,nice_prefix,index_thread):

    njobs = len(procdates)

    print("index_thread,njobs =",index_thread,njobs)

    thread_work_file = iam.replace(".py","_thread") + str(index_thread) + ".out"

    try:
        fh = open(thread_work_file, 'w', encoding="utf-8")
    except:
        print(f"*** Error: Could not open output file {thread_work_file}; quitting...")
        exit(64)

    fh.write(f"\nStart of run_single_core_job: index_thread={index_thread},dryrun={dryrun},nparallel={nparallel}\n")

    date_processed_list = []
    start_time_list = []
    end_time_list = []
    compute_time_list = []
    exitcode_list = []

    for index_job in range(njobs):

        index_core = index_job % nparallel
        if index_thread != index_core:
            continue

        procdate = procdates[index_job]
        bash_cmds = bash_cmds_for_nights[index_job]

        start_time = datetime.now(local_tz)

        fh.write(f"procdate={procdate},index_thread={index_thread}\n")
        #fh.write(f"bash_cmds={bash_cmds}\n")


        # Loop over bash command and execute them here.

        max_exit_code = 0

        for cmd in bash_cmds:
            exitcode = execute_command(cmd,procdate,dryrun,fh,nice_prefix)

            if exitcode > max_exit_code:
                max_exit_code = exitcode

        end_time = datetime.now(local_tz)
        compute_time = end_time - start_time
        compute_time_str = str(compute_time).split('.')[0]

        date_processed_list.append(procdate)
        start_time_list.append(start_time)
        end_time_list.append(end_time)
        compute_time_list.append(compute_time_str)
        exitcode_list.append(max_exit_code)

        fh.write(f"Loop end: index_job={index_job}\n")

        # End of loop over job index.


    fh.write(f"\nEnd of run_single_core_job: index_thread={index_thread}\n")

    fh.close()

    message = f"Finish normally for index_thread = {index_thread}"

    return message,date_processed_list,start_time_list,end_time_list,compute_time_list,exitcode_list


def execute_parallel_processes(procdates,bash_cmds_for_nights,dryrun,nparallel,local_tz,nice_prefix):

    if nparallel is None:
        nparallel = os.cpu_count()  # Use all available cores if not specified

    print("---->nparallel =",nparallel)

    with ProcessPoolExecutor(max_workers=nparallel) as executor:
        # Submit all tasks to the executor and store the futures in a list

        try:
           futures = [executor.submit(run_single_core_job,procdates,bash_cmds_for_nights,dryrun,nparallel,local_tz,nice_prefix,thread_index) for thread_index in range(nparallel)]
        except Exception as e:
            print(f"*** Error in lauching paralle jobs = {e}")

        # Iterate over completed futures and update progress
        for i, future in enumerate(as_completed(futures)):
            index = futures.index(future)  # Find the original index/order of the completed future
            print(f"Completed: {i+1} processes, lastly for index={index}")

    start_time_dict = {}
    end_time_dict = {}
    compute_time_dict = {}
    exitcode_dict = {}

    for future in futures:
        index = futures.index(future)
        try:
            print(future.result())

            message = future.result()[0]
            date_processed_list = future.result()[1]
            start_time_list = future.result()[2]
            end_time_list = future.result()[3]
            compute_time_list = future.result()[4]
            exitcode_list = future.result()[5]

            for d,s,e,c,x in zip(date_processed_list,start_time_list,end_time_list,compute_time_list,exitcode_list):
                start_time_dict[d] = s
                end_time_dict[d] = e
                compute_time_dict[d] = c
                exitcode_dict[d] = x

        except Exception as e:
            print(f"*** Error in thread index {index} = {e}")

    return start_time_dict,end_time_dict,compute_time_dict,exitcode_dict


########################################################################################################
# Main program.
########################################################################################################

def main():

    exitcode = 0


    # Database registration of products is ALWAYS done if steps involving stacks or beyond are done,
    # so it is not explicitly listed.  Database registration is NOT done if only the 2d step is specified.

    STEPS = ["2d",
             "stacks_etc",
             "order_stuff",
             "l12",
             "wls",
             "etalon"]

    args = parse_args(STEPS)
    git_tag = get_git_tag()
    git_branch = get_git_branch()
    local_tz = pytz.timezone(args.local_tz)

    dryrun = args.dry_run
    nparallel = args.ncpu
    logfile = args.logfile
    startdate = args.startdate
    enddate = args.enddate


    # The number of CPUs allocated for this script is in competition
    # with the number of CPUs allocated for the parallel kpf command.

    ncpus_machine = os.cpu_count()
    ncpus_kpf = min(16,int(ncpus_machine / nparallel))
    if ncpus_kpf == 0:
        ncpus_kpf = 1

    log_exists = os.path.isfile(args.logfile)
    logging.basicConfig(filename=args.logfile, level=logging.INFO,
                        format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not log_exists:
        with open(args.logfile, 'w') as f:
            f.write(f"{'Datecode':<10}  {'Starttime':<19}  {'Endtime':<19}  {'Runtime':<11}  {'Version':<10}  {'Exitcode':<10}\n")
        os.chmod(args.logfile, 0o666)

    print("git_tag =", git_tag)
    print("git_branch =", git_branch)
    print("dryrun =", dryrun)
    print("startdate =", startdate)
    print("enddate =", enddate)
    print("nparallel =", nparallel)
    print("logfile =", logfile)

    if is_only_numbers(startdate) and len(startdate) == 8:
        pass
    else:
        print("*** Error: startdate is not yyyymmdd format; quitting...")
        exitcode = 64
        exit(exitcode)

    if is_only_numbers(enddate) and len(enddate) == 8:
        pass
    else:
        print("*** Error: enddate is not yyyymmdd format; quitting...")
        exitcode = 64
        exit(exitcode)

    if int(startdate) > int(enddate):
        print("*** Error: startdate is greater than enddate; quitting...")
        exitcode = 64
        exit(exitcode)

    procdates = get_dates_in_range(startdate, enddate)
    for d in procdates:
        print(d.strftime("%Y%m%d"))

    if not args.forward:
        procdates = procdates[::-1]  # reversed order

    print(f"procdates = {procdates}")

    processed_dates = set()
    if not args.force:
        processed_dates = load_processed_dates(args.logfile, git_tag)

    nice_prefix = [] if args.not_nice else ['nice', '-n', '15']

    procdates_str = []
    bash_commands_for_nights_to_reprocess = []

    for d in procdates:

        procdate = d.strftime("%Y%m%d")

        if procdate in processed_dates:
            print(f"Skipping previously reprocessed observation date: {procdate}")
            continue

        print(f"=====> Now setting up reprocessing commands for observation date = {procdate}")


        # Define required inputs.

        recipe_2d = '/code/KPF-Pipeline/recipes/kpf_masters_2D.recipe'
        recipe_stacks = '/code/KPF-Pipeline/recipes/kpf_masters_stacks.recipe'
        config = '/code/KPF-Pipeline/configs/kpf_masters_drp.cfg'
        pythonscript = '/code/KPF-Pipeline/scripts/make_smooth_lamp_pattern_new.py'
        pylogfile = form_output_file_for_python_script(pythonscript,procdate,args.verbose)
        pythonscript2 = '/code/KPF-Pipeline/scripts/reformat_smooth_lamp_fitsfile_for_kpf_drp.py'
        pylogfile2 = form_output_file_for_python_script(pythonscript2,procdate,args.verbose)
        pythonscript3 = '/code/KPF-Pipeline/database/scripts/cleanupMastersOnDiskAndDatabaseForDate.py'
        pylogfile3 = form_output_file_for_python_script(pythonscript3,procdate,args.verbose)
        pythonscript4 = '/code/KPF-Pipeline/database/scripts/registerCalFilesForDate.py'
        # The following is special for the first run of database/scripts/registerCalFilesForDate.py
        pylogfile4 = f'registerCalFilesForDate_{procdate}_2D.out'
        recipe_create_order_trace = '/code/KPF-Pipeline/recipes/create_order_trace_files.recipe'
        config_create_order_trace = '/code/KPF-Pipeline/configs/create_order_trace_files.cfg'
        recipe_create_order_rectification = '/code/KPF-Pipeline/recipes/create_order_rectification_file.recipe'
        config_create_order_rectification = '/code/KPF-Pipeline/configs/create_order_rectification_file.cfg'
        recipe_create_order_mask = '/code/KPF-Pipeline/recipes/create_order_mask_file.recipe'
        config_create_order_mask = '/code/KPF-Pipeline/configs/create_order_mask_file.cfg'
        recipe_l1 = '/code/KPF-Pipeline/recipes/kpf_drp.recipe'
        config_l1 = '/code/KPF-Pipeline/configs/kpf_masters_l1.cfg'
        logssubdir = f'pipeline_masters_drp_l1_{procdate}.log'
        recipe_wls_auto = '/code/KPF-Pipeline/recipes/wls_auto.recipe'
        config_wls_auto = '/code/KPF-Pipeline/configs/wls_auto.cfg'
        pythonscript_run_analysis = '/code/KPF-Pipeline/cronjobs/run_analysis_for_masters.py'
        pylogfile_run_analysis = form_output_file_for_python_script(pythonscript_run_analysis,procdate,args.verbose)
        pythonscript_registerCalFilesForDate = '/code/KPF-Pipeline/database/scripts/registerCalFilesForDate.py'
        pylogfile_registerCalFilesForDate = form_output_file_for_python_script(pythonscript_registerCalFilesForDate,procdate,args.verbose)


        # Commands to execute inside container.
        # 1. Have chatbot create a --dryrun version first to verify correctness.
        # 2. Then past the bash_cmds list with notes reqeusting arguments that allow separating into different sections.
        # 3. Sections:
        # bash_cmds_2d = []
        # bash_check_2d = []                    # Were some 2D files created?
        # bash_remove_old_masters = []
        # bash_check_removed=[]                 # Were old masters removed?

        # bash_check3 = []
        # bash_check4 = []
        # bash_check5 = []
        # bash_check6 = []

        # The copying of files should be:
        #  Read L0 from file, don't copy to sandbox
        #  Write 2D to sandbox, don't copy again anywhere, delete after.
        #  The stacks are first written sandbox pool directory.
        #

        # More goals:
        # 1. Minimize copying of files.
        # 2. Write files directly to final destination when possible.
        # 3. Add checks after each major step to verify expected files were created.
        # 4. Separate commands into logical sections for better readability and maintainability.
        # 5. Do not recompute files unless necessary.

        bash_cmd_create_directories = [
            # Create all needed directories
            f"mkdir -p /data/2D/{procdate}",
            f"mkdir -p /data/L1/{procdate}",
            f"mkdir -p /data/L2/{procdate}",
            f"mkdir -p /data/logs/{procdate}",
            f"mkdir -p /data/masters/pool",
            # Allows git to work inside docker container
            f"git config --global --add safe.directory /code/KPF-Pipeline",
            # Checks for L0 dir existence.
            f"test -r /data/L0/{procdate}"
        ]

        bash_cmd_remove_previous_masters = [
            # Removes all previous masters from sandbox to more easily identify failures.
            f"rm -rf /data/masters/{procdate}",
            f"rm -rf /data/masters/wlpixelfiles/*kpf_{procdate}*",
            f"rm -rf /data/analysis/{procdate} 2>/dev/null || true",
            f"rm -rf /data/masters/pool/kpf_{procdate}*"
        ]

        bash_cmd_make_init = [
            # Install KPF and run recipes.
            f"make init"
        ]

        bash_cmd_create_2D = [
            # Masters 2D creation for all files.
            f"kpf --ncpus {ncpus_kpf} --reprocess /data/L0/{procdate}/ -r {recipe_2d} -c {config}"
        ]

        bash_cmd_create_stacks = [
            # Masters stacks creation; i.e., kpf_20241010_master_bias_autocal-bias.fits, flat, dark.
            f"kpf --date {procdate} -r {recipe_stacks} -c {config}"
        ]

        bash_cmd_smooth_lamp = [
            # Call python inside container to create smooth lamp pattern from the flats, _orig is only used temporarily.
            f"python {pythonscript} /data/masters/pool/kpf_{procdate}_master_flat.fits /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits",
            f"python {pythonscript2} /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits /data/masters/pool/kpf_{procdate}_master_flat.fits /data/masters/pool/kpf_{procdate}_smooth_lamp.fits",
            f"rm -f /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits"
        ]

        bash_cmd_cleanup_disk_and_database = [
            # This Python script command does the following:
            # 1. Clean out all files and directories in /masters/<yyyyymmdd> for given date,
            #    and also remove the top-level directory /masters/<yyyyymmdd>.
            # 2. Remove records in the CalFiles table of the KPF operations database
            #    that are associated with startdate = <yyyymmdd>.
            f"python {pythonscript3} {procdate}"
        ]

        bash_cmd_copy_masters_unnecessarily = [
            # Cleanup masters on disk and database for this date.
            f"mkdir -p /masters/{procdate}",
            f"sleep 1",
            # Copy masters to permanent location. Can we write these to final location eventually?
            f"cp -pf /data/masters/pool/kpf_{procdate}* /masters/{procdate}",
            f"cp -pf /data/logs/{procdate}/pipeline_{procdate}.log /masters/{procdate}/pipeline_masters_drp_l0_{procdate}.log || true"
        ]

        bash_cmd_register_cal_files = [
            # Register cal files for this date (2D files).
            # Update the database log files names to reflect the special log file name for this first run.
            f"python {pythonscript4} {procdate}",
            f"mv -f /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate} /code/KPF-Pipeline/{pylogfile4}",
            # Copy log files to masters area.
            f"cp -pf /code/KPF-Pipeline/{pylogfile}  /masters/{procdate}",
            f"cp -pf /code/KPF-Pipeline/{pylogfile2} /masters/{procdate}",
            f"cp -pf /code/KPF-Pipeline/{pylogfile3} /masters/{procdate}",
            f"cp -pf /code/KPF-Pipeline/{pylogfile4} /masters/{procdate}",
            # Remove originals.
            f"rm /code/KPF-Pipeline/{pylogfile}",
            f"rm /code/KPF-Pipeline/{pylogfile2}",
            f"rm /code/KPF-Pipeline/{pylogfile3}",
            f"rm /code/KPF-Pipeline/{pylogfile4}"
        ]

        bash_create_order_trace_rectification_mask_and_l1_wls_analysis = [
            # Create order trace, rectification, and mask files.
            # Normally /masters is the production directory and /data/masters is the sandbox.
            # HTI todo: Double check we are not double copying using pool dirctory
            f"mkdir -p /data/masters/{procdate}",
            # Copy masters back to sandbox for further processing. Isn't this done already?
            f"cp -pf /masters/{procdate}/kpf_{procdate}_master_flat.fits /data/masters/{procdate}",
            # Order trace file creation.
            f"kpf -r {recipe_create_order_trace}  -c {config_create_order_trace} --date {procdate}",
            f"cp -pf /data/masters/{procdate}/*.csv /masters/{procdate}",
            # Order rectification file creation.
            f"kpf -r {recipe_create_order_rectification}  -c {config_create_order_rectification} --date {procdate}",
            f"cp -pf /data/masters/{procdate}/kpf_{procdate}_master_flat_*.fits /masters/{procdate}",
            # Order mask file creation.
            f"kpf -r {recipe_create_order_mask}  -c {config_create_order_mask} --date {procdate}",
            f"cp -pf /data/masters/{procdate}/*order_mask.fits /masters/{procdate}",
            f"cp -pf /data/logs/{procdate}/pipeline_{procdate}.log /masters/{procdate}/pipeline_order_trace_{procdate}.log",
            # Do we really need to copy this again. Add existence statements and don't copy if not created?
            f"cp -pfr /masters/{procdate}/kpf_{procdate}*.fits /data/masters/{procdate}"
        ]

        bash_create_l1 = [
            # L1 to L2 masters creation. HTI todo: Can we write these directly to final location? Yes, change reprocess input path.
            f"kpf --ncpus {ncpus_kpf} --reprocess /data/masters/{procdate}/ --masters -r {recipe_l1} -c {config_l1} ",
            f"sleep 10",
            # Copy L2 masters to permanent location. or change above to write directly to final location.
            f"cp -pf /data/masters/{procdate}/* /masters/{procdate}",
            # Make logs subdirectory.
            f"mkdir -p /masters/{procdate}/{logssubdir}",
            f"cp -pf /data/logs/{procdate}/kpf_{procdate}_*.log /masters/{procdate}/{logssubdir}"
        ]

        bash_create_wls = [
            # Wavelength solution creation to sandbox (config sets destination.)
            f"kpf -r {recipe_wls_auto}  -c {config_wls_auto} --date {procdate}",
            # Remove old masters before copying new WLS files. (or change to write directly to final location above and remove this next line.)
            f"rm /masters/{procdate}/*master_WLS*",
            # Copy WLS files and log to permanent location. (remove is direct write above).
            f"cp -pf /data/masters/{procdate}/*master_WLS* /masters/{procdate}"
        ]

        bash_unnecessary_moves = [
            # if output is specified to /masters, then we don't need the copy.
            f"mkdir -p /masters/{procdate}/wlpixelfiles",
            f"cp -pf /data/masters/wlpixelfiles/*kpf_{procdate}* /masters/{procdate}/wlpixelfiles",
            f"cp -pf /code/KPF-Pipeline/pipeline_{procdate}.log /masters/{procdate}/pipeline_wls_auto_{procdate}.log",
            # If written directly, then no need to copy.
            f"rm /code/KPF-Pipeline/pipeline_{procdate}.log"
        ]

        bash_run_etalon_analysis =[
            # Running etalon analysis setup, including polly.
            f"mkdir -p /data/analysis/{procdate}",
            f"python {pythonscript_run_analysis} {procdate}",
            # Write directly to final location and remove copy commands.
            f"cp -pfr /data/analysis/{procdate}/* /masters/{procdate}",
            f"cp -pf /code/KPF-Pipeline/{pylogfile_run_analysis} /masters/{procdate}",
            f"rm /code/KPF-Pipeline/{pylogfile_run_analysis}"
        ]

        bash_register_database2= [
            f"python {pythonscript_registerCalFilesForDate} {procdate}",
            # Can we write directly to final location?
            f"cp -pf /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate} /masters/{procdate}",
            # Confirm this is not running and then remove.
            f"rm /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate}"
        ]


        # Combine relevant bash commands into a single list.

        bash_cmds = []

        bash_cmd_2d = bash_cmd_create_directories + \
                    bash_cmd_remove_previous_masters + \
                    bash_cmd_make_init + \
                    bash_cmd_create_2D

        bash_cmd_stacks_etc = bash_cmd_create_stacks + \
                              bash_cmd_smooth_lamp + \
                              bash_cmd_cleanup_disk_and_database + \
                              bash_cmd_copy_masters_unnecessarily + \
                              bash_cmd_register_cal_files

        bash_cmd_order_stuff = bash_create_order_trace_rectification_mask_and_l1_wls_analysis

        bash_cmd_l12 = bash_create_l1

        bash_cmd_wls = bash_create_wls + \
                       bash_unnecessary_moves

        bash_cmd_etalon = bash_run_etalon_analysis

        STEPS_DICT = {"2d": bash_cmd_2d,
                      "stacks_etc": bash_cmd_stacks_etc,
                      "order_stuff": bash_cmd_order_stuff,
                      "l12": bash_cmd_l12,
                      "wls": bash_cmd_wls,
                      "etalon": bash_cmd_etalon}


        for step_name in STEPS:

            if step_name in args.steps:
                bash_cmds += STEPS_DICT[step_name]


        if len(args.steps) == 1 and args.steps[0] == "2d":
            pass
        else:
            bash_cmds += bash_register_database2


        # Aggregate bash commands for given night.

        procdates_str.append(procdate)
        bash_commands_for_nights_to_reprocess.append(bash_cmds)


        # End of loop over procdate(s).

    if len(procdates_str) == 0:

        print("No observation dates to reprocess (hint: may need to use --force)")

    else:

        ################################################################################
        # Execute bash command for nights in parallel.  The number of parallel processes
        # competes for cores with the number of cpus allocated for kpf commands.
        # If nparallel = 1, then bypass concurrent.futures for ease in debugging.
        ################################################################################

        if nparallel > 1:
            start_time_dict,end_time_dict,compute_time_dict,exitcode_dict = \
                execute_parallel_processes(procdates_str,
                                           bash_commands_for_nights_to_reprocess,
                                           dryrun,
                                           nparallel,
                                           local_tz,
                                           nice_prefix)
        else:
            if args.verbose:
                print(f"Calling run_single_core_job directly...")

            thread_index = 0
            message,date_processed_list,start_time_list,end_time_list,compute_time_list,exitcode_list = \
                run_single_core_job(procdates_str,
                                    bash_commands_for_nights_to_reprocess,
                                    dryrun,
                                    nparallel,
                                    local_tz,
                                    nice_prefix,
                                    thread_index)

            start_time_dict = {}
            end_time_dict = {}
            compute_time_dict = {}
            exitcode_dict = {}

            for d,s,e,c,x in zip(date_processed_list,start_time_list,end_time_list,compute_time_list,exitcode_list):
                start_time_dict[d] = s
                end_time_dict[d] = e
                compute_time_dict[d] = c
                exitcode_dict[d] = x


        # If not dry run, then log the execution metadata.

        if dryrun:
            exitcode = 0
        else:

            datecodes = start_time_dict.keys()

            for datecode in datecodes:
                start_time = start_time_dict[datecode]
                end_time = end_time_dict[datecode]
                compute_time_str = compute_time_dict[datecode]
                status = exitcode_dict[datecode]
                tf = '%Y-%m-%d %H:%M:%S'
                logging.info(f"{datecode:<10}  " +
                             f"{start_time.strftime(tf)}  " +
                             f"{end_time.strftime(tf)}  " +
                             f"{compute_time_str:<11}  " +
                             f"{git_tag:<10}  " +
                             f"{status:<10}")

                if args.verbose:
                    print(f"Processing complete for {datecode}.")

            os.chmod(args.logfile, 0o666)

    print("Terminating with exitcode = ",exitcode)
    exit(exitcode)


if __name__ == '__main__':
    main()
