####################################################################
# Python script to run complete masters pipeline inside container.
# Input date of data to process is required:
# export PROCDATE=yyyymmdd
####################################################################

import sys
import os
import re
import subprocess
import signal
import psutil

iam = "kpf_masters_pipeline.py"
iam_version = "1.0"

print("iam =",iam)
print("iam_version =",iam_version)

iam_pid = os.getpid()

print("iam_pid =",iam_pid)


# Get date from command-line argument.

exitcode = 0

try:
    procdate = (sys.argv)[1]
except:
    print("*** Error: procdate is not specified on command line; quitting...")
    exitcode = 64
    exit(exitcode)


def form_output_file_for_python_script(pythonscript):

    filename_match = re.match(r".+/(.+)\.py", pythonscript)

    output_file = None

    try:
        output_file = filename_match.group(1) + f"_{procdate}.out"

    except:
        print("*** Error: could not form output file from python script {pythonscript}; continuing...")

    return output_file


# Define required inputs.

recipe_2d = '/code/KPF-Pipeline/recipes/kpf_masters_2D.recipe'
recipe_stacks = '/code/KPF-Pipeline/recipes/kpf_masters_stacks.recipe'
config = '/code/KPF-Pipeline/configs/kpf_masters_drp.cfg'
pythonscript = 'scripts/make_smooth_lamp_pattern_new.py'
pylogfile = form_output_file_for_python_script(pythonscript)
pythonscript2 = 'scripts/reformat_smooth_lamp_fitsfile_for_kpf_drp.py'
pylogfile2 = form_output_file_for_python_script(pythonscript2)
pythonscript3 = 'database/scripts/cleanupMastersOnDiskAndDatabaseForDate.py'
pylogfile3 = form_output_file_for_python_script(pythonscript3)
pythonscript4 = 'database/scripts/registerCalFilesForDate.py'
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
pythonscript_run_analysis = 'cronjobs/run_analysis_for_masters.py'
pylogfile_run_analysis = form_output_file_for_python_script(pythonscript_run_analysis)
pythonscript_registerCalFilesForDate = 'database/scripts/registerCalFilesForDate.py'
pylogfile_registerCalFilesForDate = form_output_file_for_python_script(pythonscript_registerCalFilesForDate)


# Commands to execute inside container.

bash_cmds = [
             f"mkdir -p /data/2D/{procdate}",
             f"mkdir -p /data/L1/{procdate}",
             f"mkdir -p /data/L2/{procdate}",
             f"mkdir -p /data/logs/{procdate}",
             f"mkdir -p /data/masters/pool",
             f"git config --global --add safe.directory /code/KPF-Pipeline",
             f"test -r /data/L0/{procdate}",
             f"rm -rf /data/masters/{procdate}",
             f"rm -rf /data/masters/wlpixelfiles/*kpf_{procdate}*",
             f"rm -rf /data/analysis/{procdate} 2>/dev/null || true",
             f"rm -rf /data/masters/pool/kpf_{procdate}*",
             f"make init",
             f"kpf --ncpus 16 --reprocess /data/L0/{procdate}/ -r {recipe_2d} -c {config}",
             f"kpf --date {procdate} -r {recipe_stacks} -c {config}",
             f"python {pythonscript} /data/masters/pool/kpf_{procdate}_master_flat.fits /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits",
             f"python {pythonscript2} /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits /data/masters/pool/kpf_{procdate}_master_flat.fits /data/masters/pool/kpf_{procdate}_smooth_lamp.fits",
             f"rm -f /data/masters/pool/kpf_{procdate}_smooth_lamp_orig.fits",
             f"python {pythonscript3} {procdate}",
             f"mkdir -p /masters/{procdate}",
             f"sleep 1",
             f"cp -pf /data/masters/pool/kpf_{procdate}* /masters/{procdate}",
             f"cp -pf /data/logs/{procdate}/pipeline_{procdate}.log /masters/{procdate}/pipeline_masters_drp_l0_{procdate}.log || true",
             f"python {pythonscript4} {procdate}",
             f"mv /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate} /code/KPF-Pipeline/{pylogfile4}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile}  /masters/{procdate}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile2} /masters/{procdate}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile3} /masters/{procdate}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile4} /masters/{procdate}",
             f"rm /code/KPF-Pipeline/{pylogfile}",
             f"rm /code/KPF-Pipeline/{pylogfile2}",
             f"rm /code/KPF-Pipeline/{pylogfile3}",
             f"rm /code/KPF-Pipeline/{pylogfile4}",
             f"mkdir -p /data/masters/{procdate}",
             f"cp -pf /masters/{procdate}/kpf_{procdate}_master_flat.fits /data/masters/{procdate}",
             f"kpf -r {recipe_create_order_trace}  -c {config_create_order_trace} --date {procdate}",
             f"cp -pf /data/masters/{procdate}/*.csv /masters/{procdate}",
             f"kpf -r {recipe_create_order_rectification}  -c {config_create_order_rectification} --date {procdate}",
             f"cp -pf /data/masters/{procdate}/kpf_{procdate}_master_flat_*.fits /masters/{procdate}",
             f"kpf -r {recipe_create_order_mask}  -c {config_create_order_mask} --date {procdate}",
             f"cp -pf /data/masters/{procdate}/*order_mask.fits /masters/{procdate}",
             f"cp -pf /data/logs/{procdate}/pipeline_{procdate}.log /masters/{procdate}/pipeline_order_trace_{procdate}.log",
             f"cp -pfr /masters/{procdate}/kpf_{procdate}*.fits /data/masters/{procdate}",
             f"kpf --ncpus 32 --reprocess /data/masters/{procdate}/ --masters -r {recipe_l1} -c {config_l1} ",
             f"sleep 10",
             f"cp -pf /data/masters/{procdate}/* /masters/{procdate}",
             f"mkdir -p /masters/{procdate}/{logssubdir}",
             f"cp -pf /data/logs/{procdate}/kpf_{procdate}_*.log /masters/{procdate}/{logssubdir}",
             f"kpf -r {recipe_wls_auto}  -c {config_wls_auto} --date {procdate}",
             f"rm /masters/{procdate}/*master_WLS*",
             f"cp -pf /data/masters/{procdate}/*master_WLS* /masters/{procdate}",
             f"mkdir -p /masters/{procdate}/wlpixelfiles",
             f"cp -pf /data/masters/wlpixelfiles/*kpf_{procdate}* /masters/{procdate}/wlpixelfiles",
             f"cp -pf /code/KPF-Pipeline/pipeline_{procdate}.log /masters/{procdate}/pipeline_wls_auto_{procdate}.log",
             f"rm /code/KPF-Pipeline/pipeline_{procdate}.log",
             f"mkdir -p /data/analysis/{procdate}",
             f"python {pythonscript_run_analysis} {procdate}",
             f"cp -pfr /data/analysis/{procdate}/* /masters/{procdate}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile_run_analysis} /masters/{procdate}",
             f"#rm /code/KPF-Pipeline/{pylogfile_run_analysis}",
             f"python {pythonscript_registerCalFilesForDate} {procdate}",
             f"cp -pf /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate} /masters/{procdate}",
             f"#rm /code/KPF-Pipeline/{pylogfile_registerCalFilesForDate}"
            ]


########################################################################################################
# Functions.
########################################################################################################

def is_only_numbers(input_string):
    """Checks if a string contains only digits."""
    return bool(re.match(r'^\d+$', input_string))


def execute_command(cmd):

    '''
    Execute command.
    '''

    code_to_execute_args = cmd.split()

    print("execute_command: code_to_execute_args =",code_to_execute_args)

    if code_to_execute_args[0][0] == "#":
        print("Skipping commented-out line...")
        return None


    # Execute code_to_execute.

    pid = None

    if code_to_execute_args[0] == "kpf":

        print("Executing kpf command...")

        code_to_execute_object = subprocess.Popen(code_to_execute_args,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL,
                                                  text=True)

        pid = code_to_execute_object.pid

        print("pid =",pid)

        code_to_execute_object.wait()

    elif code_to_execute_args[0] == "cp" or code_to_execute_args[0] == "rm":

        print("Executing with shell=True...")

        code_to_execute_object = subprocess.run(cmd,
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL,
                                                shell=True,
                                                text=True)

    else:

        print("Capturing output...")

        code_to_execute_object = subprocess.run(code_to_execute_args,
                                                text=True,
                                                capture_output=True)

        code_to_execute_stdout = code_to_execute_object.stdout

        code_to_execute_stderr = code_to_execute_object.stderr

        if code_to_execute_args[0] == "python":

            pythonscript = code_to_execute_args[1]

            output_file = form_output_file_for_python_script(pythonscript)

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
            print("code_to_execute_stdout =\n",code_to_execute_stdout)
            print("code_to_execute_stderr (should be empty since STDERR is combined with STDOUT) =\n",code_to_execute_stderr)


    # Print return code.

    print("returncode =",code_to_execute_object.returncode)


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
                        print(f"Process with PID {child_pid} killed successfully.")
                    except ProcessLookupError:
                        print(f"Process with PID {child_pid} not found.")
                    except Exception as e:
                        print(f"Error killing process: {e}")

            except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
                print(f"Could not extract child_pid,child_name,child_status from child={child}; continuing...")
            except:
                print(f"Could not extract child_pid,child_name,child_status from child={child}; continuing...")


########################################################################################################
# Main program.
########################################################################################################

if __name__ == '__main__':

    if is_only_numbers(procdate) and len(procdate) == 8:
        print("procdate =",procdate)
    else:
        print("*** Error: procdate is not yyyymmdd format; quitting...")
        print(f"procdate={procdate}")
        exitcode = 64
        exit(exitcode)

    for cmd in bash_cmds:

        execute_command(cmd)


    print("Terminating with exitcode = ",exitcode)
    exit(exitcode)
