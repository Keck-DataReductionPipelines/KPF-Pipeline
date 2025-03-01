import sys
import os
import subprocess
from glob import glob
from astropy.io import fits

pythonpath = os.environ['PYTHONPATH']

print("pythonpath =",pythonpath)

script_to_execute = "polly/tools/run_analysis_single.py"

ORDERLETS : list[str] = [
    "SCI1",
    "SCI2",
    "SCI3",
    "CAL"
    ]


TIMESOFDAY = ["morn", "eve", "night", "midnight"]


def find_L1_etalon_files(OBS_DATE: str, TIMEOFDAY: str) -> dict[str, list[str]]:
    """
    Locates relevant L1 files for a given date and time of day. At the moment
    it loops through all files and looks at the "OBJECT" keyword in their
    headers.

    TODO:
     - Don't just take every matching frame! There are three "blocks" of three
       etalon frames taken every morning (and evening?). Should take only the
       single block that is closest to the SoCal observations.
     - Use a database lookup (on shrek) to select files
    """

    all_files: list[str] = glob(f"/data/kpf/masters/{OBS_DATE}/*L1.fits")

    out_files: list[str] = []

    for f in all_files:
        object = fits.getval(f, "OBJECT")
        if "etalon" in object.lower():
            timeofday = object.split("-")[-1]
            if timeofday == TIMEOFDAY:

                if "WLS" in f:                 # Ensure WLS files are not selected.
                    continue

                out_files.append(f)

    return out_files


def build_command_line_args(input_file,output_dir,orderlet):

    '''
    Build command line.
    '''

    code_to_execute_args = ["python"]
    code_to_execute_args.append(script_to_execute)
    code_to_execute_args.append("-f")
    code_to_execute_args.append(input_file)
    code_to_execute_args.append("-o")
    code_to_execute_args.append(orderlet)
    code_to_execute_args.append("--outdir")
    code_to_execute_args.append(output_dir)
    code_to_execute_args.append("--spectrum_plot")
    code_to_execute_args.append("True")
    code_to_execute_args.append("--fsr_plot")
    code_to_execute_args.append("True")
    code_to_execute_args.append("--fit_plot")
    code_to_execute_args.append("True")

    print("code_to_execute_args =",code_to_execute_args)

    return code_to_execute_args


def execute_command(code_to_execute_args):

    '''
    Execute python script.
    '''

    print("execute_command: code_to_execute_args =",code_to_execute_args)


    # Execute code_to_execute.

    code_to_execute_object = subprocess.run(code_to_execute_args, capture_output=True, text=True)
    print("returncode =",code_to_execute_object.returncode)

    code_to_execute_stdout = code_to_execute_object.stdout
    print("code_to_execute_stdout =\n",code_to_execute_stdout)

    code_to_execute_stderr = code_to_execute_object.stderr
    if code_to_execute_stderr:
        print("code_to_execute_stderr =\n",code_to_execute_stderr)


if __name__ == '__main__':

    # Run Jake Pember's run_analysis_single for all relevant files
    # in the masters directory for a given data.  Loop over
    # relevant orderlets.

    ETALON_ANALYSIS_DATE = sys.argv[1]
    print("ETALON_ANALYSIS_DATE =",ETALON_ANALYSIS_DATE)

    sandbox = os.environ["KPFCRONJOB_SBX"]
    outdir = f"{sandbox}/analysis/{ETALON_ANALYSIS_DATE}"
    print("outdir =",outdir)

    execute_command(["which", "python"])
    execute_command(["python", "--version"])

    for TIMEOFDAY in ["morn", "eve", "night"]:

        # Find matching etalon files
        SPEC_FILES = find_L1_etalon_files(ETALON_ANALYSIS_DATE, TIMEOFDAY)

        if not SPEC_FILES:
            print(f"*** Warning: No input files for {ETALON_ANALYSIS_DATE} {TIMEOFDAY}")
            exitcode = 32
            print("exitcode = ",exitcode)
            exit(exitcode)

        for l1_file in SPEC_FILES:

            for orderlet in ORDERLETS:

                cmd = build_command_line_args(l1_file,outdir,orderlet)
                execute_command(cmd)


    exitcode = 0
    print("exitcode = ",exitcode)
    exit(exitcode)

