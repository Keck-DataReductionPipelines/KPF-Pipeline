Reprocessing
============

**Reprocessing Observations:** 
  
To process many nights of data, the recommended procedure is to use `reprocess.py`.  
This script has deletes old 2D/L1/L2/QLP/outliers/logs/logs_QLP files before starting reprocessing of a given night.  
Using the linux utility 'nice', it's execution is deprioritized (nice=15) so that regular processing isn't slowed down.  
It also procuces or appends to a log file (default name: reprocess_obs.log) with columns Datecode, Start Time, End Time, Run Time, and Version.
Dates that have been reprocessed with the same pipeline versions (according to the log file) are skipped. 
In production processing by the DRP development team, this command is in the xterm called *Reprocessing*.::

    reprocess_obs.py yyyymmdd YYYYMMDD

Here's the docstring showing all of the options.::

    usage: reprocess_obs.py [-h] [--ncpu NCPU] [--delete] [--verbose] [--force] [--logfile LOGFILE]
                             [--forward] [--not-nice] [--dry-run] [--local-tz LOCAL_TZ]
                            startdate enddate
    
    Reprocess KPF data over a date range.
    
    positional arguments:
      startdate            Start date in YYYYMMDD format
      enddate              End date in YYYYMMDD format
    
    options:
      -h, --help           show this help message and exit
      --ncpu NCPU          Number of CPUs to use
      --delete             Delete existing 2D/L1/L2/QLP files before reprocessing
      --verbose            Verbose stdout
      --force              Process even if datecode/version are listed in the logfile
      --logfile LOGFILE    Log file path
      --forward            Process datecodes in chronological order (reverse is default)
      --not-nice           Do not apply standard nice (=15) deprioritization
      --dry-run            Print commands without executing them
      --local-tz LOCAL_TZ  Local timezone (default: America/Los_Angeles)

One can also reprocess KPF data using the `kpf` command and a recipe.  
The command below launches 50 processes to reprocess L0 files into 2D/L1/L2 files for the date YYYYMMDD.:: 

    kpf --ncpu 50 --watch /data/L0/YYYYMMDD/ --reprocess -c configs/kpf_drp.config -r recipes/kpf_drp.recipe

**Reprocessing Masters:**

Reprocessing master files from yyyymmdd to YYYYMMDD is accomplished with a series of commands.  
First activate the kpf-masters conda environment.
From the ``cronjobs/`` directory, generate a series of shell scripts with the format ``runDailyPipelines_YYYYMMDD.sh`` 
You can then use the generated script ``runMastersPipeline_From_YYYYMMDD_To_YYYYMMDDDD.sh`` 
to run the dates you specified or use the ``parallel`` utility with a command like 
the one below for better control over compute resources.::

    conda activate /scr/doppler/conda/envs/kpf-masters
    cd cronjobs
    ./generateDailyRunScriptsBetweenTwoDates.pl yyyymmdd YYYYMMDD
    ls runDailyPipelines_202311*.sh | sort -r | parallel --delay 600 -j 5 --progress --bar --resume --joblog masters_reprocessing.log sh {}


The example above is for November, 2023 (yyyymmdd=20231101, YYYYMMDD=20231130).  
The name of the log file (``masters_reprocessing.log``) can be adjusted.  
It is not recommended to run more than 5-7 jobs (``-j`` option) at once to 
avoid I/O overload; staggered processing (``--delay 600``) helps. 
In production processing by the DRP development team, this command is in the 
xterm called *Masters Repocessing*.

**Quicklook reprocessing -- qlp_parallel.py:**

For a daterange from yyyymmdd to YYYYMMDD with NCPU cpus.::

    ./scripts/qlp_parallel.py yyyymmdd YYYYMMDD --ncpu NCPU --l0 --2d --l1 --l2 --master

The full description is here::

    Description:
      This command line script uses the 'parallel' utility to execute the recipe 
      called 'recipes/quicklook_match.recipe' to generate standard Quicklook data 
      products.  The script selects all KPF files based on their
      type (L0/2D/L1/L2/master) from the standard data directory using a date 
      range specified by the parameters start_date and end_date.  L0 files are 
      included if the --l0 flag is set or none of the --l0, --2d, --l1, --l2
      flags are set (in which case all data types are included).  The --2d, 
      --l1, and --l2 flags have similar functions.  The script assumes that it
      is being run in Docker and will return with an error message if not. 
      If start_date is later than end_date, the arguments will be reversed 
      and the files with later dates will be processed first.
      
      Invoking the --print_files flag causes the script to print filenames
      but not create QLP data products.
      
      The --ncpu parameter determines the maximum number of cores used.  
      
      The following feature is not operational if this script is run inside of 
      a Docker container: If the --load parameter (a percentage, e.g. 90 = 90%) 
      is set to a non-zero value, this script will be throttled so that no new 
      files will have QLPs processed until the load is below that value.  Note 
      that throttling works in steady state; it is possible to overload the 
      system with the first set of jobs if --ncpu is set too way high.  

    Arguments:
      start_date     Start date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS
      end_date       End date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS

    Options:
      --l0           Select all L0 files in date range
      --2d           Select all 2D files in date range
      --l1           Select all L1 files in date range
      --l2           Select all L2 files in date range
      --master       Select all master files in date range
      --ncpu         Number of cores used for parallel processing; default=10
      --load         Maximum load (1 min average); default=0 (only activated if !=0)
      --print_files  Display file names matching criteria, but don't generate Quicklook plots
      --help         Display this message
   
    Usage:
      python qlp_parallel.py YYYYMMDD.SSSSS YYYYMMDD.SSSSS --ncpu NCPU --load LOAD --l0 --2d --l1 --l2 --master --print_files
    
    Examples:
      ./scripts/qlp_parallel.py 20230101.12345.67 20230101.17 --ncpu 50 --l0 --2d
      ./scripts/qlp_parallel.py 20240501 20240505 --ncpu 150 --load 90


**Reprocess specific observations -- kpf_slowtouch.sh:**

Individual observations can be reprocessed by touching the L0 files, or touching
the 2D/L1/L2 files to start reprocessing at a later stage. To reprocess a set 
of files, use the script `kpf_slowtouch.sh`.  Files are touched slowly 
(usually with 0.2 sec between touching individual files) to avoid overloading 
the file event triggers system that initiate reprocessing of specific files.::

    ./scripts/kpf_slowtouch.sh

This script is used to touch a list of KPF L0 files that have names like 
KP.20230623.12345.67.fits.  This is useful to initiate reprocessing 
using the KPF DRP.  The list of L0 files can be provided in multiple ways:

#. As command-line arguments when invoking the script.
#. In the first column of a CSV file specified with the -f option. This is useful for CSV files with a large set of L0 filenames downloaded from Jump.  Such files might have double quotes around the L0 filename, which the script will remove when appropriate.
#. All filenames in a directory specified with the -d option.

The (optional) command-line options are::

    -f <filename>       : The script will read the KPF L0 filenames 
                          from the first column of a CSV with the name <filename>.
                          Useful for lists of L0 files downloaded from Jump.
    -d <directory>      : Adds every file in <directory> to the list of L0 files.
    -p <path>           : Sets the L0 path to <path>.
                          Default value: /data/kpf/L0
    -s <sleep_interval> : Sets the interval between file touches.
                          Default value: 0.2 [sec]
    -e                  : Echo the touch commands instead of executing them.

Some example uses of this script are:

#. To provide filenames using command line arguments: ``./kpf_slowtouch.sh KP.20230623.12345.67.fits KP.20230623.12345.68.fits``
#. To provide filenames using a CSV file: ``./kpf_slowtouch.sh -f filenames.csv``
#. To provide files listed in a directory: ``./kpf_slowtouch.sh -d /path/to/directory``
#. To change the default L0 path and sleep interval between touches: ``./kpf_slowtouch.sh KP.20230623.12345.67.fits -p /new/path -s 0.5``
#. To echo the touch commands instead of executing them: ``./kpf_slowtouch.sh KP.20230623.12345.67.fits -e``

**Monitoring processing progress -- kpf_processing_progress.py:**

Print the status of processing for a date range::

    ./scripts/kpf_processing_progress.py YYYYMMDD YYYYMMDD

The full description is here::

    Description:
      This script is used to assess the status and progress of processing KPF data.
      It searches over a range of dates specified by the first two arguments which are 
      of the form YYYYMMDD.  For each date (with /data/kpf/L0/YYYYMMDD as the 
      assumed L0 directory), it examines each L0 file and the associated 2D/L1/L2 
      files in their related directories.  If the first argument is a date after the 
      second argument, then the dates are printed in reverse chronological order (later 
      dates first).  The output of this script is a table with columns indicating the 
      date for each row, the most recent modification date for and L0 file in that 
      directory, the fraction of 2D files processed, the fraction of L1 files processed, 
      and the fraction of L2 files processed.  Sample output is shown below.
      
      > ./scripts/kpf_processing_progress.py 20231231 20230101 --current_version 2.5

      
      DATECODE | LAST L0 MOD DATE | 2D PROCESSING  | L1 PROCESSING  | L2 PROCESSING 
      ------------------------------------------------------------------------------
      20231221 | 2023-12-21 10:18 |  256/256  100% |  254/256   99% |  229/230   99%
      20231220 | 2023-12-20 16:00 |  342/342  100% |  342/342  100% |  315/315  100%
      20231219 | 2023-12-19 16:00 |  406/406  100% |  406/406  100% |  377/379   99%
      20231218 | 2023-12-18 16:00 |  531/531  100% |  528/531   99% |  501/504   99%
      20231217 | 2023-12-17 16:00 |  524/524  100% |  524/524  100% |  497/497  100%
      20231216 | 2023-12-16 16:00 |  527/527  100% |  524/527   99% |  497/500   99%
      
      The following criteria are used to determine if 2D/L1/L2 files are "processed":
      
          - not in the junk file list ('/data/kpf/reference/Junk_Observations_for_KPF.csv');
            if the file is missing, all files are assumed to not be junk
          - have the Green, Red, or CaHK extension present in the L0 file
          - not a Dark or Bias exposure [only applied to L2 files]
          - the 2D/L1/L2 exists
          - the modification time of the 2D/L1/L2 file is later than the 
            modification time of the associated L0 file
          - the DRP version number is equal to or greater than the current DRP version 
            number of the master branch on Github [only if --check_version option 
            selected]
      
                    #    - not junk
                    #    - Green, Red, or CaHK extension present
                    #    - not a Dark or Bias exposure
                    #    - file present
                    #    - L2 modification time more recent than L0 modification time
                    #    - current DRP version number (if check_version option selected)
      
      Command-line options listed below enable touching of the L0 files associated 
      with 2D/L1/L2 files that are not present, printing those filenames, printing the 
      filenames of the 2D/L1/L2 files themselves, and turning on the DRP version check.

    Options:
      --help             Display this message
      --print_files      Display missing file names (or files that fail other criteria)
      --print_files_2D   Display missing 2D file names (or files that fail other criteria)
      --print_files_L1   Display missing L1 file names (or files that fail other criteria)
      --print_files_L2   Display missing L2 file names (or files that fail other criteria)
      --touch_files      Touch the base L0 files of missing 2D/L1/L2 files
      --check_version    Checks that each 2D/L1/L2 file has the current Git version for the KPF-Pipeline
      --current_version  The current version of determining completion status; e.g. --current version 2.5
   
    Usage:
      kpf_processing_progress.py YYYYMMDD [YYYYMMDD] [--print_files] [--print_files_2D] [--print_files_L1] [--print_files_L2] [--touch_files] [--check_version]
   
    Example:
      ./scripts/kpf_processing_progress.sh 20231114 20231231 --print_files
