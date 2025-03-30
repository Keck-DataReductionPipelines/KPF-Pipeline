Production Processing
=====================

The following commands should be run from separate Docker containers to start production processing of KPF data.

**Main processing threads:**

Launch 50 processes to watch for L0 files and process them into 2D/L1/L2 files::

    kpf --ncpu 50 --watch /data/L0/ -c configs/kpf_drp.config -r recipes/kpf_drp.recipe

**Quicklook processing threads:** 

Launch QLP instances for all data levels with the default recipe::

    ./scripts/launch_qlp.sh

Alternatively, launch QLP instances for only recent observations::

    ./scripts/launch_qlp.sh --only_recent

  
**Ingestion for Observational Database:**
  
Start a script that will watch for new L0/2D/L1/L2 files and ingest them.  
Another thread of the script will periodically scan the data directories to search for files 
(or updates that were missed with the watch thread) to ingest.  
Periodic scans start one hour after the previous one completed.::

    ./scripts/ingest_dates_kpf_tsdb.py  

The above script will take care of most ingestion needs.  To ingest from date 
yyyymmdd to YYYYMMDD, use::

    ./scripts/ingest_dates_kpf_tsdb.py yyyymmdd YYYYMMDD

**Generate Time Series Plots**: 

This script will generate time series plots of telemetry and other information on regular intervals using the Observational Database::

    ./scripts/generate_time_series_plots.py

Other Processing Tasks
**********************

**Reprocessing L0 files:** 
  
Launch 50 processes to reprocess L0 files into 2D/L1/L2 files for the date YYYYMMDD::

    kpf --ncpu 50 /data/L0/YYYYMMDD/ -c configs/kpf_drp.config -r recipes/kpf_drp.recipe

**Quicklook reprocessing:**

For a daterange from yyyymmdd to YYYYMMDD with NCPU cpus.::

    ./scripts/qlp_parallel.py yyyymmdd YYYYMMDD --ncpu NCPU --l0 --2d --l1 --l2 --master

Alternatively, launch QLP instances for only recent observations::

    ./scripts/launch_qlp.sh --only_recent

**Reprocess specific observations:**

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

Command-line options (all are optional)::

    -f <filename>       : The script will read the KPF L0 filenames 
                          from the first column of a CSV with the name <filename>.
                          Useful for lists of L0 files downloaded from Jump.
    -d <directory>      : Adds every file in <directory> to the list of L0 files.
    -p <path>           : Sets the L0 path to <path>.
                          Default value: /data/kpf/L0
    -s <sleep_interval> : Sets the interval between file touches.
                          Default value: 0.2 [sec]
    -e                  : Echo the touch commands instead of executing them.

Examples:
#. To provide filenames using command line arguments: `./kpf_slowtouch.sh KP.20230623.12345.67.fits KP.20230623.12345.68.fits`
#. To provide filenames using a CSV file: `./kpf_slowtouch.sh -f filenames.csv`
#. To provide files listed in a directory: `./kpf_slowtouch.sh -d /path/to/directory`
#. To change the default L0 path and sleep interval between touches: `./kpf_slowtouch.sh KP.20230623.12345.67.fits -p /new/path -s 0.5`
#. To echo the touch commands instead of executing them: `./kpf_slowtouch.sh KP.20230623.12345.67.fits -e`

**Monitoring processing progress**

Print the status of processing for a date range::

    ./scripts/kpf_processing_progress.py YYYYMMDD YYYYMMDD
