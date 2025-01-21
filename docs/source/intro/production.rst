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

    ./scripts/ingest_watch_kpf_tsdb.py

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

**Monitoring processing progress**

Print the status of processing for a date range::

    ./scripts/kpf_processing_progress.py YYYYMMDD YYYYMMDD
