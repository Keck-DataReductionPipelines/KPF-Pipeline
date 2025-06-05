Processing KPF Observations
===========================

The following commands should be run from separate Docker containers to start production processing of KPF data. 

**Main processing threads:**

Launch 50 processes to watch for L0 files and process them into 2D/L1/L2 files.  In production processing by the DRP development team, this command is in the xterm called *Realtime Processing*.::

    kpf --ncpu 50 --watch /data/L0/ -c configs/kpf_drp.config -r recipes/kpf_drp.recipe

**Quicklook processing threads:** 

Launch QLP instances for all data levels with the default recipe.  The QLP instances should be split between two commands, one that looks at recent files (generated in the last day from realtime processing and one that covers reprocessed files from 1+ days ago, the latter is deprioritized using 'nice=10').  In production processing by the DRP development team, these two commands are in the xterm called *QLP --only_recent* and *QLP --not_recent*.::

    ./scripts/launch_qlp.sh --only_recent

    ./scripts/launch_qlp.sh --not_recent
 
**Time Series Database Ingestion:**
  
Start a script that will watch for new L0/2D/L1/L2 files and ingest them.  
Another thread of the script will periodically scan the data directories to search for files 
(or updates that were missed with the watch thread) to ingest.  
Periodic scans start one hour after the previous one completed.  
In production processing by the DRP development team, this command is in the xterm called *TSDB Ingestion*.::

    ./scripts/ingest_watch_kpf_tsdb.py  

**Generation of Time Series Plots**: 

This script will generate time series plots of telemetry and other information on regular intervals using the Observational Database.
In production processing by the DRP development team, this command is in the xterm called *TSDB Plots*.::

    ./scripts/generate_time_series_plots.py
