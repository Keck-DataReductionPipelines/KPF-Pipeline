TSDB Ingestion
##############

The code for TSDB ingestion is described :doc:`here </docs/source/tsdb/analyzetimeseries>`.
During production processing, data are ingested into the TSDB in the three ways described below. 
Note that 'ingestion' by all of the above mechanisms is a process retrieves data from all available L0/2D/L1/L2 files for the ObsID 
(e.g., ``KP.20200101.12345.67``) of a created/modified file, not just the created/modified file itself.
When the argument ``force_ingest=False``, ingestion is limited to observations with modification times of one of their 
L0/2D/L1/L2 files that is later than the modification times recorded in the TSDB from the last ingestion.
When ``force_ingest=True``, an observation is ingested regardless of the modification times.

First, the script ``scripts/ingest_watch_kpf_tsdb.py`` has methods for watching and for period ingestion.  
This script is designed to ingest files in near real time as they are created/modified by various processes.
The 'watcher' threads of this script detects creation and modification events for files with names of the form 
`KP*.fits` in the directories ``/data/kpf/L0/``, ``/data/kpf/2D/``, ``/data/kpf/L1/``, and ``/data/kpf/L2/`` 
(in Docker, these directories are ``/data/L0/``, ``/data/2D/``, ``/data/L1/``, and ``/data/L2/``).
Observservations are enqueued for ingestion when the created/modified file has been quiet for at least 3 seconds 
(to avoid ingesting partially written files).
The enqueued observations are ingested in batches every 30 seconds with ``force_ingest=False`` (``force_ingest`` is explained below.)

Second, the 'periodic' thread of ``scripts/ingest_watch_kpf_tsdb.py`` scans the data directories listed above every 48 hours with ``force_ingest=True``.  
This catches any files that were missed for whatever reason by the watcher threads.

Third, during reprocessing with the ``scripts/reprocess_obs.py`` script, observations for a datecode (eg., `20251201`) 
are ingested with ``force_ingest=True`` after the main recipe has been run on that datecode
and again with ``force_ingest=True`` after the drift correction recipe has been run.
