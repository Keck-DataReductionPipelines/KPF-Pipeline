Example of processing data from a single night
==============================================

The part of the pipeline that generates stacks for master calibration files
is not yet optimized to run on personal computers due to extremely high RAM requirements.
It is easiest to download a directory of master calibration files for the night
that you would like to process. The master calibration directory should be named 
by it's date (YYYYMMDD) and placed in a subdirectory called ``masters`` within
the  ``KPFPIPE_DATA`` directory (e.g. ``$KPFPIPE_DATA/masters/20230527/``).

Once you have the master calibration files in place you'll need to edit the following variables
in ``configs/kpf_drp_local.cfg``::

    masterbias # path to a master bias L0 file (e.g. ``$KPFPIPE_DATA/masters/20230527/kpf_20230527_master_bias.fits``)
    masterdark # path to a master dark L0 file (e.g. ``$KPFPIPE_DATA/masters/20230527/kpf_20230527_master_dark.fits``)
    masterflat # path to a master flat L0 file used for the flat field correction (e.g. ``$KPFPIPE_DATA/masters/20230527/kpf_20230527_master_flat.fits``)
    flat_file  # path to a master flat L0 file used to find the order locations (e.g. ``$KPFPIPE_DATA/masters/20230527/kpf_20230527_master_flat.fits``)
    wls_file   # path to a wavelength solution L1 file (e.g. ``$KPFPIPE_DATA/masters/20230527/kpf_20230527_master_WLS_cal-LFC-eve_L1.fits``)

Now download the data from a given night and place it under ``$KPFPIPE_DATA/L0/`` (e.g. ``$KPFPIPE_DATA/L0/20230527/``).

For single-threaded processing you can launch the pipeline with::

    kpf --date 20230527 -r recipes/kpf_drp.recipe -c configs/kpf_drp_local.cfg

or for parallel processing use a combination of the ``--watch``, ``--ncpus``, and ``--reprocess`` flags::

    kpf --reprocess --watch /data/L0/20230527/ --ncpus=8 -r recipes/kpf_drp.recipe -c configs/kpf_drp_local.cfg

In this mode the pipeline will continue to monitor the input directory for new incoming
files forever so you'll need to exit with Ctrl-C once you see that there are no more messages being reported.

In both single-threaded and parallel modes the outputs will be saved under the appropriate date directores in ``$KPFPIPE_DATA/2D/``,
``$KPFPIPE_DATA/L1/``, and ``$KPFPIPE_DATA/L2/``.

