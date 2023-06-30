
Pipelines for Master Calibration Files
================================================================

    There are several pipelines that generate master calibration files, including for bias, dark, flat, and arclamp exposures.
    Generally, images contained in L0 raw FITS files with the same value of some FITS keyword, like OBJECT and/or IMTYPE, are stacked, and
    then there is possibly some form of normalization, in order to create a master calibration file.
    Within a FITS file, the image data within FITS extensions GREEN_CCD, RED_CCD, and CA_HK are separately stacked.
    The uncertainty images associated with the stacking are given in the product with separate extensions
    for the different detectors, in which the extension names have suffix '_UNC'.
    Images for the number of stack inputs at each pixel location have suffix '_CNT'.
    The number of inputs and the minimum and maximum modified Julian date of the inputs are tracked in the
    PRIMARY FITS header via the NFRAMES, MINMJD, and MAXMJD keywords, respectively.


Master Bias
------------
    
    For a master bias, only exposures with FITS-keyword EXPTIME = 0.0 are considered as inputs.
    The images are stacked, outliers beyond sigma = +/-2.1 are rejected, and then an average is computed.
    The IMTYPE keyword is set to 'Bias' in the PRIMARY FITS header of this product.
    If 1% of the pixels have less than 10 samples in the stack, then a flag in the keyword INFOBITS is set in the PRIMARY FITS header;
    for GREEN_CCD, RED_CCD, and CA_HK, the flags are bits 0, 1, and 2, respectively.


Master Dark
-------------
    
    For a master dark, exposures with FITS-keywords IMTYPE = 'Dark'  and EXPTIME >= 300.0 are considered as inputs.
    The master bias is subtracted from the input image data, and the results are normalized by the exposure time.
    The images are stacked, outliers beyond sigma = +/-2.2 are rejected, and then an average is computed.
    The IMTYPE keyword is set to 'Dark' in the PRIMARY FITS header of this product.
    If 1% of the pixels have less than 10 samples in the stack, then a flag in the keyword INFOBITS is set in the PRIMARY FITS header;
    for GREEN_CCD, RED_CCD, and CA_HK, the flags are bits 0, 1, and 2, respectively.


Master Flat
------------
    
    For a master flat, exposures with FITS-keywords IMTYPE = 'Flatlamp'  and EXPTIME <= 60.0 are considered as inputs.
    The master bias is subtracted from the input image data, and the results are normalized by the exposure time.
    The master dark is then subtracted.
    The images are stacked, outliers beyond sigma = +/-2.3 are rejected, and then an average is computed.
    The resulting stacked image is normalized by a smooth counterpart (after applying a 2-D Gaussian blur with sigma = 2.0).
    The flat normalization is done separately for each orderlet, where orderlet traces are defined by the mask file::

    /KPF-Pipeline-TestData/order_mask_3_2_20230502.fits

    The mode statistic within each orderet is the normalization factor.
    Pixel values in the master flat outside of the traces are set to unity.
    Also, pixel values in the master flat with values less than 50 DN in the unnormalized image before smoothing are set to unity.
    The IMTYPE keyword is set to 'Flat' in the PRIMARY FITS header of this product.
    If 1% of the pixels have less than 10 samples in the stack, then a flag in the keyword INFOBITS is set in the PRIMARY FITS header;
    for GREEN_CCD, RED_CCD, and CA_HK, the flags are bits 0, 1, and 2, respectively.


Master Arclamp
--------------
    
    For a master arclamp, exposures with FITS-keywords IMTYPE = 'Arclamp'  and the same OBJECT setting are considered as inputs.
    For a given run of this pipeline, the desired OBJECT for the image stacking, such as 'autocal-une-sky', is given.
    Only GREEN_CCD and RED_CCD FITS extensions are handled at this time.
    The master bias is subtracted from the input image data, and the results are normalized by the exposure time.
    The master dark is then subtracted.
    The master flat is applied (via image division).
    The images are stacked, outliers beyond sigma = +/-2.4 are rejected, and then an average is computed.
    The IMTYPE keyword is set to 'Arclamp' in the PRIMARY FITS header of this product.
    If 1% of the pixels have less than 5 samples in the stack, then a flag in the keyword INFOBITS is set in the PRIMARY FITS header;
    for GREEN_CCD, RED_CCD, and CA_HK, the flags are bits 0, 1, and 2, respectively.


Daily Operations
----------------------

    The pipelines for master calibration files are executed daily in a cronjob at 3 p.m. PT::

    /KPF-Pipeline/cronjobs/runDailyPipelines.sh

    This bash script executes four different Perl scripts in succession::
    
    /KPF-Pipeline/cronjobs/kpfmastersruncmd_l0.pl
    /KPF-Pipeline/cronjobs/kpfmastersruncmd_l1.pl
    /KPF-Pipeline/cronjobs/kpfmasters_wls_auto.pl
    /KPF-Pipeline/database/cronjobs/kpfmasters_register_in_db.pl

    Each of the Perl scripts runs a docker container in the detached mode, which, in turn, executes the relevant recipe.
    Here are the four steps:

    1. Create 2D-image FITS files and then execute the bias, dark, flat, and arclamp master pipelines, as described above,
       for all data taken nightly.  A master bias, dark, and flat are generated, as well as a number of master arclamps for different OBJECTS.
    2. Generate the 1D spectral master products.
    3. Generate the master wavelength solution (WLS).
    4. Load records into the CalFIles table of the pipeline operations database for all master files.
  
    Note that the database port and password are obtained from the .pgpass file in the user's home directory.
    A number of environment variables must be set for these scripts to provide other required parameters:

        KPFPIPE_TEST_DATA
            Legacy KPF directory for outputs of testing.  E.g., /KPF-Pipeline-TestData

        KPFPIPE_MASTERS_BASE_DIR
            Base directory of master files for permanent storage.  E.g., /data/kpf/masters

        KPFCRONJOB_SBX
            Sandbox directory for intermediate files.  E.g., /data/user/rlaher/sbx
 
        KPFCRONJOB_CODE
            Code directory of KPF-Pipeline git repo where the docker run command is executed.  E.g., /data/user/rlaher/git/KPF-Pipeline

        KPFCRONJOB_LOGS
            Logs directory where log file (STDOUT) from this script goes (see runDailyPipelines.sh).
            Normally this is the code directory of KPF-Pipeline git repo.
            E.g., /data/user/rlaher/git/KPF-Pipeline

        KPFCRONJOB_DOCKER_NAME_L0
            Prefix of docker container name for this Perl script, a unique name so it can be monitored by docker ps command.
            E.g., russkpfmastersdrpl0

        KPFCRONJOB_DOCKER_NAME_L1
            Prefix of docker container name for this Perl script, a unique name so it can be monitored by docker ps command.
            E.g., russkpfmastersdrpl1

        KPFCRONJOB_DOCKER_NAME_WLS
            Prefix of docker container name for this Perl script, a unique name so it can be monitored by docker ps command.
            E.g., russkpfmasterswlsauto

        KPFCRONJOB_DOCKER_NAME_DBSCRIPT
            Prefix of docker container name for this Perl script, a unique name so it can be monitored by docker ps command.
            E.g., russkpfmastersregisterindb

        KPFDBUSER
            Name of database user with privileges for pipeline operations (i.e., with GRANT kpfporole).
            E.g., apollo
           
        KPFDBNAME
            Name of pipeline operations database.
            E.g., kpfopsdb


    Here are the master product files generated for June 6, 2023::

    -rw-r--r--. 1 rlaher citah      5440 Jun  6 16:36 kpfmastersruncmd_l0_20230606.out
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_201091.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_201091_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_201091_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-etalon-all-morn.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-etalon-all-morn_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-etalon-all-morn_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-thar-all-morn.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-thar-all-morn_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-thar-all-morn_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-une-all-morn.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-une-all-morn_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_autocal-une-all-morn_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_cal-LFC-morn.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_cal-LFC-morn_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_cal-LFC-morn_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_arclamp_slewcal.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_arclamp_slewcal_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_arclamp_slewcal_L2.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_bias.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_bias_L1.fits
    -rw-r--r--. 1 rlaher citah 532739520 Jun  6 16:56 kpf_20230606_master_dark.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_dark_L1.fits
    -rw-r--r--. 1 rlaher citah 799093440 Jun  6 16:56 kpf_20230606_master_flat.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 16:56 kpf_20230606_master_flat_L1.fits
    -rw-r--r--. 1 rlaher citah   1794240 Jun  6 16:56 kpf_20230606_master_flat_L2.fits
    -rw-r--r--. 1 rlaher citah      5496 Jun  6 16:56 kpfmastersruncmd_l1_20230606.out
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 17:32 kpf_20230606_master_WLS_autocal-thar-all-morn_L1.fits
    -rw-r--r--. 1 rlaher citah  32993280 Jun  6 17:32 kpf_20230606_master_WLS_cal-LFC-morn_L1.fits
    -rw-r--r--. 1 rlaher citah      4250 Jun  6 17:32 kpfmasters_wls_auto_20230606.out
    -rw-r--r--. 1 root   root      82700 Jun  6 17:32 registerCalFilesForDate_20230606.out
    -rw-r--r--. 1 rlaher citah      1576 Jun  6 17:32 kpfmasters_register_in_db_20230606.out
