#! /bin/bash -l

source /home/kpfdrprun/.bash_profile

# Exit if $KPFCRONJOB_LOGS is not defined
if [[ -z "$KPFCRONJOB_LOGS" ]]; then
    echo "Error: KPFCRONJOB_LOGS is not defined."
    exit 1
fi

procdate=$(date +\%Y\%m\%d)
mastersonly=0
wls=0

# Parse the command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --date) procdate="$2"; shift ;; # date to process
        --mastersonly) mastersonly=1 ;; # run only the masters
        --wls) wls=1 ;;
        --timer) timer="$2"; shift ;;  # add exit timer
    esac
    shift
done


# Output the processed date
echo "Processing date: $procdate"

printenv >& $KPFCRONJOB_LOGS/jobs/createMasters_${procdate}.env

# If WLS only is defined on command line (--wls)
if [[ "$wls" -eq 0 ]]; then
    echo "Creating the Level 0 Masters, running $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_lev0.py"
    python $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_lev0.py --date $procdate >& $KPFCRONJOB_LOGS/jobs/kpf_${procdate}_mastersruncmd_lev0.out
    echo Return value from keck_kpfcron_lev0.py = $?

    echo "Creating the Level 1 Masters, running $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_lev1.py"
    python $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_lev1.py --date $procdate >& $KPFCRONJOB_LOGS/jobs/kpf_${procdate}_mastersruncmd_lev1.out
    echo Return value from keck_kpfcron_lev1.py = $?
fi

# create the WLS masters
echo "Creating WLS Masters,  running $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_wls_auto.py"
python $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_wls.py --date $procdate >& $KPFCRONJOB_LOGS/jobs/kpf_${procdate}_masters_wls_auto.out
echo Return value from keck_kpfcron_wls.py = $?

# register the masters in the database
echo "Registering new Masters in database,  running $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_register_in_db.py"
python $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_register.py --date $procdate >& $KPFCRONJOB_LOGS/jobs/kpf_${procdate}_masters_register_in_db.out
echo Return value from keck_kpfcron_register.py = $?

# run the pipeline to process all data for one night
if [[ "$mastersonly" -eq 0 ]]; then
    echo "Running the KPF pipeline on a full night of date: $KPFCRONJOB_CODE/cronjobs/keck_kpfpipe_nightly.py"
    python $KPFCRONJOB_CODE/cronjobs/keck_kpfcron_nightly.py --date $procdate >& $KPFCRONJOB_LOGS/jobs/kpf_${procdate}_nightly_in_db.out
    echo Return value from keck_kpfpipe_nightly.py = $?
fi
