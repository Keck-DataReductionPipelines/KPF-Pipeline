#! /bin/bash -l

procdate=$(date +\%Y\%m\%d)

printenv >& $KPFCRONJOB_LOGS/jobs/runDailyPipelines_$procdate.env

echo "Processing date: $procdate"

# If the case-sensitive string "wls" is given on the command line, then execute
# only the commands to produce the WLS products and register them in the database.

if [[ $1 != "wls" ]] ; then
    $KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l0.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmastersruncmd_l0_$procdate.out
    echo Return value from kpfmastersruncmd_l0.pl = $?
    $KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l1.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmastersruncmd_l1_$procdate.out
    echo Return value from kpfmastersruncmd_l1.pl = $?
fi

$KPFCRONJOB_CODE/cronjobs/kpfmasters_wls_auto.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmasters_wls_auto_$procdate.out
echo Return value from kpfmasters_wls_auto.pl = $?

$KPFCRONJOB_CODE/cronjobs/kpfmasters_etalon_analysis.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmasters_etalon_analysis_$procdate.out
echo Return value from kpfmasters_etalon_analysis.pl = $?

$KPFCRONJOB_CODE/database/cronjobs/kpfmasters_register_in_db.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmasters_register_in_db_$procdate.out
echo Return value from kpfmasters_register_in_db.pl = $?
