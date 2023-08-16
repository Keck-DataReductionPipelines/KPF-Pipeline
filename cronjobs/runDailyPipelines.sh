#! /bin/bash -l

printenv >& $KPFCRONJOB_LOGS/jobs/runDailyPipelines.env

procdate=$(date +\%Y\%m\%d)

echo "Processing date: $procdate"
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l0.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmastersruncmd_l0_$procdate.out
echo Return value from kpfmastersruncmd_l0.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l1.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmastersruncmd_l1_$procdate.out
echo Return value from kpfmastersruncmd_l1.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmasters_wls_auto.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmasters_wls_auto_$procdate.out
echo Return value from kpfmasters_wls_auto.pl = $?
$KPFCRONJOB_CODE/database/cronjobs/kpfmasters_register_in_db.pl $procdate >& $KPFCRONJOB_LOGS/jobs/kpfmasters_register_in_db_$procdate.out
echo Return value from kpfmasters_register_in_db.pl = $?
