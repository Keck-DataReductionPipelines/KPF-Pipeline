#! /bin/bash -l

procdate=$(date +\%Y\%m\%d)

printenv >& $KPFCRONJOB_LOGS/runDailyPipelines_$procdate.env

echo "Processing date: $procdate"
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l0.pl $procdate >& $KPFCRONJOB_LOGS/kpfmastersruncmd_l0_$procdate.out
echo Return value from kpfmastersruncmd_l0.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l1.pl $procdate >& $KPFCRONJOB_LOGS/kpfmastersruncmd_l1_$procdate.out
echo Return value from kpfmastersruncmd_l1.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmasters_wls_auto.pl $procdate >& $KPFCRONJOB_LOGS/kpfmasters_wls_auto_$procdate.out
echo Return value from kpfmasters_wls_auto.pl = $?
$KPFCRONJOB_CODE/database/cronjobs/kpfmasters_register_in_db.pl $procdate >& $KPFCRONJOB_LOGS/kpfmasters_register_in_db_$procdate.out
echo Return value from kpfmasters_register_in_db.pl = $?
