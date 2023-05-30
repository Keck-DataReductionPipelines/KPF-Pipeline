#! /bin/bash -l

printenv >& $KPFCRONJOB_LOGS/runDailyPipelines.env

echo "Processing date: $(date +\%Y\%m\%d)"
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l0.pl $(date +\%Y\%m\%d) >& $KPFCRONJOB_LOGS/kpfmastersruncmd_l0_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l0.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmastersruncmd_l1.pl $(date +\%Y\%m\%d) >& $KPFCRONJOB_LOGS/kpfmastersruncmd_l1_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l1.pl = $?
$KPFCRONJOB_CODE/cronjobs/kpfmasters_wls_auto.pl $(date +\%Y\%m\%d) >& $KPFCRONJOB_LOGS/kpfmasters_wls_auto_$(date +\%Y\%m\%d).out
echo Return value from kpfmasters_wls_auto.pl = $?
$KPFCRONJOB_CODE/database/cronjobs/kpfmasters_register_in_db.pl $(date +\%Y\%m\%d) >& $KPFCRONJOB_LOGS/kpfmasters_register_in_db_$(date +\%Y\%m\%d).out
echo Return value from kpfmasters_register_in_db.pl = $?
