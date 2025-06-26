#! /bin/bash -l

printenv >& $KPFCRONJOB_LOGS/jobs/runPerlScriptsUnderCrontab.env

procdate=$(date +\%Y\%m\%d)

echo "Processing date: $procdate"

$KPFCRONJOB_CODE/cronjobs/updateSoftwareAndReferenceFits.pl >& $KPFCRONJOB_LOGS/jobs/updateSoftwareAndReferenceFits_$procdate.out
echo Return value from updateSoftwareAndReferenceFits.pl = $?

$KPFCRONJOB_CODE/cronjobs/cleanOldFilesFromDisk.pl >& $KPFCRONJOB_LOGS/jobs/cleanOldFilesFromDisk_$procdate.out
echo Return value from cleanOldFilesFromDisk.pl = $?
