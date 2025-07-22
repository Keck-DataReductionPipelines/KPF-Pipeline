#! /bin/bash -l

printenv >& $KPFCRONJOB_LOGS/jobs/runPerlScriptsUnderCrontab.env

procdate=$(date +\%Y\%m\%d)

echo "Processing date: $procdate"

$KPFCRONJOB_CODE/cronjobs/updateSoftwareAndReferenceFits.pl >& $KPFCRONJOB_LOGS/jobs/updateSoftwareAndReferenceFits_$procdate.out
echo Return value from updateSoftwareAndReferenceFits.pl = $?

$KPFCRONJOB_CODE/cronjobs/runCleanOldFilesFromDiskInContainer.pl $procdate >& $KPFCRONJOB_LOGS/jobs/runCleanOldFilesFromDiskInContainer_$procdate.out
echo Return value from runCleanOldFilesFromDiskInContainer.pl = $?
