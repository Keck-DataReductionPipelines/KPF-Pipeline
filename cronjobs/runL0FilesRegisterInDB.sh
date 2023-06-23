#! /bin/bash -l

printenv >& $KPFCRONJOB_LOGS/runL0FilesRegisterInDB.env

procdate=$(date +\%Y\%m\%d)

echo "Processing date: $procdate"
$KPFCRONJOB_CODE/database/cronjobs/l0files_register_in_db.pl $procdate >& $KPFCRONJOB_LOGS/l0files_register_in_db_$procdate.out
echo Return value from l0files_register_in_db.pl = $?
