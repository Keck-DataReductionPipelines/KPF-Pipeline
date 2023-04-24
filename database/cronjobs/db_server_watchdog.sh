#! /bin/bash -l

now=$(date)

ps -ef | grep rlaher/db/pg15.2/bin/postgres | grep -v grep

if [[ "$?" != 0 ]]; then

    echo "[$now] Database server is not running; restarting it now..."

    pg_ctl -D $KPFDB/dbdata -l $KPFDB/dblogs/log start

else
    echo "[$now] Database server is running."
fi
