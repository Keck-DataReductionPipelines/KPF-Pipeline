#! /bin/bash -l

now=$(date)

ps -ef | grep rlaher/db/pg15.2/bin/postgres | grep /data/user/rlaher/kpfdb | grep -v grep

if [[ "$?" != 0 ]]; then

    echo "[$now] Database server for pipeline-operations database is not running; restarting it now..."

    pg_ctl -D $KPFDB/dbdata -l $KPFDB/dblogs/log start

else
    echo "[$now] Database server for pipeline-operations database is running."
fi

ps -ef | grep rlaher/db/pg15.2/bin/postgres | grep /data/user/rlaher/timeseriesdb | grep -v grep

if [[ "$?" != 0 ]]; then

    echo "[$now] Database server for time-series database is not running; restarting it now..."

    pg_ctl -D $TIMESERIESDB/dbdata -l $TIMESERIESDB/dblogs/log -o "-p 6127" start

else
    echo "[$now] Database server for time-series database is running."
fi
