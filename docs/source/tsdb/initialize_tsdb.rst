Initializing the Time-Series PostgreSQL Database
####################################################

Introduction
************************************
Below describes how to create, for the first time, the time-series PostgreSQL database (TSDB).

The TSDB is not in the same cluster as the KPF Operations Database, and runs under a different port number.

Assume PostgreSQL software is built with the termcap and readline libraries, installed, and in the
terminal-window environment:

.. code-block::

    $ echo $PATH
    /data/user/rlaher/db/pg15.2/bin
    $ echo $LD_LIBRARY_PATH
    /data/user/rlaher/db/pg15.2/lib:/data/user/rlaher/db/termcap/build/lib:/data/user/rlaher/db/readline/build/lib

Although the above example uses PostgreSQL version 15.2, the latest PostgreSQL version may be used.

The Linux account that performs the steps below "owns" the Postgres database (USER environment variable).


Initialize and Start PostgreSQL Server
****************************************

Make separate directories for the database data and logs:

.. code-block::

    $ mkdir -p /data/user/rlaher/timeseriesdb/dbdata
    $ mkdir -p /data/user/rlaher/timeseriesdb/dblogs

    $ cd /data/user/rlaher/timeseriesdb

Set up for database initialization:

.. code-block::

    $ touch .pwfile.txt
    $ chmod 600 .pwfile.txt
    $ vi .pwfile.txt
    (insert the following line)
    My#Complex!Password&1234

    $ export TIMESERIESDB=/data/user/rlaher/timeseriesdb
    $ export PGPASSFILE=/data/user/rlaher/timeseriesdb/.pwfile.txt
    $ export PGPASSWORD=My\#Complex\!Password\&1234

    $ printenv TIMESERIESDB
    $ printenv PGPASSFILE
    $ printenv PGPASSWORD
    $ printenv USER

    $ which initdb
    /data/user/rlaher/db/pg15.2/bin/initdb

    $ initdb --pwfile=$PGPASSFILE -D $TIMESERIESDB/dbdata -A md5 -U $USER >& initdb.out

    $ echo $?
    0

Start up the database server:

.. code-block::


    $ pg_ctl -D $TIMESERIESDB/dbdata -l $TIMESERIESDB/dblogs/log -o "-p 6127" start
    waiting for server to start.... done
    server started

    $ ps -ef | grep postgres | grep 6127 | grep -v grep
    rlaher    524244       1  0 Jul30 ?        02:17:19 /data/user/rlaher/db/pg15.2/bin/postgres -D /data/user/rlaher/timeseriesdb/dbdata -p 6127

Here is one way to stop the database server if a restart is needed to boot up a new configuration.

.. code-block::

    $ pg_ctl -D $TIMESERIESDB/dbdata -l $TIMESERIESDB/dblogs/log -o "-p 6127" -m smart stop
    waiting for server to shut down.... done
    server stopped

Create the time-series database (call it `timeseriesopsdb`) with createdb command (and enter password):

.. code-block::

    $createdb -h localhost -p 6127 -U rlaher -W timeseriesopsdb
    My#Complex!Password&1234

    $ psql -p 6127 -d timeseriesopsdb
    timeseriesopsdb=> \list
                                                     List of databases
          Name       | Owner  | Encoding |   Collate   |    Ctype    | ICU Locale | Locale Provider | Access privileges
    -----------------+--------+----------+-------------+-------------+------------+-----------------+-------------------
     postgres        | rlaher | UTF8     | en_US.UTF-8 | en_US.UTF-8 |            | libc            |
     template0       | rlaher | UTF8     | en_US.UTF-8 | en_US.UTF-8 |            | libc            | =c/rlaher        +
                     |        |          |             |             |            |                 | rlaher=CTc/rlaher
     template1       | rlaher | UTF8     | en_US.UTF-8 | en_US.UTF-8 |            | libc            | =c/rlaher        +
                     |        |          |             |             |            |                 | rlaher=CTc/rlaher
     timeseriesopsdb | rlaher | UTF8     | en_US.UTF-8 | en_US.UTF-8 |            | libc            |
    (4 rows)

For convenience, add entry to .pgpass file in home directory:

.. code-block::

    $ vi /data/user/rlaher/.pgpass
    (insert the following line)
    localhost:6127:timeseriesopsdb:rlaher:My#Complex!Password&1234


Add Group Roles For Inheritance
****************************************

Different levels of privileges to be granted later to each group role (as appropriate for the name of the role):

.. code-block::

    $ psql -p 6127 -d timeseriesopsdb

    timeseriesopsdb=> create role kpfadminrole LOGIN SUPERUSER CREATEDB CREATEROLE;
    timeseriesopsdb=> create role kpfporole;
    timeseriesopsdb=> create role kpfreadrole;

    timeseriesopsdb=> select * from pg_roles where rolname in ('rlaher', 'kpfadminrole', 'kpfporole', 'kpfreadrole');

       rolname    | rolsuper | rolinherit | rolcreaterole | rolcreatedb | rolcanlogin | rolreplication | rolconnlimit | rolpassword | rolvaliduntil | rolbypassrls | rolc
    onfig |  oid
    --------------+----------+------------+---------------+-------------+-------------+----------------+--------------+-------------+---------------+--------------+-----
    ------+-------
     rlaher       | t        | t          | t             | t           | t           | t              |           -1 | ********    |               | t            |
          |    10
     kpfadminrole | t        | t          | t             | t           | t           | f              |           -1 | ********    |               | f            |
          | 16392
     kpfporole    | f        | t          | f             | f           | f           | f              |           -1 | ********    |               | f            |
          | 16393
     kpfreadrole  | f        | t          | f             | f           | f           | f              |           -1 | ********    |               | f            |
          | 16394
    (4 rows)

