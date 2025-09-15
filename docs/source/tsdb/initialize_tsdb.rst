Initializing Time-Series PostgreSQL Database
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

The Linux account that performs the steps below "owns" the Postgres database as administrator (USER environment variable).


Initialize And Start PostgreSQL Server
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
The group roles to be added are ``kpfadminrole``, ``kpfreadrole``, and ``kpfporole``.
The latter is group role for routine pipeline operations.

.. code-block::

    $ psql -p 6127 -d timeseriesopsdb

    timeseriesopsdb=> create role kpfadminrole LOGIN SUPERUSER CREATEDB CREATEROLE;
    timeseriesopsdb=> create role kpfreadrole;
    timeseriesopsdb=> create role kpfporole;

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


Add User Roles
****************************************

The user roles inherit the group roles.
The user roles to be added are ``timeseriesdba``, ``timeseriesreadonlyuser``, and ``timeseriesopsuser``.
The latter is user role for routine pipeline operations.
Enter a password for the user role (twice), then a password for the database administrator.
The appropriate user-role passwords can be added to the .pgpass file in home directories of the assigned TSDB users.

.. code-block::

    $ createuser -h localhost -p 6127 --connection-limit=10 --echo --role=kpfadminrole --pwprompt timeseriesdba -W
    Enter password for new role:
    Enter it again:
    Password:
    SELECT pg_catalog.set_config('search_path', '', false);
    CREATE ROLE timeseriesdba PASSWORD 'md5328138f2dfe618c37b96de28ecb91649' NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT LOGIN CONNECTION LIMIT 10 IN ROLE kpfadminrole;

    $ echo $?
    0

    $ psql -d timeseriesopsdb -h localhost -p 6127 -U timeseriesdba
    timeseriesopsdb=#


    $ createuser -h localhost -p 6127 --connection-limit=100 --echo --role=kpfreadrole --pwprompt timeseriesreadonlyuser -W
    Enter password for new role:
    Enter it again:
    Password:
    SELECT pg_catalog.set_config('search_path', '', false);
    CREATE ROLE timeseriesreadonlyuser PASSWORD 'md5a477f1f215d6ab77e67d1228034a7529' NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT LOGIN CONNECTION LIMIT 100 IN ROLE kpfreadrole;

    $ echo $?
    0

    $ psql -d timeseriesopsdb -h localhost -p 6127 -U timeseriesreadonlyuser
    timeseriesopsdb=>


    $ createuser -h localhost -p 6127 --connection-limit=100 --echo --role=kpfporole --pwprompt timeseriesopsuser -W
    Enter password for new role:
    Enter it again:
    Password:
    SELECT pg_catalog.set_config('search_path', '', false);
    CREATE ROLE timeseriesopsuser PASSWORD 'md54cf91a5a9fa1c099fb7231024d23dc88' NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT LOGIN CONNECTION LIMIT 100 IN ROLE kpfporole;

    $ echo $?
    0

    psql -d timeseriesopsdb -h localhost -p 6127 -U timeseriesopsuser
    timeseriesopsdb=>



Apply Superuser Grants To Group Admin Role
********************************************

.. code-block::

    $ psql -p 6127 -d timeseriesopsdb

    timeseriesopsdb=# GRANT ALL PRIVILEGES ON SCHEMA public TO kpfadminrole;
    GRANT
    timeseriesopsdb=# GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kpfadminrole;
    GRANT
    timeseriesopsdb=# GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kpfadminrole;
    GRANT



Changes To Default Database Configuration
********************************************

Below are additional configuration changes that have been implemented:

.. code-block::

    $ vi postgresql.conf

    listen_addresses = '*'
    shared_buffers = 128GB
    max_connections=300
    logging_collector = on
    log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
    log_file_mode = 0640
    log_rotation_age = 7d


The database server must be stopped and restarted for the changes to take effect:

.. code-block::

    $ pg_ctl -D $TIMESERIESDB/dbdata -l $TIMESERIESDB/dblogs/log -o "-p 6127" -m smart stop
    waiting for server to shut down.... done
    server stopped

    $ pg_ctl -D $TIMESERIESDB/dbdata -l $TIMESERIESDB/dblogs/log -o "-p 6127" start
    waiting for server to start.... done
    server started
