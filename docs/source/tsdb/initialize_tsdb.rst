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

