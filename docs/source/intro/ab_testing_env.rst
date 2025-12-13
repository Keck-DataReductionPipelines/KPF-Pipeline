Creating A/B Test Databases
===========================

The pipeline ships with a helper script, ``scripts/clone_abtest_dbs.py``,
that duplicates both of the Postgres databases used by the DRP so you can test
code changes inside Docker without touching the production DBs. The clones live
in a dockerized Postgres host and receive names that include the current git
hash, e.g. ``kpfopsdb_abtest_a1b2c3``.

Prerequisites
-------------

* ``psql`` and ``pg_dump`` must be installed on the host that will run the
  script (outside Docker).
* The usual database environment variables need to be set in your shell or in
  a startup file that you can pass to the script:

  * primary DB: ``KPFPIPE_DB_SERVER``, ``KPFPIPE_DB_PORT``, ``KPFPIPE_DB_NAME``,
    ``KPFPIPE_DB_USER``, ``KPFPIPE_DB_PASS`` (``DB*`` fallbacks are also read)
  * time-series DB: ``KPFPIPE_TSDB_SERVER``, ``KPFPIPE_TSDB_PORT``,
    ``KPFPIPE_TSDB_NAME``, ``KPFPIPE_TSDB_USER``, ``KPFPIPE_TSDB_PASS`` (and
    the ``TSDB*`` variants)
* The helper automatically reads ``.env`` in the repo root (or the file
  indicated via ``KPFPIPE_ENV_FILE``) and consumes any ``KEY=VALUE`` lines it
  finds there, so placing the credentials in that file is sufficient if you do
  not want to export them manually every time.

Postgres Container Helper
-------------------------------------------------------

The test copies of the DBs live in a local Postgres container that you can
start and stop with the following command:

.. code-block:: bash

   python scripts/abtest_pg.py start

The helper launches ``postgres:15`` with persistent storage in
``./abtest-pgdata``.

.. code-block:: bash

   # Source (production) credentials stay in the usual KPFPIPE_* / DB* vars.
   # Destination overrides tell the cloning script where to create the copies.
   export ABTEST_DB_SERVER=127.0.0.1  # override via ABTEST_HOST_OVERRIDE if desired
   export ABTEST_DB_PORT=7125          # override if ABTEST_PG_HOST_PORT changes
   export ABTEST_DB_USER=kpfopsuser
   export ABTEST_DB_PASS=kpfopspass
   export ABTEST_TSDB_SERVER=127.0.0.1
   export ABTEST_TSDB_PORT=7125
   export ABTEST_TSDB_USER=timeseriesopsuser
   export ABTEST_TSDB_PASS=timeseriesopspass

When finished, tear the helper down with ``python scripts/abtest_pg.py stop``.
The compose helper purposely loads its own environment file,
``docker/abtest-postgres/compose.env``, so the presence of a repo-level
``.env`` (with shell syntax, etc.) will not break Docker Compose. Edit that
file if you need to override the helper's defaults.

The helper forwards Postgres port ``5432`` to host port ``7125`` by default as specified in the compose.env file.

The cloning script automatically reads ``docker/abtest-postgres/compose.env``
and uses any ``ABTEST_*`` overrides it finds there (without overwriting values
already present in your shell). You can still supply additional ``--env-file``
arguments if you want to load other KEY=VALUE files.
Hostnames that resolve only inside Compose (e.g., ``abtest-postgres``) are
automatically swapped to ``127.0.0.1`` by default so the clone script and
``make docker`` can reach the helper from the host. Set
``ABTEST_HOST_OVERRIDE`` if you prefer a different hostname alias.

Cloning workflow
----------------

1. From the repo root, run the script outside Docker:

   .. code-block:: bash

      python scripts/clone_abtest_dbs.py \
          [--source-env-file /path/to/superuser.env] \
          [--env-file /path/to/startup.sh] \
          [--force] \
          [--tag custom_hash]

   * ``--env-file`` lets you feed KEY=VALUE files if the variables are not in
     the current shell.
   * ``--source-env-file`` loads KEY=VALUE files *before* the regular env-file
     list, which is handy for injecting privileged credentials (e.g., a superuser
     for the source database when dumping sequences).
   * ``--show-privileges`` prints the table/sequence privileges for the source
     users and exits (helpful when diagnosing permission failures).
   * ``--force`` drops any existing clone for the resolved tag before
     recreating it.
   * ``--tag`` overrides the tag that defaults to ``git rev-parse --short HEAD``.
   * If the clone databases already exist and ``--force`` is omitted, the script
     will skip recreation and still write the dotenv file pointing at the
     existing clones.

2. The script creates both clone databases, then writes a dotenv file named
   ``.env.abtest.<tag>`` (override via ``--out``). Review the log output for
   the specific database names and hosts.

Launching Docker against the clones
-----------------------------------

Source the generated env file before invoking ``make docker`` (or the masters
variant) so the container inherits the overridden DB settings:

.. code-block:: bash

   source .env.abtest.<tag>
   make docker

Every environment variable required by the pipeline is present in the dotenv
file, so you can use it in other shells as needed. The generated file exports
only the values needed for the cloned DBs (``KPFPIPE_DB_*`` and
``KPFPIPE_TSDB_*``) and also sets ``KPFPIPE_ENV_FILE`` to its own path so that
``docker-run.sh`` picks it up automatically.

The ``make docker`` helpers automatically parse the repository ``.env`` (or the
path supplied via ``KPFPIPE_ENV_FILE``) and export any ``KEY=VALUE`` lines before
starting the container. Non-assignment lines are ignored, so you can keep the
KPFPIPE/DB/TSDB variables there without manually exporting them for every shell.

If you would like the helper and DRP shell managed together, drop into the DRP
container via:

.. code-block:: bash

   python scripts/abtest_pg.py start     # if not already running
   python scripts/abtest_pg.py shell     # interactive shell (similar to make docker)

The compose-managed DRP container uses host networking and the same volume mounts
as ``make docker``, so the database host/port values work identically. Note that
``make docker`` builds the ``kpf-drp:latest`` image first (if needed), while
``abtest_pg.py shell`` assumes the image already exists. Also, ``make docker``
conditionally maps a port if ``KPFPIPE_PORT`` is set, while the compose shell
uses host networking exclusively.

Exit the shell and run ``python scripts/abtest_pg.py stop`` when finished.

Cleanup
-------

When the test databases are no longer needed, drop them with the same script:

.. code-block:: bash

   python scripts/clone_abtest_dbs.py --cleanup [--tag <tag>]

The ``--cleanup`` action removes both clone databases associated with the tag
and deletes the matching ``.env.abtest.<tag>`` file if it exists. The default
tag again comes from the current git commit if ``--tag`` is omitted.

