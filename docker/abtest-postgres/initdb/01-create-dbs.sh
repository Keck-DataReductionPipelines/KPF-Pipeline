#!/bin/bash
set -euo pipefail

create_db() {
    local dbname="$1"
    local dbuser="$2"
    local dbpass="$3"

    psql --username "$POSTGRES_USER" --dbname postgres <<-SQL
        DO
        \$\$
        BEGIN
            IF NOT EXISTS (
                SELECT
                FROM   pg_catalog.pg_roles
                WHERE  rolname = '$dbuser'
            ) THEN
                CREATE ROLE $dbuser LOGIN PASSWORD '$dbpass';
            END IF;
        END
        \$\$;
SQL

    psql --username "$POSTGRES_USER" --dbname postgres <<-SQL
        ALTER ROLE $dbuser WITH CREATEDB SUPERUSER;
SQL

    psql --username "$POSTGRES_USER" --dbname postgres <<-SQL
        DO
        \$\$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_database WHERE datname = '$dbname'
            ) THEN
                CREATE DATABASE $dbname OWNER $dbuser ENCODING 'UTF8';
            END IF;
        END
        \$\$;
SQL
}

create_db "kpfopsdb" "${ABTEST_DB_USER:-kpfopsuser}" "${ABTEST_DB_PASS:-kpfopspass}"
create_db "timeseriesopsdb" "${ABTEST_TSDB_USER:-timeseriesopsuser}" "${ABTEST_TSDB_PASS:-timeseriesopspass}"

