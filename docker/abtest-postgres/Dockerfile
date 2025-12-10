FROM postgres:15

# Install q3c extension so restores that depend on CREATE EXTENSION q3c succeed.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends postgresql-15-q3c \
 && rm -rf /var/lib/apt/lists/*

# Default Postgres image entrypoint handles initialization and running.

