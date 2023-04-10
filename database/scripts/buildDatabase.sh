#! /bin/bash

#
# Script to build PostgreSQL database software (v. 15.2) from source code, start
# database server running, and then create a database (myopsdb) with $USER as superuser.
# The database server will be installed and running under the user account, not root.
# Connection parameters will be added to ~/.pgass file.
# Database files (mydb) will be in separate directory from database software (pg15.2).
# After this script is done, the superuser should "SET ROLE kpfreadrole;"
# in the psql client for non-superuser database querying.
# A non-superuser database user called apollo is created for use by pipeline operations.
#

#####################################################
# Parameters.
#####################################################

KPF_PIPELINE_GIT_REPO=/data/user/rlaher/git/KPF-Pipeline
PGPORT=6135
DB_BUILD_DATABASE_PATH=/data/user/rlaher/test/mydb
DB_BUILD_DBNAME=myopsdb
DB_BUILD_PGPASSWORD=testpassword
DB_USER_APOLLO=apollo
DB_USER_APOLLO_PW=Arrow

DB_TAR_GZ_FILE_URL=https://ftp.postgresql.org/pub/source/v15.2/postgresql-15.2.tar.gz
DB_BUILD_BASE=/data/user/rlaher/test/pg15.2
DB_BUILD_LIBS=/data/user/rlaher/db/termcap/build/lib:/data/user/rlaher/db/readline/build/lib:/usr/lib64
DB_BUILD_INCL=/data/user/rlaher/db/termcap/build/include:/data/user/rlaher/db/readline/build/include:/usr/include


#####################################################
# Modify below this line only if you are an expert!
#####################################################

echo DB_TAR_GZ_FILE_URL = $DB_TAR_GZ_FILE_URL
echo DB_BUILD_BASE = $DB_BUILD_BASE
echo DB_BUILD_LIBS = $DB_BUILD_LIBS
echo DB_BUILD_INCL = $DB_BUILD_INCL

mkdir -p $DB_BUILD_BASE
cd $DB_BUILD_BASE

echo PWD = $PWD

echo Building Postrgres database on $(date +\%Y\%m\%d)

printenv > buildDatabase.env

wget --no-check-certificate $DB_TAR_GZ_FILE_URL

gunzip postgresql-15.2.tar.gz 
tar xvf postgresql-15.2.tar 

mkdir build_dir
cd build_dir

$DB_BUILD_BASE/postgresql-15.2/configure --with-libraries=$DB_BUILD_LIBS  --with-includes=$DB_BUILD_INCL --prefix=$DB_BUILD_BASE

echo Exit code from configure = $?

make
echo Exit code from make = $?

make check
echo Exit code from make check = $?

make install
echo Exit code from make install = $?


export PATH=$DB_BUILD_BASE/bin:$PATH
export LD_LIBRARY_PATH=$DB_BUILD_BASE/lib:$LD_LIBRARY_PATH


echo PATH = $PATH
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PGPORT = $PGPORT

PGPASSFILE=$DB_BUILD_DATABASE_PATH/.pwfile.txt

mkdir -p $DB_BUILD_DATABASE_PATH
cd $DB_BUILD_DATABASE_PATH
mkdir dbdata
mkdir dblogs


echo $DB_BUILD_PGPASSWORD > $PGPASSFILE
chmod 600 $PGPASSFILE

echo "localhost:$PGPORT:*:$USER:$DB_BUILD_PGPASSWORD" >> ~/.pgpass
chmod 600 ~/.pgpass


$DB_BUILD_BASE/bin/initdb --pwfile=$PGPASSFILE -D $DB_BUILD_DATABASE_PATH/dbdata -A md5 -U $USER >& initdb.out

echo Exit code from $DB_BUILD_BASE/bin/initdb --pwfile=$PGPASSFILE -D $DB_BUILD_DATABASE_PATH/dbdata -A md5 -U $USER = $?

$DB_BUILD_BASE/bin/pg_ctl -D $DB_BUILD_DATABASE_PATH/dbdata -l $DB_BUILD_DATABASE_PATH/dblogs/log start

echo Exit code from $DB_BUILD_BASE/bin/pg_ctl -D $DB_BUILD_DATABASE_PATH/dbdata -l $DB_BUILD_DATABASE_PATH/dblogs/log start = $?

$DB_BUILD_BASE/bin/createdb -h localhost -p $PGPORT -U $USER $DB_BUILD_DBNAME

echo Exit code from $DB_BUILD_BASE/bin/createdb -h localhost -p $PGPORT -U $USER $DB_BUILD_DBNAME = $?

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -c "select * from pg_tables;" >& test_query.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -c \"select \* from pg_tables\;\" = $?


#
# CREATE TABLESPACES
#

mkdir -p $DB_BUILD_DATABASE_PATH/tablespacedata1
mkdir -p $DB_BUILD_DATABASE_PATH/tablespaceindx1

echo CREATE TABLESPACE pipeline_data_01 LOCATION \'$DB_BUILD_DATABASE_PATH/tablespacedata1\'\; > tablespaces.sql
echo CREATE TABLESPACE pipeline_indx_01 LOCATION \'$DB_BUILD_DATABASE_PATH/tablespaceindx1\'\; >> tablespaces.sql

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f tablespaces.sql >& tablespaces.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f tablespaces.sql = $?


#
# CREATE ROLES
#

echo CREATE ROLE kpfadminrole LOGIN SUPERUSER CREATEDB CREATEROLE\; > roles.sql
echo CREATE ROLE kpfporole\; >> roles.sql
echo CREATE ROLE kpfreadrole\; >> roles.sql
echo GRANT kpfadminrole to $USER\; >> roles.sql
echo GRANT kpfporole to $USER\; >> roles.sql
echo GRANT kpfreadrole to $USER\; >> roles.sql

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f roles.sql >& roles.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f roles.sql = $?


#
# CREATE USER apollo with psql-client login privileges and unlimited connection for use by pipeline operator, which
# inherits from ROLE kpfporole generally allowed table grants for INSERT,UPDATE,SELECT,REFERENCES (no DELETE or TRUNCATE). 
#

echo CREATE USER $DB_USER_APOLLO CONNECTION LIMIT -1 ENCRYPTED PASSWORD \'$DB_USER_APOLLO_PW\'\; > users.sql
echo GRANT kpfporole to $DB_USER_APOLLO\; >> users.sql

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f users.sql >& users.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f users.sql = $?

echo "localhost:$PGPORT:$DB_BUILD_DBNAME:$DB_USER_APOLLO:$DB_USER_APOLLO_PW" >> ~/.pgpass
chmod 600 ~/.pgpass


#
# CREATE KPF TABLES AND PROCEDURES
#

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsTables.sql >& kpfOpsTables.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsTables.sql = $?

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsTableGrants.sql >& kpfOpsTableGrants.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsTableGrants.sql = $?

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsProcs.sql >& kpfOpsProcs.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsProcs.sql = $?

$DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsProcGrants.sql >& kpfOpsProcGrants.out

echo Exit code from $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER -f $KPF_PIPELINE_GIT_REPO/database/schema/kpfOpsProcGrants.sql = $?


#
# TERMINATE
#

if [[ "$?" == 0 ]]; then

    echo
    echo ##################################################################
    echo "Congratulations! Database server is running and database $DB_BUILD_DBNAME created."
    echo
    echo Stop the server with this command:
    echo $DB_BUILD_BASE/bin/pg_ctl -D $DB_BUILD_DATABASE_PATH/dbdata -l $DB_BUILD_DATABASE_PATH/dblogs/log stop
    echo
    echo Your ~/.pgpass file has been augmented with database connection parameters.
    echo
    echo Put these three lines in your environment to run the psql client:
    echo "export PATH=$DB_BUILD_BASE/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=$DB_BUILD_BASE/lib:\$LD_LIBRARY_PATH"
    echo "export PGPORT=$PGPORT"
    echo
    echo Here is how to connect to your new database:
    echo $DB_BUILD_BASE/bin/psql -h localhost -d $DB_BUILD_DBNAME -p $PGPORT -U $USER
    echo ##################################################################
    echo

fi
