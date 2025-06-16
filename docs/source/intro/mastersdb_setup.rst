
Installing/Configuring the Pipeline Operations Database
=================================================================================

The pipeline operations database runs under a regular unix account,
and not a root account (in order not to involve special access and a
system administrator).
A bash script has been developed that does the initial installation and
configuration of the the pipeline operations database::
  
    KPF-Pipeline/database/scripts/buildDatabase.sh

This script also starts up an instance of the database server, so that
the database is ready for immediate use.
A database superuser will be set up with the
same name as the unix account under which the pipeline
operations database is installed.  Essentially, the owner of the unix
account under which this bash script is executed will be a superuser
of this database.  In the example below, the unix account is "rlaher".


There are parameters at the beginning of the bash script that need to
be reviewed and modified according to the particular
requirements and environment of the project.
The following are the default values coded in the script::

    KPF_PIPELINE_GIT_REPO=/data/user/rlaher/git/KPF-Pipeline
    PGPORT=6125
    DB_BUILD_DATABASE_PATH=/data/user/rlaher/test/mydb
    DB_BUILD_DBNAME=kpfopsdb
    DB_BUILD_PGPASSWORD=testpassword
    DB_USER_APOLLO=apollo
    DB_USER_APOLLO_PW=Arrow
    DB_TAR_GZ_FILE_URL=https://ftp.postgresql.org/pub/source/v15.2/postgresql-15.2.tar.gz
    DB_BUILD_BASE=/data/user/rlaher/test/pg15.2
    DB_BUILD_LIBS=/data/user/rlaher/db/termcap/build/lib:/data/user/rlaher/db/readline/build/lib:/usr/lib64
    DB_BUILD_INCL=/data/user/rlaher/db/termcap/build/include:/data/user/rlaher/db/readline/build/include:/usr/include

Notes on the bash-script parameters:

    KPF_PIPELINE_GIT_REPO
        Pathname to KPF-Pipeline checkout of git repository.

    PGPORT
        Port number of connections from database client applications.
        Do not specify 5432 (which is the default port number, normally
        used by a PostgresSQL server running under root).

    DB_BUILD_DATABASE_PATH
        Pathname to a directory where the low-level database files
        will be stored.  This includes database files for logs,
        configuration, data tables, indexes, and table spaces.
        

    DB_BUILD_DBNAME
        Name of the pipeline operations database.
        
    DB_BUILD_PGPASSWORD
        Superuser password for the pipeline operations database.
    
    DB_USER_APOLLO
        Database user name for non-superuser account (default is
        "apollo"), for use in pipeline operations.  This account has
        psql-client login privileges, is allowed an unlimited number
        of database connections, and inherits from ROLE kpfporole,
        which generally has table grants for INSERT, UPDATE, SELECT,
        REFERENCES (no DELETE or TRUNCATE). 
       
    DB_USER_APOLLO_PW
        Password for the apollo database user.
    
    DB_TAR_GZ_FILE_URL
        URL from where the PostgreSQL-database source code is
        downloaded.
       
    DB_BUILD_BASE
        Pathname to a directory where the database software will be
        built and the bin, lib, and include directories will be
        installed.
        
    DB_BUILD_LIBS
        Locations of the lib directories for the termcap and readline
        libraries, as well as other libraries required for this
        database-software build.
        
    DB_BUILD_INCL
        Locations of include include directories for the termcap and
        readline libraries, as well as other libraries required for this
        database-software build.

The database superuser is encouraged to review the steps in the bash script to gain
an understanding of precisely how the database is set up.

After the database-server instance has been started by the bash
script, several SQL commands are loaded into the pipeline operations
database, in order to set up the required tablespaces, roles, and users::

    CREATE TABLESPACE pipeline_data_01 LOCATION '/data/user/rlaher/test/mydb/tablespacedata1';
    CREATE TABLESPACE pipeline_indx_01 LOCATION '/data/user/rlaher/test/mydb/tablespaceindx1';
    CREATE ROLE kpfadminrole LOGIN SUPERUSER CREATEDB CREATEROLE;
    CREATE ROLE kpfporole;
    CREATE ROLE kpfreadrole;
    GRANT kpfadminrole to rlaher;
    GRANT kpfporole to rlaher;
    GRANT kpfreadrole to rlaher;
    CREATE USER apollo CONNECTION LIMIT -1 ENCRYPTED PASSWORD 'Arrow';
    GRANT kpfporole to apollo;

Then, the following SQL files are loaded into the
pipeline operations database (in the order listed), which contain the definitions
of the database tables and stored procedures required for pipeline operations::
  
    KPF-Pipeline/database/schema/kpfOpsTables.sql
    KPF-Pipeline/database/schema/kpfOpsTableGrants.sql
    KPF-Pipeline/database/schema/kpfOpsProcs.sql
    KPF-Pipeline/database/schema/kpfOpsProcGrants.sql

After the bash script finishes, assuming it ran normally, the user will see the following useful
output at the very end::

    Congratulations! Database server is running and database kpfopsdb created.

    Stop the server with this command:
    /data/user/rlaher/test/pg15.2/bin/pg_ctl -D /data/user/rlaher/test/mydb/dbdata -l /data/user/rlaher/test/mydb/dblogs/log stop

    Your /data/user/rlaher/.pgpass file has been augmented with database connection parameters.

    Put these three lines in your environment to run the psql client:
    export PATH=/data/user/rlaher/test/pg15.2/bin:$PATH
    export LD_LIBRARY_PATH=/data/user/rlaher/test/pg15.2/lib:$LD_LIBRARY_PATH
    export PGPORT=6125

    Here is how to connect to your new database:
    /data/user/rlaher/test/pg15.2/bin/psql -h localhost -d kpfopsdb -p 6125 -U rlaher

There are currently three database tables in the pipeline operations database:  CalFiles, L0Files, and
L0infobits.  The CalFiles database table stores a record for each master calibration file (many records
are inserted daily for biases, darks, flats, arclamps, etc.).  The
L0Files database table stores a record for each exposure.
The L0infobits database table stores definitions of the bit-wise information flags
corresponding to the infobits column in the L0Files table (the
CalFiles infobits have not yet been formalized into a database table,
and are described on :doc:`../api/pipeline/masters`).
These database tables are queried by the pipeline for relevant nearest-in-time
master calibration files, and generally for quality-control purposes.
Here are the table-column definitions::

  
    -----------------------------
    -- TABLE: CalFiles
    -----------------------------
  
    CREATE TABLE calfiles (
        cid integer NOT NULL,                         -- Primary key
        level smallint NOT NULL,                      -- Product level (L0, L1, or L2)
        caltype character varying(32) NOT NULL,       -- FITS-header keyword: IMTYPE in extension 4-6 (lowercase)
        "object" character varying(32) NOT NULL,      -- FITS-header keyword: TARGOBJ or OBJECT (lowercase)
        contentbits integer NOT NULL,                 -- BIT-WISE FLAGS FOR INCLUDING CCDs: BIT0: GREEN; BIT1: RED; BIT2: CA_HK
        nframes smallint,                             -- FITS-header keyword: NFRAMES (GREEN CCD)
        minmjd double precision,                      -- FITS-header keyword: MINMJD (GREEN CCD)
        maxmjd double precision,                      -- FITS-header keyword: MAXMJD (GREEN CCD)
        infobits integer,                             -- FITS-header keyword: INFOBITS
        startdate date NOT NULL,                      -- Start date for application of master (for earliest-in-time selection)
        enddate date NOT NULL,                        -- End date for application of master (may not be used)
        filename character varying(255) NOT NULL,     -- Path and filename of master calibration file.
        checksum character varying(32) NOT NULL,      -- MD5 checksum
        status smallint DEFAULT 0 NOT NULL,           -- Set to zero if bad and one if good
        createdby character varying(30) NOT NULL,     -- Script that inserted the record
        created timestamp without time zone NOT NULL, -- FITS-header keyword: CREATED (GREEN CCD) in Zulu time
        "comment" character varying(255)              -- Descriptive comment
    );

    
    -----------------------------
    -- TABLE: L0Files
    -----------------------------

    CREATE TABLE l0files (
        rid integer NOT NULL,                         -- Primary key
        dateobs date NOT NULL,                        -- FITS-header keyword: DATE-OBS
        ut time without time zone NOT NULL,           -- FITS-header keyword: UT
        datebeg timestamp without time zone NOT NULL, -- FITS-header keyword: DATE-BEG
        mjdobs double precision NOT NULL,             -- FITS-header keyword: MJD-OBS
        exptime real NOT NULL,                        -- FITS-header keyword: EXPTIME
        progname character varying(64) NOT NULL,      -- FITS-header keyword: PROGNAME
        imtype character varying(64) NOT NULL,        -- FITS-header keyword: IMTYPE
        sciobj character varying(64) NOT NULL,        -- FITS-header keyword: SCI-OBJ
        calobj character varying(64) NOT NULL,        -- FITS-header keyword: CAL-OBJ
        skyobj character varying(64) NOT NULL,        -- FITS-header keyword: SKY-OBJ
        "object" character varying(64) NOT NULL,      -- FITS-header keyword: TARGOBJ or OBJECT
        contentbits integer NOT NULL,                 -- BIT-WISE FLAGS FOR INCLUDING CCDs: BIT0: GREEN; BIT1: RED; BIT2: CA_HK 
        infobits bigint DEFAULT 0 NOT NULL,           -- Bit-wise information flags
        filename character varying(255) NOT NULL,     -- Full path and filename
        checksum character varying(32) NOT NULL,      -- MD5 checksum of entire file
        status smallint DEFAULT 0 NOT NULL,           -- Set to zero if bad and one if good (verify automatically with
                                                      -- DATASUM and CHECKSUM keywords, or set this manually later, if necessary)
        created timestamp without time zone           -- Timestamp of database record INSERT or last UPDATE
            DEFAULT now() NOT NULL,
        targname character varying(64),               -- FITS-header keyword: TARGNAME
        gaiaid character varying(64),                 -- FITS-header keyword: GAIAID
        twomassid character varying(64),              -- FITS-header keyword: 2MASSID
        ra double precision,                          -- FITS-header keyword: RA converted to decimal
        dec double precision,                         -- FITS-header keyword: DEC converted to decimal
        medgreen1 real,                               -- Median of GREEN_AMP1 image
        p16green1 real,                               -- 16th percentile of GREEN_AMP1 image
        p84green1 real,                               -- 84th percentile of GREEN_AMP1 image
        medgreen2 real,                               -- Median of GREEN_AMP2 image
        p16green2 real,                               -- 16th percentile of GREEN_AMP2 image
        p84green2 real,                               -- 84th percentile of GREEN_AMP2 image
        medgreen3 real,                               -- Median of GREEN_AMP3 image
        p16green3 real,                               -- 16th percentile of GREEN_AMP3 image
        p84green3 real,                               -- 84th percentile of GREEN_AMP3 image
        medgreen4 real,                               -- Median of GREEN_AMP4 image
        p16green4 real,                               -- 16th percentile of GREEN_AMP4 image
        p84green4 real,                               -- 84th percentile of GREEN_AMP4 image
        medred1 real,                                 -- Median of RED_AMP1 image
        p16red1 real,                                 -- 16th percentile of RED_AMP1 image
        p84red1 real,                                 -- 84th percentile of RED_AMP1 image
        medred2 real,                                 -- Median of RED_AMP2 image
        p16red2 real,                                 -- 16th percentile of RED_AMP2 image
        p84red2 real,                                 -- 84th percentile of RED_AMP2 image
        medcahk real,                                 -- Median of CA_HK image
        p16cahk real,                                 -- 16th percentile of CA_HK image
        p84cahk real,                                 -- 84th percentile of CA_HK image
        comment character varying(255),               -- Reason for status=0, etc.
        CONSTRAINT l0files_ra_check CHECK (((ra >= 0.0) AND (ra < 360.0))),
        CONSTRAINT l0files_dec_check CHECK (((dec >= -90.0) AND (dec <= 90.0)))
    );


    -----------------------------
    -- TABLE: L0infobits
    --
    -- Definitions of infobits for L0Files table only
    -- (CalFiles infobits have different definitions).
    -----------------------------

    CREATE TABLE l0infobits (
        bid integer NOT NULL,                        -- Primary key
        bit smallint NOT NULL,                       -- Bit number (allowed range is 0-63, inclusive)
        param1 real,                                 -- Parameter 1
        param2 real,                                 -- Parameter 2
        param3 real,                                 -- Parameter 3
        created timestamp without time zone          -- Creation timestamp of database record
            DEFAULT now() NOT NULL,
        definition character varying(256) NOT NULL,  -- Definition of bit and parameter(s)
        CONSTRAINT l0infobits_bit_check CHECK (((bit >= 0) AND (bit <= 63)))
    );
