--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsTables
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 17 February 2023
--------------------------------------------------------------------------------------------------------------------------


-----------------------------
-- TABLE: CalFiles
-----------------------------

SET default_tablespace = pipeline_data_01;

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

ALTER TABLE calfiles OWNER TO kpfadminrole;

CREATE SEQUENCE calfiles_cid_seq
    START WITH 1
    INCREMENT BY 1
    NO MAXVALUE
    NO MINVALUE
    CACHE 1;

ALTER SEQUENCE calfiles_cid_seq OWNER TO kpfadminrole;

ALTER TABLE calfiles ALTER COLUMN cid SET DEFAULT nextval('calfiles_cid_seq'::regclass);

SET default_tablespace = pipeline_indx_01;

ALTER TABLE ONLY calfiles ADD CONSTRAINT calfiles_pkey PRIMARY KEY (cid);

CREATE INDEX calfiles_caltype_idx ON calfiles (caltype);
CREATE INDEX calfiles_startdate_idx ON calfiles (startdate);
CREATE INDEX calfiles_enddate_idx ON calfiles (enddate);
CREATE INDEX calfiles_status_idx ON calfiles (status);


-----------------------------
-- TABLE: L0Files
-----------------------------

SET default_tablespace = pipeline_data_01;

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

ALTER TABLE l0files OWNER TO kpfadminrole;

CREATE SEQUENCE l0files_rid_seq
    START WITH 1
    INCREMENT BY 1
    NO MAXVALUE
    NO MINVALUE
    CACHE 1;

ALTER SEQUENCE l0files_rid_seq OWNER TO kpfadminrole;

ALTER TABLE l0files ALTER COLUMN rid SET DEFAULT nextval('l0files_rid_seq'::regclass);

SET default_tablespace = pipeline_indx_01;

ALTER TABLE ONLY l0files ADD CONSTRAINT l0files_pkey PRIMARY KEY (rid);

ALTER TABLE ONLY l0files ADD CONSTRAINT l0filespk UNIQUE (dateobs, ut);

CREATE INDEX l0files_dateobs_idx ON l0files (dateobs);
CREATE INDEX l0files_ut_idx ON l0files (ut);
CREATE INDEX l0files_mjdobs_idx ON l0files (mjdobs);
CREATE INDEX l0files_status_idx ON l0files (status);
CREATE INDEX l0files_infobits_idx ON l0files (infobits);

-- Uncomment only if Q3C EXTENSION has been installed into database.
-- Q3C is not required for normal operations, but will speed up ad-hoc cone searches on (ra, dec).
-- CREATE INDEX l0files_radec_idx ON l0files (q3c_ang2ipix(ra, dec));
-- CLUSTER l0files_radec_idx ON l0files;
-- ANALYZE l0files;


-----------------------------
-- TABLE: L0infobits
--
-- Definitions of l0infobits for L0Files table only
-- (CalFiles l0infobits have different definitions).
-----------------------------

SET default_tablespace = pipeline_data_01;

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

ALTER TABLE l0infobits OWNER TO kpfadminrole;

CREATE SEQUENCE l0infobits_bid_seq
    START WITH 1
    INCREMENT BY 1
    NO MAXVALUE
    NO MINVALUE
    CACHE 1;

ALTER SEQUENCE l0infobits_bid_seq OWNER TO kpfadminrole;

ALTER TABLE l0infobits ALTER COLUMN bid SET DEFAULT nextval('l0infobits_bid_seq'::regclass);

SET default_tablespace = pipeline_indx_01;

ALTER TABLE ONLY l0infobits ADD CONSTRAINT l0infobits_pkey PRIMARY KEY (bid);

ALTER TABLE ONLY l0infobits ADD CONSTRAINT l0infobitspk UNIQUE (bit, created);

CREATE INDEX l0infobits_bit_idx ON l0infobits (bit);
