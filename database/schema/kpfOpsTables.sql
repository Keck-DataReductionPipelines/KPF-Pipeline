sql.--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsTables
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 17 February 2023
--------------------------------------------------------------------------------------------------------------------------


-----------------------------
-- TABLE: Calfiles
-----------------------------

SET default_tablespace = pipeline_data_01;

CREATE TABLE calfiles (
    cid integer NOT NULL,                            -- Primary key
    level smallint NOT NULL,                         -- Product level (L0, L1, or L2)
    caltype character varying(32) NOT NULL,          -- FITS-header keyword: IMTYPE in extension 4-6 (lowercase)
    "object" character varying(32) NOT NULL,         -- FITS-header keyword: TARGOBJ or OBJECT (lowercase)
    nframes smallint,                                -- FITS-header keyword: NFRAMES (GREEN CCD)
    minmjd double precision,                         -- FITS-header keyword: MINMJD (GREEN CCD)
    maxmjd double precision,                         -- FITS-header keyword: MAXMJD (GREEN CCD)
    infobits integer,                                -- FITS-header keyword: INFOBITS (GREEN CCD)
    startdate date NOT NULL,                         -- Start date for application of master (for earliest-in-time selection)
    enddate date NOT NULL,                           -- End date for application of master (may not be used)
    filename character varying(255) NOT NULL,
    checksum character varying(32) NOT NULL,
    status smallint DEFAULT 0 NOT NULL,
    createdby character varying(30) NOT NULL,
    created timestamp without time zone NOT NULL,    -- FITS-header keyword: CREATED (GREEN CCD) in Zulu time
    "comment" character varying(255)
);

ALTER TABLE calfiles OWNER TO kpfadminrole;

CREATE SEQUENCE calfiles_cid_seq
    START WITH 1
    INCREMENT BY 1
    NO MAXVALUE
    NO MINVALUE
    CACHE 1;

ALTER TABLE public.calfiles_cid_seq OWNER TO kpfadminrole;

ALTER TABLE calfiles ALTER COLUMN cid SET DEFAULT nextval('calfiles_cid_seq'::regclass);

SET default_tablespace = pipeline_indx_01;

ALTER TABLE ONLY calfiles ADD CONSTRAINT calfiles_pkey PRIMARY KEY (cid);

CREATE INDEX calfiles_caltype_idx ON calfiles (caltype);
CREATE INDEX calfiles_startdate_idx ON calfiles (startdate);
CREATE INDEX calfiles_enddate_idx ON calfiles (enddate);
CREATE INDEX calfiles_status_idx ON calfiles (status);

