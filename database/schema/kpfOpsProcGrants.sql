--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsProcGrants.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 21 February 2023
--------------------------------------------------------------------------------------------------------------------------

grant EXECUTE on FUNCTION registerCalFile (
    startDate_           date,
    endDate_             date,
    level_               smallint,
    caltype_             character varying(32),
    object_              character varying(32),
    contentbits_         integer,
    nframes_             smallint,
    minmjd_              double precision,
    maxmjd_              double precision,
    infobits_            integer,
    filename_            character varying(255),
    checksum_            character varying(32),
    status_              smallint,
    createdBy_           character varying(32),
    created_             timestamp without time zone,
    comment_             character varying(255)
) to kpfporole;

grant EXECUTE on FUNCTION getCalFile (
    obsDate_         date,
    level_           smallint,
    caltype_         character varying(32),
    object_          character varying(32),
    contentbitmask_  integer
) to kpfporole, kpfreadrole;
