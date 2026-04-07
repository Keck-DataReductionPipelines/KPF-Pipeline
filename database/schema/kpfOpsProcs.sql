--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsProcs.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 21 February 2023
--------------------------------------------------------------------------------------------------------------------------


-- Insert a new record into or update an existing record in the CalFiles table.
--
create function registerCalFile (
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
)
    returns integer as $$

    declare

        cId_            integer;
        endDate__       date;
        created__       timestamp without time zone;
        caltype__       character varying(32);
        object__        character varying(32);

    begin

        caltype__ := lower(caltype_);
        object__ := lower(object_);

        if (endDate_ is null) then
            endDate__ := startDate_ + cast( '1 day' as interval);
        else
            endDate__ := endDate_;
        end if;

        if (created_ is null) then
            created__ := now();
        else
            created__ := created_;
        end if;

        select cId
        into cId_
        from CalFiles
        where startDate = startDate_
        and endDate = endDate__
        and level = level_
        and caltype = caltype__
        and object = object__;

        if not found then


            -- Insert CalFiles record.

            begin

                insert into CalFiles
                (startDate,
                 endDate,
                 level,
                 caltype,
                 object,
                 contentbits,
                 nframes,
                 minmjd,
                 maxmjd,
                 infobits,
                 filename,
                 checksum,
                 status,
                 createdBy,
                 created,
                 comment
                )
                values
                (startDate_,
                 endDate__,
                 level_,
                 caltype__,
                 object__,
                 contentbits_,
                 nframes_,
                 minmjd_,
                 maxmjd_,
                 infobits_,
                 filename_,
                 checksum_,
                 status_,
                 createdBy_,
                 created__,
                 comment_
                )
                returning cId into strict cId_;
                exception
                    when no_data_found then
                        raise exception
                            '*** Error in registerCalFile: Row could not be inserted into CalFiles table.';
            end;

        else


            -- Update CalFiles record.

            update CalFiles
            set startDate = startDate_,
                endDate = endDate__,
                level = level_,
                caltype = caltype__,
                object = object__,
                contentbits = contentbits_,
                nframes = nframes_,
                minmjd = minmjd_,
                maxmjd = maxmjd_,
                infobits = infobits_,
                filename = filename_,
                checksum = checksum_,
                status = status_,
                createdBy = createdBy_,
                created = created__,
                comment = comment_
            where cId = cId_;

        end if;

        return cId_;

    end;

$$ language plpgsql;


-- Get the nearest-in-time-before/after calibration file
-- for a given observation date, level, caltype, and object.
-- Before in time is given preference to after in time.
-- The status of the calibration file must be greater than zero.
--
create function getCalFile (
    obsDate_         date,
    level_           smallint,
    caltype_         character varying(32),
    object_          character varying(32),
    contentbitmask_  integer
)
    returns setof record as $$

    declare

        cId_             integer;
        caltype__        character varying(32);
        object__         character varying(32);
        filename_        character varying(255);
        checksum_        character varying(32);
        infobits_        integer;
        startDate_       date;
        r_               record;
        minnframes_      smallint;

    begin

        minnframes_ := 5;
        caltype__ := lower(caltype_);
        object__ := lower(object_);

        select cId, filename, checksum, infobits, startDate
        into cId_, filename_, checksum_, infobits_, startDate_
        from CalFiles
        where status > 0
        and level = level_
        and caltype = caltype__
        and object = object__
        and ((nframes >= minnframes_) or (nframes is null))
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        order by
          abs(cast(extract(days from (cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone))) as numeric)) asc,
          cast(extract(days from cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone)) as integer) desc
        limit 1;

        if found then
            select cId_, level_, caltype__, object__, filename_, checksum_, infobits_, startDate_ into r_;
            return next r_;
        end if;

        return;  -- Required to indicate function is finished executing.

    end;


$$ language plpgsql;


-- Overloaded getCalFile function with additional parameter
-- for maximum startdate age relative to the observation date.
-- Get the nearest-in-time-before/after calibration file
-- for a given observation date, level, caltype, and object.
-- Before in time is given preference to after in time.
-- The status of the calibration file must be greater than zero.
--
create function getCalFile (
    obsDate_         date,
    level_           smallint,
    caltype_         character varying(32),
    object_          character varying(32),
    contentbitmask_  integer,
    maxage_          interval
)
    returns setof record as $$

    declare

        cId_             integer;
        caltype__        character varying(32);
        object__         character varying(32);
        filename_        character varying(255);
        checksum_        character varying(32);
        infobits_        integer;
        startDate_       date;
        r_               record;
        negmaxage_       interval;
        minnframes_      smallint;

    begin

        minnframes_ := 5;
        caltype__ := lower(caltype_);
        object__ := lower(object_);
        negmaxage_ := cast((cast(maxage_ as text) || ' ago') as interval);

        select cId, filename, checksum, infobits, startDate
        into cId_, filename_, checksum_, infobits_, startDate_
        from CalFiles
        where status > 0
        and level = level_
        and caltype = caltype__
        and object = object__
        and ((nframes >= minnframes_) or (nframes is null))
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        and cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone) <= maxage_
        and cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone) >= negmaxage_
        order by
          abs(cast(extract(days from (cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone))) as numeric)) asc,
          cast(extract(days from cast(obsDate_ as timestamp without time zone) - cast(startDate as timestamp without time zone)) as integer) desc
        limit 1;

        if found then
            select cId_, level_, caltype__, object__, filename_, checksum_, infobits_, startDate_ into r_;
            return next r_;
        end if;

        return;  -- Required to indicate function is finished executing.

    end;


$$ language plpgsql;


-- Insert a new record into or update an existing record in the ReadNoise table.
--
create function registerReadNoise (
    rId_           integer,
    rngreen1_      real,
    rngreen2_      real,
    rngreen3_      real,
    rngreen4_      real,
    rnred1_        real,
    rnred2_        real,
    rnred3_        real,
    rnred4_        real,
    rncahk_        real,
    greenreadtime_ real,
    redreadtime_   real,
    readspeed_     character varying(16)
)
    returns void as $$

    declare

        rId__     integer;

    begin


        -- Insert or update record, as appropriate.

        select rId
        into rId__
        from ReadNoise
        where rId = rId_;

        if not found then


            -- Insert ReadNoise record.

            begin

                insert into ReadNoise
                (rId, rngreen1, rngreen2, rngreen3, rngreen4, rnred1, rnred2, rnred3, rnred4, rncahk,
		greenreadtime, redreadtime, readspeed)
                values
                (rId_, rngreen1_, rngreen2_, rngreen3_, rngreen4_, rnred1_, rnred2_, rnred3_, rnred4_, rncahk_,
		greenreadtime_, redreadtime_, readspeed_);
                exception
                    when no_data_found then
                        raise exception
                            '*** Error in registerReadNoise: ReadNoise record for rId=% not inserted.', rId_;

            end;

        else


            -- Update ReadNoise record.

            update ReadNoise
            set rngreen1 = rngreen1_,
                rngreen2 = rngreen2_,
                rngreen3 = rngreen3_,
                rngreen4 = rngreen4_,
		rnred1 = rnred1_,
                rnred2 = rnred2_,
                rnred3 = rnred3_,
                rnred4 = rnred4_,
                rncahk = rncahk_,
		greenreadtime = greenreadtime_,
                redreadtime = redreadtime_,
                readspeed = readspeed_
            where rId = rId_;

        end if;

    end;

$$ language plpgsql;


-- Get the nearest-in-time-before-only calibration file
-- for a given observation date, level, caltype, and object.
-- The status of the calibration file must be greater than zero.
--
create function getCalFileBefore (
    obsDate_         timestamp,
    level_           smallint,
    caltype_         character varying(32),
    object_          character varying(32),
    contentbitmask_  integer,
    maxage_          interval
)
    returns setof record as $$

    declare

        cId_             integer;
        caltype__        character varying(32);
        object__         character varying(32);
        filename_        character varying(255);
        checksum_        character varying(32);
        infobits_        integer;
        startDate_       date;
        r_               record;
        minnframes_      smallint;

    begin

        minnframes_ := 5;
        caltype__ := lower(caltype_);
        object__ := lower(object_);

        select cId, filename, checksum, infobits, startDate
        into cId_, filename_, checksum_, infobits_, startDate_
        from CalFiles
        where status > 0
        and level = level_
        and caltype = caltype__
        and object = object__
        and ((nframes >= minnframes_) or (nframes is null))
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        and cast(obsDate_ as timestamp without time zone) > startDate
        and cast(obsDate_ as timestamp without time zone) - startDate <= maxage_
        order by
          startDate desc
        limit 1;

        if found then
            select cId_, level_, caltype__, object__, filename_, checksum_, infobits_, startDate_ into r_;
            return next r_;
        end if;

        return;  -- Required to indicate function is finished executing.

    end;


$$ language plpgsql;


-- Get the nearest-in-time-after-only calibration file
-- for a given observation date, level, caltype, and object.
-- The status of the calibration file must be greater than zero.
--
create function getCalFileAfter (
    obsDate_         timestamp,
    level_           smallint,
    caltype_         character varying(32),
    object_          character varying(32),
    contentbitmask_  integer,
    maxage_          interval
)
    returns setof record as $$

    declare

        cId_             integer;
        caltype__        character varying(32);
        object__         character varying(32);
        filename_        character varying(255);
        checksum_        character varying(32);
        infobits_        integer;
        startDate_       date;
        r_               record;
        minnframes_      smallint;

    begin

        minnframes_ := 5;
        caltype__ := lower(caltype_);
        object__ := lower(object_);

        select cId, filename, checksum, infobits, startDate
        into cId_, filename_, checksum_, infobits_, startDate_
        from CalFiles
        where status > 0
        and level = level_
        and caltype = caltype__
        and object = object__
        and ((nframes >= minnframes_) or (nframes is null))
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        and cast(obsDate_ as timestamp without time zone) < startDate
        and startDate - cast(obsDate_ as timestamp without time zone) <= maxage_
        order by
          startDate asc
        limit 1;

        if found then
            select cId_, level_, caltype__, object__, filename_, checksum_, infobits_, startDate_ into r_;
            return next r_;
        end if;

        return;  -- Required to indicate function is finished executing.

    end;


$$ language plpgsql;


-- Compute modified Julian date from observation date/time.
--
create function computeMJD (
    timestamp_      timestamp
)

    returns double precision as $$

    declare

        year_            double precision;
        month_           double precision;
        day_             double precision;
        hour_            double precision;
        min_             double precision;
        sec_             double precision;
        a_               double precision;
        y_               double precision;
        m_               double precision;
        jdn_             double precision;
        jd_              double precision;
        mjd_             double precision;

    begin

        year_  := extract(year from timestamp_);
        month_ := extract(month from timestamp_);
        day_   := extract(day from timestamp_);
        hour_  := extract(hour from timestamp_);
        min_   := extract(minute from timestamp_);
        sec_   := extract(second from timestamp_);

        a_ := floor((14.0 - month_)/12.0);
        y_ := year_ + 4800.0 - a_;
        m_ := month_ + 12.0 * a_ - 3.0;
        jdn_ := day_ + floor((153.0 * m_ + 2.0)/5.0) + 365.0 * y_ + floor(y_/4.0)
                - floor(y_/100.0) + floor(y_/400.0) - 32045.0;
        jd_ := sec_/86400.0 + min_/1440.0 + (hour_ - 12.0)/24.0 + jdn_;
        mjd_ := jd_ - 2400000.5;

        return mjd_;

    end;

$$ language plpgsql;


-- Get all L0 FITS files for specified night, IMTYPE, OBJECT, and contentbitmask.
-- Since raw calibration data are taken nomimally in the afternoon (Pacific time),
-- this time interval can span UT midnight, and special logic is used to expand
-- the query domain.
--
create function getL0FitsFilesForCalibration (
    dateobs_ date,
    imtype_ character varying(32),
    object_ character varying(32),
    contentbitmask_  integer,
    hoursbeforemidnight_ real,
    hoursaftermidnight_ real
)
    returns setof record as $$

    declare

        mjd_             double precision;
        startmjd_        double precision;
        endmjd_          double precision;
        r_               record;

    begin

        select computeMJD(cast(dateobs_ as timestamp)) into mjd_;
        startmjd_ := mjd_ - cast(hoursbeforemidnight_ as double precision) / 24.0;
        endmjd_   := mjd_ + cast(hoursaftermidnight_ as double precision) / 24.0;

        for r_ in
        select rId, filename, checkSum, infobits, ut, mjdobs
        from L0Files
        where lower(imtype) = lower(imtype_)
        and lower(object) = lower(object_)
        and mjdobs > startmjd_
        and mjdobs < endmjd_
        and status > 0
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        order by mjdobs
        loop
            return next r_;
        end loop;

        return;  -- Required to indicate function is finished executing.

    end;

$$ language plpgsql;
