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


-- Get the nearest-in-time-before calibration file
-- for a given observation date, level, caltype, and object.
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

    begin

        caltype__ := lower(caltype_);
        object__ := lower(object_);

        select cId, filename, checksum, infobits, startDate
        into cId_, filename_, checksum_, infobits_, startDate_
        from CalFiles
        where status > 0
        and startDate <= obsDate_
        and level = level_
        and caltype = caltype__
        and object = object__
        and cast((contentbits & contentbitmask_) as integer) = contentbitmask_
        order by startDate desc                  -- Descending order for backward-looking.
        limit 1;

        if found then
            select cId_, level_, caltype__, object__, filename_, checksum_, infobits_, startDate_ into r_;
            return next r_;
        end if;

        return;  -- Required to indicate function is finished executing.

    end;


$$ language plpgsql;
