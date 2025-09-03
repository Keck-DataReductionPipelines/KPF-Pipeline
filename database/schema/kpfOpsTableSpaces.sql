--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsTableSpaces.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 18 February 2023
--------------------------------------------------------------------------------------------------------------------------



-- mkdir -p $KPFDB/tablespacedata1
-- mkdir -p $KPFDB/tablespaceindx1
CREATE TABLESPACE pipeline_data_01 LOCATION '/data/user/rlaher/kpfdb/tablespacedata1';
CREATE TABLESPACE pipeline_indx_01 LOCATION '/data/user/rlaher/kpfdb/tablespaceindx1';


-- Put the CalFiles table and its indexes on a fast storage device.
-- mkdir -p /pgdbs/kpfdb/tablespacedata2
-- mkdir -p /pgdbs/kpfdb/tablespaceindx2
CREATE TABLESPACE pipeline_data_02 LOCATION '/pgdbs/kpfdb/tablespacedata2';
CREATE TABLESPACE pipeline_indx_02 LOCATION '/pgdbs/kpfdb/tablespaceindx2';
