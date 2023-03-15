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
