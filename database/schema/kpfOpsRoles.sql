--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsRoles.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 17 February 2023
--------------------------------------------------------------------------------------------------------------------------

create role kpfadminrole LOGIN SUPERUSER CREATEDB CREATEROLE;
create role kpfporole;
create role kpfreadrole;

GRANT kpfadminrole to rlaher;
GRANT kpfporole to rlaher;
GRANT kpfreadrole to rlaher;

GRANT kpfporole to kpfporuss;

GRANT kpfreadrole to bfulton;

GRANT kpfreadrole to cwang;

-- Verified bfulton inherits the following:
ALTER ROLE kpfreadrole CONNECTION LIMIT -1;
