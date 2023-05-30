--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsTableGrants.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 18 February 2023
--------------------------------------------------------------------------------------------------------------------------


-- kpfreadrole

REVOKE ALL ON TABLE calfiles FROM kpfreadrole;
GRANT SELECT ON TABLE calfiles TO GROUP kpfreadrole;

REVOKE ALL ON SEQUENCE calfiles_cid_seq FROM kpfreadrole;


-- kpfadminrole

REVOKE ALL ON TABLE calfiles FROM kpfadminrole;
GRANT ALL ON TABLE calfiles TO GROUP kpfadminrole;

REVOKE ALL ON SEQUENCE calfiles_cid_seq FROM kpfadminrole;
GRANT ALL ON SEQUENCE calfiles_cid_seq TO GROUP kpfadminrole;


-- kpfporole

REVOKE ALL ON TABLE calfiles FROM kpfporole;
GRANT INSERT,UPDATE,SELECT,REFERENCES ON TABLE calfiles TO kpfporole;

REVOKE ALL ON SEQUENCE calfiles_cid_seq FROM kpfporole;
GRANT USAGE ON SEQUENCE calfiles_cid_seq TO kpfporole;
