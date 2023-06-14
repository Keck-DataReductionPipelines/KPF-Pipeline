--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsTableGrants.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 18 February 2023
--------------------------------------------------------------------------------------------------------------------------


-------------------
-- CalFiles table
-------------------

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


-------------------
-- L0files table
-------------------

-- kpfreadrole

REVOKE ALL ON TABLE l0files FROM kpfreadrole;
GRANT SELECT ON TABLE l0files TO GROUP kpfreadrole;

REVOKE ALL ON SEQUENCE l0files_rid_seq FROM kpfreadrole;


-- kpfadminrole

REVOKE ALL ON TABLE l0files FROM kpfadminrole;
GRANT ALL ON TABLE l0files TO GROUP kpfadminrole;

REVOKE ALL ON SEQUENCE l0files_rid_seq FROM kpfadminrole;
GRANT ALL ON SEQUENCE l0files_rid_seq TO GROUP kpfadminrole;


-- kpfporole

REVOKE ALL ON TABLE l0files FROM kpfporole;
GRANT INSERT,UPDATE,SELECT,REFERENCES ON TABLE l0files TO kpfporole;

REVOKE ALL ON SEQUENCE l0files_rid_seq FROM kpfporole;
GRANT USAGE ON SEQUENCE l0files_rid_seq TO kpfporole;


-------------------
-- L0infobits table
-------------------

-- kpfreadrole

REVOKE ALL ON TABLE l0infobits FROM kpfreadrole;
GRANT SELECT ON TABLE l0infobits TO GROUP kpfreadrole;

REVOKE ALL ON SEQUENCE l0infobits_bid_seq FROM kpfreadrole;


-- kpfadminrole

REVOKE ALL ON TABLE l0infobits FROM kpfadminrole;
GRANT ALL ON TABLE l0infobits TO GROUP kpfadminrole;

REVOKE ALL ON SEQUENCE l0infobits_bid_seq FROM kpfadminrole;
GRANT ALL ON SEQUENCE l0infobits_bid_seq TO GROUP kpfadminrole;


-- kpfporole

REVOKE ALL ON TABLE l0infobits FROM kpfporole;
GRANT INSERT,UPDATE,SELECT,REFERENCES ON TABLE l0infobits TO kpfporole;

REVOKE ALL ON SEQUENCE l0infobits_bid_seq FROM kpfporole;
GRANT USAGE ON SEQUENCE l0infobits_bid_seq TO kpfporole;









