--------------------------------------------------------------------------------------------------------------------------
-- kpfOpsL0InfobitsInserts.sql
--
-- Russ Laher (laher@ipac.caltech.edu)
--
-- 09 June 2023
--------------------------------------------------------------------------------------------------------------------------

INSERT INTO l0infobits (bit, param1, param2, definition) values (0, 5.0, 10000.0, 'GREEN_AMP1 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (1, 5.0, 10000.0, 'GREEN_AMP2 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (2, 5.0, 10000.0, 'GREEN_AMP3 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (3, 5.0, 10000.0, 'GREEN_AMP4 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (4, 5.0, 10000.0, 'RED_AMP1 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (5, 5.0, 10000.0, 'RED_AMP2 Dead-detector flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (6, 5.0, 10000.0, 'CA_HK flag: More than 5% of pixels in readout channel have values less than 10000 D.N.');

INSERT INTO l0infobits (bit, param1, param2, definition) values (7, 15.0, 5.0e8, 'GREEN_AMP1 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (8, 15.0, 5.0e8, 'GREEN_AMP2 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (9, 15.0, 5.0e8, 'GREEN_AMP3 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (10, 15.0, 5.0e8, 'GREEN_AMP4 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (11, 15.0, 5.0e8, 'RED_AMP1 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (12, 15.0, 5.0e8, 'RED_AMP2 Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
INSERT INTO l0infobits (bit, param1, param2, definition) values (13, 15.0, 5.0e8, 'CA_HK Saturated-detector flag: More than 15% of pixels in readout channel have values more than 5.0e8 D.N.');
