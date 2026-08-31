====================
Receipt & DRP Config
====================

Two provenance extensions present on every data level. Neither holds science
data; together they record what was done to a file and under what settings, so
a reduction can be traced and reproduced. Content to follow.

RECEIPT
=======

Placeholder: the file's processing history -- one row per pipeline operation
applied to it, accumulated as the file advances through the levels, so any
product carries the full record of how it was made.

DRP_CONFIG
==========

Placeholder: the pipeline configuration in force for the reduction, stored as a
two-column key/value table, so a product can be re-derived under the exact
settings that produced it.
