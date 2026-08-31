===========
Data Models
===========

The four science data levels and the extensions that make up each one. Content
to follow.

General Info
============

Placeholder for what the data levels are, how a file advances from one to the
next, and the conventions shared by all of them. Content to follow.

L0
==

Placeholder: the raw KPF exposure as delivered by the instrument, one image
extension per detector amplifier plus the ancillary and provenance tables.

.. csv-table:: L0 extensions
   :file: _generated/L0-extensions.csv
   :header-rows: 1
   :widths: 6, 22, 12, 8, 8, 44

L1
==

Placeholder: assembled 2D frames -- amplifiers stitched, overscan removed,
calibrations applied -- with per-frame variance.

.. csv-table:: L1 extensions
   :file: _generated/L1-extensions.csv
   :header-rows: 1
   :widths: 6, 22, 12, 8, 8, 44

L2
==

Placeholder: extracted 1D spectra on the EPRV standard layout, one flux,
wavelength, variance and blaze extension per trace.

.. csv-table:: L2 extensions
   :file: _generated/L2-extensions.csv
   :header-rows: 1
   :widths: 6, 22, 12, 8, 8, 44

L4
==

Placeholder: derived radial velocities and the cross-correlation functions they
come from.

.. csv-table:: L4 extensions
   :file: _generated/L4-extensions.csv
   :header-rows: 1
   :widths: 6, 22, 12, 8, 8, 44
