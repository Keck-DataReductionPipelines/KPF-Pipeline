===============
Quality Control
===============

Every keyword registered on the ``QUALITY_CONTROL`` extension, one table per
science level. Content to follow.

The extension carries two kinds of card: 0/1 quality-check flags, written by the
level's QC class, and the diagnostic metrics recorded alongside them. A keyword
appears in the table for the level whose QC pass writes it.

L0
==

Placeholder: checks on the raw exposure -- read noise, saturation, and the
integrity of the delivered extensions.

.. csv-table:: L0 QUALITY_CONTROL keywords
   :file: _generated/L0-QUALITY_CONTROL-keywords.csv
   :header-rows: 1
   :widths: 14, 38, 10, 8, 14, 16

L1
==

Placeholder: checks on the assembled 2D frame -- calibration application,
background level, and per-frame variance.

.. csv-table:: L1 QUALITY_CONTROL keywords
   :file: _generated/L1-QUALITY_CONTROL-keywords.csv
   :header-rows: 1
   :widths: 14, 38, 10, 8, 14, 16

L2
==

Placeholder: checks on the extracted 1D spectra -- trace-by-trace signal to
noise, wavelength solution quality, and blaze behaviour.

.. csv-table:: L2 QUALITY_CONTROL keywords
   :file: _generated/L2-QUALITY_CONTROL-keywords.csv
   :header-rows: 1
   :widths: 14, 38, 10, 8, 14, 16

L4
==

Placeholder: checks on the derived radial velocities and the cross-correlation
functions behind them.

.. csv-table:: L4 QUALITY_CONTROL keywords
   :file: _generated/L4-QUALITY_CONTROL-keywords.csv
   :header-rows: 1
   :widths: 14, 38, 10, 8, 14, 16
