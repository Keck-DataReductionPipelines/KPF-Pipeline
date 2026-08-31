==============
PRIMARY Header
==============

Every keyword registered on the ``PRIMARY`` header, across all four science
levels. Content to follow.

``Level`` is the data level at which a keyword is introduced; the PRIMARY
header is cumulative, so a keyword introduced at L0 is present at L1, L2 and L4
as well. A ``#`` in a keyword name is a template marker expanded over the five
traces (``EXSNR#`` -> ``EXSNR1`` .. ``EXSNR5``). ``Populated By`` names the
pipeline stage that writes the value; blank means the keyword is registered but
not yet sourced.

.. csv-table:: PRIMARY header keywords
   :file: _generated/PRIMARY-keywords.csv
   :header-rows: 1
   :widths: 12, 6, 34, 10, 8, 14, 16
