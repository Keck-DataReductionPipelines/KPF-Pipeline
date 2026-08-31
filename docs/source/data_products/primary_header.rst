==============
PRIMARY Header
==============

Every keyword registered on the ``PRIMARY`` header, across all four science
levels. Content to follow.

A ``#`` in a keyword name is a template marker expanded over the five traces
(``EXSNR#`` -> ``EXSNR1`` .. ``EXSNR5``). ``Populated By`` names the pipeline
stage that writes the value; blank means the keyword is registered but not yet
sourced.

Unlike the other Data Products tables, this one is maintained by hand so the
whole header reads in observing order rather than config-file order.
``tests/regression/test_docs.py`` holds it to the config registries.

.. csv-table:: PRIMARY header keywords
   :file: PRIMARY-keywords.csv
   :header-rows: 1
   :widths: 12, 34, 10, 8, 14, 16
