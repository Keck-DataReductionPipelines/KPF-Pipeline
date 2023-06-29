Module Barycentric Correction
=============================

The barycentric correction module utilizes the "barycorrpy" package to calculate the barycentric velocity correction over a specified period of time, typically spanning multiple days.

To use this module, you need to provide the starting day in Julian data format, specify the period of days for which the correction is required, and provide the observation-related configuration. The module will then calculate the barycentric velocity correction values for each day within the specified period.

This module outputs a list of barycentric velocity correction values, with each value corresponding to a specific day in the given period. Additionally, the module provides the minimum and maximum redshift values derived from the list of the corrections.

.. toctree::
   :maxdepth: 2

   alg_barycentric_correction.rst
   primitive_barycentric_correction.rst
