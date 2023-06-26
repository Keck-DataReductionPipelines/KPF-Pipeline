Module Ca H&K 
==============

The purpose of the Ca H&K module is to extract Ca H&K spectra and convert 2D raw image (L0) into 1D spectra (L1) data.

This module provides the following primitivie to be imported to the recipe:

	* ``CaHKExtraction``: Calcium H&K extraction.

		The following processes are performed,

			- performing flat and bias correction on the H&K science spectrum.
			- extracting Ca H&K spectra from the extension CA_HK of L0 data based on the provided position information of all the orders (both science and sky). The extracted spectra from the same column are summed together. In the resulting L1 data, the reduced Ca H&K spectra from the science fiber and the sky fiber are stored in the CA_HK_SCI and CA_HK_SKY extensions, respectively.
			- loading the wavelength solution files for science and sky fibers and converting the WLS (wavelength solution) data to extensions CA_HK_SCI_WAVE and CA_HK_SKY_WAVE in L1 data, respectively.

.. toctree::
   :maxdepth: 2

   alg_ca_hk.rst
   primitive_ca_hk_extraction.rst
