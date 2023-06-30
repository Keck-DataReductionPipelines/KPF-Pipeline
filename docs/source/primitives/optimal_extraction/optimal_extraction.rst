Module Spectral Extraction
==========================

The purpose of the spectral extraction module is to convert 2D raw image (L0) into 1D spectra (L1) with high fidelity on photon counts and line shapes.

This module provides the following primitivies to be imported to the recipe:

	* ``OrderRectification``:  spectral order rectification

		Order rectification involves straightening up the spectral order trace, transforming it from a cursive shape to a straight line shape. The original 2D raw image (L0) is converted to be an L0 instance with spectral orders that are not curved. Using the order trace results, the pixels along the normal or vertical direction of each curved order trace are gathered, and then subjected to various weighting methods for further processing. Finally, these pixels are relocated to a 2D rectangular area.

	* ``SpectralExtraction``: spectral order summation extraction or optimal extraction

		The 2D science raw data along each order is converted into 1D data by using either

			- summation extraction method: the pixels along the rectified spectral order in same the column are directly summed up, or
			- optimal extraction method: the pixels of the rectified spectral order are weighted and summed up column by column. The weights for this process
                          are obtained from the corresponding pixels in the rectified flat L0 image.

		Using spectral extraction, the KPF DRP (Data Reduction Pipeline) processes the 2D spectral data from GREEN_CCD and RED_CCD extensions of L0. These data are then transformed into 1D data and stored in [GREEN, RED]_SCI_FLUX[1, 2, 3], [GREEN, RED]_SKY_FLUX, and [GREEN, RED]_CAL_FLUX extensions of L1, corresponding to the science fibers, calibration fiber and the sky fibers, respectively.

	* ``BaryCorrTable``: BARY_CORR table in L1 to include photo-weighted midpoints and barycentric velocity correction per order.

		For KPF DRP, the computation of the BARY_CORR table is based on the EXPMETER_SCI table in L0 data if it exists and the keywords from the primary header:
		*IMTYPE, DATE-MID, DATE-BEG, EXPTIME, TARGFRAM, TARGRA, TARGDEC, TARGPMRA, TARGPMDC, TARGPLAX, TARGEPOC, TARGRADV*.

		The resulting BARY_CORR table includes the following columns,

			- **GEOMID_UTC**: geometric midpoint UTC
			- **GEOMID_BJD**: geometric midpoint BJD
			- **PHOTON_BJD**: weighted-photon midpoint BJD
			- **BARYVEL**: barycentric velocity(m/sec)

.. toctree::
   :maxdepth: 2

   alg_optimal_extraction.rst
   alg_bary_corr.rst
   primitive_optimal_extraction.rst
   primitive_order_rectification.rst
   primitive_bary_corr.rst

