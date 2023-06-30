Module Order Trace
==================

The purpose of the order trace module  is to determine the location and the curvature of the spectral order from fiber flat exposures, the L0 flat files.

This module provides the following primitives which can be imported to the recipe:

	* ``OrderTrace``: spectral order trace

		This process involves the tracing and fitting of the order maximum, which is approximately located at the center of the spectral order. This tracing and fitting are done using a cubic polynomial (degree 3). Furthermore, the extraction windows above and below the fitted curve are determined to cover the entire spectral order.

		The order trace result is stored in an instance of Pandas DataFrame and optionally a csv file in which each row contains the following numbers to represent the location and the coverage of each spectral order:

			- polynomial coefficients in the order of increasing degree,

			- the extraction windows below and above the fitted polynomial, a.k.a. bottom edge and top edge,

			- the starting and ending horizontal positions of the trace on the 2D image domain.


		For KPF, the order trace module outputs two csv files for the image of green CCD and red CCD, respectively.

		Here is an example containing the order trace result of the first few spectral orders::

			,Coeff0,Coeff1,Coeff2,Coeff3,BottomEdge,TopEdge,X1,X2
			0,0.5445672380235284,0.004080092310042913,2.0577957001418133e-06,-3.5091410725241624e-10,6.7088,6.4642,86,4079
			1,15.266582090451204,0.009653957574193602,-2.982172565863953e-07,-5.072604426717458e-11,8.5358,9.4401,146,4079
			2,34.06984874062919,0.009905479153527023,-4.360738114529129e-07,-3.499266829075321e-11,8.5599,8.8955,44,4079
			3,49.1875320114883,0.009978684769828283,-4.518502978011858e-07,-3.045119160129834e-11,4.3219,4.4365,0,4079
			4,64.90185616615426,0.00919113555034767,-2.1738284969730856e-07,-5.283099431400399e-11,9.5635,9.4245,0,4079
			:

	* ``OrderMask``: spectral order mask

		Using the order trace result, order mask produces an L0 instance with the same image size as the
                input L0 spectral data. In the generated 2D image of order mask L0, the pixels corresponding to the orderlets are set to 1, while non-orderlet pixels
                are set to be 0 by default. Orderlet pixels refers to the piexls that belong to the spectral orders as defined in the order trace result.
	        This order mask L0 image provides a binary representation where the orderlets are distinguished from the non-orderlet areas.

.. toctree::
   :maxdepth: 2

   alg_order_trace.rst
   primitive_order_trace.rst
   alg_order_mask.rst
   primitive_order_mask.rst
