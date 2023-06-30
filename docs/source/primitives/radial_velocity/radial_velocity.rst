Module Radial Veloctiy
=======================

The purpose of the radial velocity module is to calculate CCFs and radial velocity from L1 data and produce L2 data with CCFs and RV for each spectral segment.


.. note::
	A spectral segment may encompass a wavelength range that spans one or more spectral orders.  In current KPF DRP, each spectral segment is considered equivalent to a single spectral order.

This module provides the following primitives to be imported to the recipe:

	* ``RadialVelocityInit``: initial work based on the target and fiber sources of each observation.

		This initial work involves preparing the velocity steps, mask lines and redshifts per orderlet using information from fiber sources, the target and the observing location, etc. These preparations are essential for computing CCFs and the radial velocity.

	* ``RadialVelocity``: CCF, radial velocity, and optional CCF reweighting.

		The CCF (cross correlation function) uses cross-correlation method on a shifted mask line in redshift space and the spectrum data of one segment to compute CCFs at each velocity step.

		Based on CCFs, the radial velocity is computed for:

			- each spectral segment of each orderlet.
			- each spectral segment of the summed CCFs across all science fibers (orderlets).
			- each orderlet based on the summed CCFs across all segments.
			- each CCD based on the summed CCFs across all segments and all science orderlets.

		In the process of radial velocity, the reweighting operation is selectively applied on CCFs based on whether a reweighting ratio table is specified. Please refer to the description on RadialVelocityReweighting for more details.

		With this primitive,  L2 data is produced  with the extensions containing CCFs and RVs of all orderlets.

		For KPF DRP, the resulting L2 instance has the following extensions for CCFs and RVs:

			- **GREEN_CCF, RED_CCF**: with 3D image containing original CCFs (not reweighted) for all orderlets of  green CCD and red CCD.
			- **GREEN_CCF_RW, RED_CCF_RW**: with 3D image containing reweighted CCFs for all orderlets of green CCD and red CCD.
			- **RV**: with the header and a table containing radial velocity values and radial velocity  errors for all orderlets, as well as their spectral segments of both green CCD and red CCD.


		The CCF-related extensions in KPF DRP consists of CCFs in 3D data format. These extensions also include the following specific keywords in their headers that provide the relevant information about the CCFs,

			.. list-table::

			   * - **key**
			     - **usage**

			   * - NAXIS
			     - dimension of the data

			   * - NAXIS1
			     - total velocity steps

			   * - NAXIS2
			     - total segments

			   * - NAXIS3
			     - total orderlets if NAXIS is 3.

			   * - STARTSEG
			     - first segment index of the CCFs

			   * - STARTV
			     - starting velocity for the CCFs

			   * - STEPS
			     - velocity interval for velocity steps

			   * - TOTALV
			     - total velocity steps

			   * - TOTALSCI
			     - total science orderlets

			   * - CCF1, CCF2, ..., CCF5
			     - related data source (extension) in L1 for each slice in the 3D CCFs

		           * - SCI_MASK, CAL_MASK, SKY_MASK
			     - masks for science fibers, calibration fiber and sky fiber


		The RV table consists of two parts, with the top part dedicated to the green CCD and the bottom part dedicated to the red CCD, and each row in the RV table corresponds to RV-related data for a specific segment of each CCD. This RV table includes the columns as follows,

			.. list-table::

			   * - **column name**
			     - **usage**

                           * - orderlet1, orderlet2, orderlet3
			     - rv values for science orderlets per segment and per CCD. the top part of the table is for GREEN CCD, and the bottom is for RED_CCD

                           * - s_wavelength, e_wavelength
			     - starting and ending wavelength per segment

                           * - segment no., order no.
			     - the associated segment index and order index for each row

                           * - RV
			     - rv values per segment and CCD based on the CCF summation across the science orderlets

                           * - RV error
			     - rv values per segment and CCD based on the unweighted CCF summation across the science orderlets

                           * - CAL RV, SKY RV
			     - rv values per segment and CCD for Cal fiber and Sky fiber

                           * - CAL error, SKY error
			     - rv error per segment and CCD for Cal fiber and Sky fiber based on unweighted CCFs.

                           * - CCFJD
			     - time in Julian Date format per segment and CCD

			   * - source[1, 2, 3], source CAL, source SKY
			     - related data source (extension) in L1

                           * - CCF Weights
			     - weights for science fibers if reweighting is applied.

		The RV extension includes the following keywords in the header that provide the relevant information for the RVs, (note: the keyword starting with 'CCD1' or 'CCD2' is for green CCD or red CCD),

			.. list-table::

			   * - **key**
			     - **usage**

			   * - STAR_RV
			     - systematic star radial velocity

			   * - CCD[1, 2]ROW
			     - start row index in the RV table for green CCD or red CCD

			   * - CCD[1,2]RV[1,2,3]
			     - rv per science fiber (SCI1-3) and per CCD based on CCF summation across segments

			   * - CCD[1,2]ERV[1,2,3]
			     - rv error per science fiber (SCI1-3) and per CCD based on CCF summation across segments

			   * - CCD[1,2]RV[C,S]
			     - rv per CAL or SKY fiber and  per CCD based on CCF summation across segments

			   * - CCD[1,2]ERV[C,S]
			     - rv error per CAL or SKY fiber and per CCD based on CCF summation across segments

			   * - CCD[1,2]RV
			     - rv per CCD based on CCF summation across segments and science orderlets

			   * - CCD[1,2]ERV
			     - rv error per CCD based on CCF summation across segments and science orderlets

                           * - RV[1,2]CORR
			     - if the science rv value is corrected by the CAL rv value per CCD

			   * - CCD[1,2]JD
			     - exposure time per CCD in Julian date format

			   * - RWCCF[1-5]
			     - if the CCFs for CCFn is reweighted. CCFn is defined in the header of CCF related extensions


	* ``RadialVelocityReweightingRef``: reweighting ratio table used for CCF reweighting.

		This primitive provides the following 2 methods which either builds or loads the ratio table for CCF reweighting:

			#. To build a ratio table dynamically, start by collecting L2 candidates along with the original CCFs. Identify the observation with the best quality as the 'template' based on the highest count order. This order can be determined by either the highest mean count or the maximum count in CCF. Normalize the peaks of the CCFs by setting the mean value (or the maximum value) of the highest count order to 1. Adjust the mean (or the maximum) values of all other segments proportionally to create a ratio table.
			#. To load a table with pre-defined static CCF weights, then the reweighting can be applied to the original (unweighted) CCFs directly inside the RadialVelocity call.
		Method 1 builds the reweighting ratio table based on a set of L2. That means the reweighting process may need to wait until the entire set of L2 data is calculated. On the other hand, Method 2 allows for immediate reweighting on CCFs as soon as  when each L2 data is initially calculated.


		KPF DRP employs the second method for CCF reweighting. In this approach, the CCFs are reweighted within the RadialVelocity module when they are initially calculated, and the subsequent RV computation is performed directly on the reweighted CCFs.


	* ``RadialVelocityReweighting``: CCF reweighting per observation based on the loaded ratio table.

		For KPF DRP, the reweighting ratio table consists of columns representing weights for different masks, such as 'espresso' masks, 'lfc', 'thar', 'etalon', and so on. The specific column selected depends on the fiber source of each orderlet. Additionally, a list of masks enabling the reweighting is one optional setting for this primitive call. The following reweighting methods are provided:

			- **ccf_max, ccf_mean**: Adjust the CCFs in such ways that either the maximum or the mean value among all the orders aligns with the ratios specified in the table.
			- **ccf_static**: First, CCFs of each order are normalized by dividing them by the CCF summation. Secondly, the CCFs are reweighted according to  the weights specified in the ratio table.


		KPF DRP applies the method of **ccf_static** at the point.

.. toctree::
   :maxdepth: 2

   alg_radial_velocity.rst
   alg_rv_init.rst
   alg_rv_mask_line.rst
   alg_rv_base.rst
   primitive_radial_velocity.rst
   primitive_rv_init.rst
   primitive_rv_reweighting.rst
   primitive_rv_reweighting_ref.rst
