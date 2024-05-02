KPF Pipeline Development
========================



Repository Structure
--------------------

<content to add here>

Development Strategy
--------------------

<add content here about many topics>

Continuous Integration (CI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
<add general statements about how CI works>

The KPF DRP uses `pytest <https://docs.pytest.org/>`_ for CI.  Tests are automatically run using Jenkins and can also be run manually from within docker with commands like: ``> pytest -x --cov=kpfpipe --cov=modules --pyargs tests/regression/test_tools.py`` (see the makefile for examples of performance and validation tests).

Developing Quality Control (QC) Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here are the steps to adding a new quality control test.

#. Develop the code to determine if a KPF file passes or fails a QC metric.  See `this Jupyter notebook <QC_Example__Developing_a_QC_Method.ipynb>`_ for an example.  
#. Start a Git branch for your feature.
#. Write a method for your QC check in  `KPF-modules/quality_control/src/quality_control.py <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/master/modules/quality_control/src/quality_control.py>`_ based on code from your Jupyter notebook.  The method should return a boolean (``QC_pass``) that is True if the input KPF object passed the QC check and False otherwise.  One method to model yours on is ``L0_data_products_check()``.  Your method should be in the appropriate class for the data level of your QC check.  For example, for a QC check to an L0 object, put the method in the ``QCL0`` class in ``quality_control.py``.
#. Add information about your QC to the QCDefinitions class in ``quality_control.py``.  You can model your dictionary entries on the ones for ``name4 = 'L0_data_products_check'``.
#. Check that your QC works as expected.  See `this Jupyter notebook <QC_Example__L0_Data_Products_Check.ipynb>`_ for an example.  You can also modify the config file specified in this command and check the result: ``kpf -c configs/qc_diagnostics_example.cfg -r recipes/qc_diagnostics_example.recipe``.
#. Commit the changes to your Git branch and submit a pull request.
#. Document the new QC-related FITS keywords in the appropriate section of 'KPF Data Format' in Readthedocs.

Developing Diagnostic Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Diagnostics are similar to QC metrics in that they are used to evaluate data quality.  The difference is that QCs have a boolean value (pass/fail), while diagnostic information is more granular and can usually be expressed as a floating point number.  Below are the steps to developing a new diagnostic and adding the information to the headers.

#. Develop the code to analyze a standard L0/2D/L1/L2/Master KPF file.  This is usually done with one of the Analyze classes; for example, in the ``Analyze2D`` class (in ``modules/quicklook/src/analyze2D.py``), the method ``measure_2D_dark_current()`` performs photometry on regions of the 2D images and saves that information as class attributes.  Using the Analyze methods is a convenient way because those same methods are used to generate Quicklook data products, providing overlap with annotations that might be used on plots.
#. Start a Git branch for your feature.
#. Write a method in ``modules/quicklook/src/diagnostics.py``.  See the method ``add_headers_dark_current_2D()`` for example code that writes diagnostics related to dark current.
#. Add your method and the appropriate logic to trigger it (e.g., only compute dark current for dark exposures) to the appropriate section of ``_perform`` in the ``DiagnosticsFramework`` class in ``modules/quicklook/src/diagnostics_framework.py``.
#. Check that your QC works as expected.  This can be done by examining the FITS headers of files generated using the recipe ``recipes/quality_control.recipe``.
#. Commit the changes to your Git branch and submit a pull request.
#. Document the new Diagnostics-related FITS keywords in the appropriate section of 'KPF Data Format' in Readthedocs.

Developing Quicklook Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^

<AWH to add content here.>

Testing 
-------
