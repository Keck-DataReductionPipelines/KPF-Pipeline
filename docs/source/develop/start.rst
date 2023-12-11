KPF Pipeline Development
========================



Repository Structure
--------------------

<content to add here>

Development Strategy
--------------------

<add content here about many topics>

Developing Quality Control (QC) Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here are the steps to adding a new quality control test.

#. Develop the code to determine if a KPF file passes or fails a QC metric.  See `this Jupyter notebook <QC_Example__Developing_a_QC_Method.ipynb>`_ for an example.  
#. Start a Git branch for your feature.
#. Write a method for your QC check in  `KPF-modules/quality_control/src/quality_control.py <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/master/modules/quality_control/src/quality_control.py>`_ based on code from your Jupyter notebook.  The method should return a boolean (``QC_pass``) that is True if the input KPF object passed the QC check and False otherwise.  One method to model yours on is ``L0_data_products_check()``.  Your method should be in the appropriate class for the data level of your QC check.  For example, for a QC check to an L0 object, put the method in the ``QCL0`` class in ``quality_control.py``.
#. Add information about your QC to the QCDefinitions class in ``quality_control.py``.  You can model your dictionary entries on the ones for ``name4 = 'L0_data_products_check'``.
#. Check that your QC works as expected.  See `this Jupyter notebook <QC_Example__L0_Data_Products_Check.ipynb>`_ for an example.  You can also modify the config file specified in this command and check the result: ``kpf -c configs/quality_control_example.cfg -r recipes/quality_control_example.recipe``.
#. Commit the changes to your Git branch and submit a pull request.

Developing Quicklook Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^

<AWH to add content here.>

Testing 
-------
