# AlgorithmDev

Python development for spectral rectification and extraction

## Usage

The AlgorithmDev is located under directory 'KPF-Pipeline'.  
The KPF-Pipeline repository is located at https://github.com/California-Planet-Search/KPF-Pipeline

- Prepare local install and developmemt 

  - python setup.py develop


- Algorithm implementation
  - AlgorithmDev/PolygonClipping.py: spectral rectification by using polygon clipping method, sum extraction, optimal extraction.
  - AlgorithmDev/radial_velocity.py: finding radial velocity of the star by using cross-correlation calculation.

- Test files: running on Jupyter notebook 
  - examples/polygon_clipping_test.ipynb: spectral rectification for simple simulation data
  - examples/paras_test.ipynb: spectral rectification on paras data
  - examples/paras_test_sum.ipynb: sum extraction + rectification on paras data
  - examples/paras_test_optimal.ipynb: optimal extraction + rectification on paras data
  - exmaples/radial_velocity_test.ipynb: radial velocity calculation and output results to fits
  - examples/radial_velocity_outputfits_test.ipynb: compare radial velocity output fits among diffent versions
  - examples/radial_velocity_std_test.ipynb: standard deviation comparison among different versions
  - examples/radial_velocity_test_neid.ipynb: rv test for NEID data
  - examples/order_trace_width_test_neid.ipynb: order trace test
  - examples/paras_test_optimal_trace_order_2.ipynb: optimal extraction and sum fraction extraction test on data from OrderTrace module by using PolygonClipping2. PolygonClipping2 is refactored from PolygonClipping. 

- The main test samples from L0 data to produce L1 and L2 data
  - examples/order_trace_width_test_neid.ipynb: to get order trace description from OrderTrace 
  - examples/paras_test_optimal_trace_order_2.ipynb: to make optimal extraction or fraction summation extraction from PolygonClipping2
  - examples/radial_velocity_test_neid.ipynb: to calculate radial velocity from RadialVelocity


