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
