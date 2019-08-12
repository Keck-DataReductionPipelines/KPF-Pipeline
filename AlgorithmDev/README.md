# AlgorithmDev

Python development for spectral rectification and extraction

## Usage

The AlgorithmDev is located under directory 'KPF-Pipeline'.  
The KPF-Pipeline repository is located at https://github.com/California-Planet-Search/KPF-Pipeline

- Prepare local install and developmemt 

  - python setup.py develop


- Algorithm implementation
  - AlgorithmDev/PolygonClipping.py: spectral rectification by using polygon clipping method, sum extraction, optimal extraction.

- Test files: running on Jupyter notebook 
  - examples/polygon_clipping_test.ipynb: spectral rectification for simple simulation data
  - examples/paras_test.ipynb: spectral rectification on paras data
  - examples/paras_test_sum.ipynb: sum extraction + rectification on paras data
  - examples/paras_test_optimal.ipynb: optimal extraction + rectification on paras data

