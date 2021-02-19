KPF Pipeline Logging
====================

In the :doc:`example_trivial`, the log messages we get from the terminal as running the simple recipe are mainly from the following 3 channels:

- logging starting with a time stamp and ``DRPF:INFO:`` (e.g. ``2021-02-03 18:23:17:DRPF:INFO:``) is from ``keckdrpframework`` package, one of the requirements of KPF-Pipe. Please refer to `KeckDRPFramework <https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_ for Framework related detail. 
- logging starting with ``[KPF-Pipe][INFO]:`` is from KPF Pipeline.
- logging starting with ``[KPFModExample]`` is from module 'KPFModExample'.

 

