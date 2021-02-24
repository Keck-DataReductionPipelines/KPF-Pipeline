KPF Pipeline Logging
====================

In the :doc:`example_trivial`, the log messages we get from the terminal as running the simple recipe are mainly from the following three sources:

1. logging starting with a time stamp and ``DRPF:INFO:`` (e.g. ``2021-02-03 18:23:17:DRPF:INFO:``) is from ``keckdrpframework`` package, one of the requirements of KPF-Pipe. Please refer to `KeckDRPFramework <https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_ for Framework related detail. 
2. logging starting with ``[KPF-Pipe][INFO]:`` is from KPF Pipeline.
3. logging starting with ``[KPFModExample]`` is from module 'KPFModExample'.

 

