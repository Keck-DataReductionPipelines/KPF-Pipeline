Pipeline
========

The ``/kpfpipe`` directory contains the source code to the pipeline. Its
structure is::

    kpfpipe
    ├── models 
    |   └── ... 
    ├── primitives 
    |   └── ... 
    ├── tools 
    |   └── ... 
    ├── tests 
    |   └── ... 
    ├── pipelines
    |   └── ...
    ├── cli.py 
    └── logger.py 

Directory ``models`` contains the implementation of KPF data models. There
are three levels of KPF data in total, and are implemented in ``level0.py``,
``level1.py``, ``level2.py`` respectively. The KPF data type inherit from
the ``Argument`` class from the Keck DRP framework.

Directory ``primitives`` contains the implementation of KPF primitives. For
each level of KPF data, there is a corresponding KPF primitive, 
and each primitive must take there data level as input. For example,
``KPF0_Primitive`` defined in ``level0.py`` takes a level 0 KPF data type 
as input. The primitive's output must also be a KPF data type. However,
the output's data level is not restricted. 

The KPF primitives inherit from the KeckDRPFramework's ``BasePrimitive``.
This means that all KPF primitives's constructors are defined as::

    from keckdrpframework.primitives.base_primitives import BasePrimitive
    from keckdrpframework.models.action import Action
    from keckdrpframework.models.processing_context import ProcessingContext

    KPF_Primitive(BasePrimitive): 

        def __init__(self, action: Action, context: ProcessingContext):



