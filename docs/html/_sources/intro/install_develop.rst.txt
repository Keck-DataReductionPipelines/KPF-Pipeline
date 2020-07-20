Installing from Develop or Feature Branches
===========================================

The ``master`` branch is considered the most stable (i.e less likely to
run into bug), but it may not be the most up-to-date. As the pipeline is
developed, you may want to install the pipeline from ``develop`` branch
or some feature branches intead. Here is a installation guide from
installing from these unstable branches so that it does not mess up
your python setup.

As before, clone the repository and navigate into it::

    git clone https://github.com/California-Planet-Search/KPF-Pipeline.git
    cd KPF-Pipeline

To switch to the ``develop`` branch, use ``git checout``::

    git checkout develop

Since the ``develop`` branch is not considered stable, create and activate
a virtual enviroment for it using ``venv``. See `Python venv documentation
<https://docs.python.org/3/library/venv.html>`_ for details::

    python -m venv venv
    source venv/bin/activate

Now your repository is contained in an isolated enviroment, so in the case that
the branch blows up in a disaster, the rest of your computer will be safe.
You can now install the pipeline package with::

    make

