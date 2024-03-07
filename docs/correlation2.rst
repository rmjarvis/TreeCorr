
Two-point Correlation Functions
===============================

There are a number of differenct classes for calculating the different possible two-point
correlation functions:

.. toctree::

    nn
    nk
    kk
    nz
    kz
    zz
    nv
    kv
    vv
    ng
    kg
    gg
    nt
    kt
    tt
    nq
    kq
    qq

Each of the above classes is a sub-class of the base class Corr2, so they have a number of
features in common about how they are constructed.  The common features are documented here.

.. autoclass:: treecorr.Corr2
    :members:


.. autofunction:: treecorr.estimate_multi_cov

.. autofunction:: treecorr.build_multi_cov_design_matrix

.. autofunction:: treecorr.set_max_omp_threads

.. autofunction:: treecorr.set_omp_threads

.. autofunction:: treecorr.get_omp_threads
