
Three-point Correlation Functions
=================================

There are currently 3 differenct classes for calculating the different possible three-point
auto-correlation functions:

.. toctree::

    nnn
    ggg
    kkk

.. note::

    There are classes that can handle cross-correlations of the same type:

    * `treecorr.NNNCrossCorrelation`
    * `treecorr.GGGCrossCorrelation`
    * `treecorr.KKKCrossCorrelation`

    However, we do not yet have the ability to compute 3-point cross-correlations across
    different types (such as NNG or KGG, etc.)

Each of the above classes is a sub-class of the base class BinnedCorr3, so they have a number of
features in common about how they are constructed.  The common features are documented here.

.. autoclass:: treecorr.BinnedCorr3
    :members:



