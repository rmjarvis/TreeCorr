# Copyright (c) 2003-2024 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: ggcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarT
from .zzcorrelation import BaseZZCorrelation
from .util import make_writer, make_reader
from .config import make_minimal_config


class TTCorrelation(BaseZZCorrelation):
    r"""This class handles the calculation and storage of a 2-point trefoil-trefoil correlation
    function, where a trefoil is any field with spin-3 rotational properties.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:      The number of bins in logr
        bin_size:   The size of the bins in logr
        min_sep:    The minimum separation being considered
        max_sep:    The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:

        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xip:        The correlation function, :math:`\xi_+(r)`.
        xim:        The correlation function, :math:`\xi_-(r)`.
        xip_im:     The imaginary part of :math:`\xi_+(r)`.
        xim_im:     The imaginary part of :math:`\xi_-(r)`.
        varxip:     An estimate of the variance of :math:`\xi_+(r)`
        varxim:     An estimate of the variance of :math:`\xi_-(r)`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix for the data vector with
                    :math:`\xi_+` first and then :math:`\xi_-`.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxip``,
        ``varxim``, and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use
        `BaseZZCorrelation.process_auto` and/or `Corr2.process_cross`, then the units will not be
        applied to ``meanr`` or ``meanlogr`` until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> tt = treecorr.TTCorrelation(config)
        >>> tt.process(cat)         # For auto-correlation.
        >>> tt.process(cat1,cat2)   # For cross-correlation.
        >>> tt.write(file_name)     # Write out to a file.
        >>> xip = tt.xip            # Or access the correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'TTCorrelation'
    _letter1 = 'T'
    _letter2 = 'T'
    _letters = 'TT'
    _builder = _treecorr.TTCorr
    _calculateVar1 = staticmethod(calculateVarT)
    _calculateVar2 = staticmethod(calculateVarT)

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `TTCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

    def finalize(self, vart1, vart2):
        """Finalize the calculation of the correlation function.

        The `BaseZZCorrelation.process_auto` and `Corr2.process_cross` commands accumulate values
        in each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            vart1 (float):  The variance per component of the first trefoil field.
            vart2 (float):  The variance per component of the second trefoil field.
        """
        super().finalize(vart1, vart2)
