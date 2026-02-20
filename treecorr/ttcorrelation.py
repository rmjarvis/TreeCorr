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
.. module:: ttcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarT
from .zzcorrelation import BaseZZCorrelation


class TTCorrelation(BaseZZCorrelation):
    r"""This class handles the calculation and storage of a 2-point trefoil-trefoil correlation
    function, where a trefoil is any field with spin-3 rotational properties.

    See the doc string of `Corr2` for a description of how the pairs are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr2` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        xip:        The correlation function, :math:`\xi_+(r)`.
        xim:        The correlation function, :math:`\xi_-(r)`.
        xip_im:     The imaginary part of :math:`\xi_+(r)`.
        xim_im:     The imaginary part of :math:`\xi_-(r)`.
        varxip:     An estimate of the variance of :math:`\xi_+(r)`
        varxim:     An estimate of the variance of :math:`\xi_-(r)`
        cov:        An estimate of the full covariance matrix for the data vector with
                    :math:`\xi_+` first and then :math:`\xi_-`.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxip``,
        ``varxim``, and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    The typical usage pattern is as follows:

        >>> tt = treecorr.TTCorrelation(config)
        >>> tt.process(cat)         # For auto-correlation.
        >>> tt.process(cat1,cat2)   # For cross-correlation.
        >>> tt.write(file_name)     # Write out to a file.
        >>> xip = tt.xip            # Or access the correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
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

    def finalize(self, vart1, vart2):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_auto` and `Corr2.process_cross` commands accumulate values
        in each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            vart1 (float):  The variance per component of the first trefoil field.
            vart2 (float):  The variance per component of the second trefoil field.
        """
        super().finalize(vart1, vart2)
