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
.. module:: nqcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarQ
from .nzcorrelation import BaseNZCorrelation


class NQCorrelation(BaseNZCorrelation):
    r"""This class handles the calculation and storage of a 2-point count-quatrefoil correlation
    function, where a quatrefoil is any field with spin-4 rotational properties.

    See the docstring of `Corr2` for a description of how the pairs are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr2` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        xi:         The correlation function, :math:`\xi(r) = \langle q_R\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RQ calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RQ calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RQ calculation. cf. `calculateXi`

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    The typical usage pattern is as follows:

        >>> nq = treecorr.NQCorrelation(config)
        >>> nq.process(cat1, cat2)         # Compute the cross-correlation.
        >>> nq.write(file_name)            # Write out to a file.
        >>> xi, xi_im = nq.xi, nq.xi_im    # Or access the correlation function directly.

    See also: `KQCorrelation`, `QQCorrelation`, `NZCorrelation`.

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
    _cls = 'NQCorrelation'
    _letter1 = 'N'
    _letter2 = 'Q'
    _letters = 'NQ'
    _builder = _treecorr.NQCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarQ)
    _zreal = 'qR'
    _zimag = 'qR_im'

    def finalize(self, varq):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            varq (float):   The variance per component of the quatrefoil field.
        """
        super().finalize(varq)

    def calculateXi(self, *, rq=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rq is None, the simple correlation function :math:`\langle q_R\rangle` is
          returned.
        - If rq is not None, then a compensated calculation is done:
          :math:`\langle q_R\rangle = (DQ - RQ)`, where DQ represents the mean radial quatrefoil
          field around the data points and RQ represents the mean radial quatrefoil field around
          random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rq is provided).  The raw, uncompensated values
        are available as ``raw_xi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rq (NQCorrelation): The cross-correlation using random locations as the lenses
                                (RQ), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        return super().calculateXi(rz=rq)

    def write(self, file_name, *, rq=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
        r"""Write the correlation function to the file, file_name.

        - If rq is None, the simple correlation function :math:`\langle q_R\rangle` is used.
        - If rq is not None, then a compensated calculation is done:
          :math:`\langle q_R\rangle = (DQ - RQ)`, where DQ represents the mean quatrefoil field.
          around the data points and RQ represents the mean quatrefoil field around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        qR              The mean real part of the quatrefoil field relative to the
                        center points
        qR_im           The mean imaginary part of the quatrefoil field relative to the
                        center points
        sigma           The sqrt of the variance estimate of each of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rq (NQCorrelation): The cross-correlation using random locations as the lenses
                                (RQ), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output files, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, rq, file_type, precision, write_patch_results, write_cov)
