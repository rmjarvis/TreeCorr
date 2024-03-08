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
.. module:: ngcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarT
from .nzcorrelation import BaseNZCorrelation
from .util import make_writer, make_reader
from .config import make_minimal_config


class NTCorrelation(BaseNZCorrelation):
    r"""This class handles the calculation and storage of a 2-point count-trefoil correlation
    function, where a trefoil is any field with spin-3 rotational properties.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:     The number of bins in logr
        bin_size:  The size of the bins in logr
        min_sep:   The minimum separation being considered
        max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:
        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xi:         The correlation function, :math:`\xi(r) = \langle t_R\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RT calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RT calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RT calculation. cf. `calculateXi`

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use `Corr2.process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> nt = treecorr.NTCorrelation(config)
        >>> nt.process(cat1,cat2)   # Compute the cross-correlation.
        >>> nt.write(file_name)     # Write out to a file.
        >>> xi = nt.xi              # Or access the correlation function directly.

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
    _cls = 'NTCorrelation'
    _letter1 = 'N'
    _letter2 = 'T'
    _letters = 'NT'
    _builder = _treecorr.NTCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarT)
    _zreal = 'tR'
    _zimag = 'tR_im'

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NTCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

    def finalize(self, vart):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vart (float):   The variance per component of the trefoil field.
        """
        super().finalize(vart)

    def calculateXi(self, *, rt=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rt is None, the simple correlation function :math:`\langle t_R\rangle` is
          returned.
        - If rt is not None, then a compensated calculation is done:
          :math:`\langle t_R\rangle = (DT - RT)`, where DT represents the mean radial trefoil
          field around the data points and RT represents the mean radial trefoil field around
          random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rt is provided).  The raw, uncompensated values
        are available as ``rawxi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rt (NTCorrelation): The cross-correlation using random locations as the lenses
                                (RT), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        return super().calculateXi(rz=rt)

    def write(self, file_name, *, rt=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
        r"""Write the correlation function to the file, file_name.

        - If rt is None, the simple correlation function :math:`\langle t_R\rangle` is used.
        - If rt is not None, then a compensated calculation is done:
          :math:`\langle t_R\rangle = (DT - RT)`, where DT represents the mean trefoil field
          around the data points and RT represents the mean trefoil field around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        tR              The mean real part of the trefoil field relative to the
                        center points.
        tR_im           The mean imaginary part of the trefoil field relative to the
                        center points.
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rt (NTCorrelation): The cross-correlation using random locations as the lenses
                                (RT), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, rt, file_type, precision, write_patch_results, write_cov)
