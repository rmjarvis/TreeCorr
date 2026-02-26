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
.. module:: kgcorrelation
"""


from . import _treecorr
from .catalog import calculateVarG, calculateVarK
from .kzcorrelation import BaseKZCorrelation


class KGCorrelation(BaseKZCorrelation):
    r"""This class handles the calculation and storage of a 2-point scalar-shear correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of some survey property, such as seeing, with shear, where "kappa" would
        really be the measured property, e.g. the observed sizes of the stars.

    See the docstring of `Corr2` for a description of how the pairs are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr2` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        xi:         The correlation function, :math:`\xi(r) = \langle \kappa\, \gamma_T\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        cov:        An estimate of the full covariance matrix.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into the final
        correlation.  This does not include sample variance, so it is always an underestimate of
        the actual variance.  To get better estimates, you need to set ``var_method`` to something
        else and use patches in the input catalog(s).  cf. `Covariance Estimates`.

    The typical usage pattern is as follows:

        >>> kg = treecorr.KGCorrelation(config)
        >>> kg.process(cat1, cat2)         # Compute the cross-correlation.
        >>> kg.write(file_name)            # Write out to a file.
        >>> xi, xi_im = kg.xi, kg.xi_im    # Or access the correlation function directly.

    See also: `KKCorrelation`, `NGCorrelation`, `KGGCorrelation`.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
                        in `Corr2`, which are ignored here. (default: None)
        logger (:class:`logging.Logger`):
                        If desired, a ``Logger`` object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KGCorrelation'
    _letter1 = 'K'
    _letter2 = 'G'
    _letters = 'KG'
    _builder = _treecorr.KGCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarG)
    _xireal = 'kgamT'
    _xiimag = 'kgamX'

    def finalize(self, vark, varg):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vark (float):   The variance of the scaler field.
            varg (float):   The variance per component of the shear field.
        """
        super().finalize(vark, varg)

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        r"""Write the correlation function to the file, file_name.

        The output file will include the following columns:

        ==========      ========================================================
        Column          Description
        ==========      ========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r\rangle` of pairs that
                        fell into each bin
        meanlogr        The mean value :math:`\langle \log(r)\rangle` of pairs
                        that fell into each bin
        kgamT           The real part of the correlation function,
                        :math:`\langle \kappa\, \gamma_T\rangle`
        kgamX           The imaginary part of the correlation function.
        sigma           The sqrt of the variance estimate of both of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      ========================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output files, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, file_type, precision, write_patch_results, write_cov)
