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
.. module:: kvcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarZ, calculateVarK
from .corr2base import Corr2
from .util import make_writer


class BaseKZCorrelation(Corr2):
    """This class is a base class for all the K?Correlation classes, where ? is one of the
    complex fields of varying spin.

    A lot of the implementation is shared among those types, so whenever possible the shared
    implementation is done in this class.
    """
    _sig1 = 'sig_k'
    _sig2 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._xi[0] = np.zeros_like(self.rnom, dtype=float)
        self._xi[1] = np.zeros_like(self.rnom, dtype=float)
        self.logger.debug('Finished building %s', self._cls)

    @property
    def xi(self):
        return self._xi[0]

    @property
    def xi_im(self):
        return self._xi[1]

    def finalize(self, vark, varz):
        self._finalize()
        self._var_num = vark * varz

    @property
    def varxi(self):
        if self._varxi is None:
            self._calculate_varxi(1)
        return self._varxi[0]

    def write(self, file_name, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        self.logger.info(f'Writing {self._letters} correlations to %s', file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    @property
    def _write_col_names(self):
        return ['r_nom','meanr','meanlogr',self._xireal,self._xiimag,'sigma','weight','npairs']

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr,
                 self.xi, self.xi_im, np.sqrt(self.varxi),
                 self.weight, self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.logr.shape
        self._xi[0] = data[self._xireal].reshape(s)
        self._xi[1] = data[self._xiimag].reshape(s)
        self._varxi = [data['sigma'].reshape(s)**2]


class KZCorrelation(BaseKZCorrelation):
    r"""This class handles the calculation and storage of a 2-point scalar-complex correlation
    function, where the complex field is taken to have spin-0 rotational properties.  If the
    spin-0 field is real, you should instead use `KKCorrelation` as it will be faster.
    This class is intended for correlations of a scalar field with a complex values that
    don't change with orientation.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:
        xi:         The correlation function, :math:`\xi(r) = \langle \kappa\, z\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of each component of :math:`\xi`
        cov:        An estimate of the full covariance matrix.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into the final
        correlation.  This does not include sample variance, so it is always an underestimate of
        the actual variance.  To get better estimates, you need to set ``var_method`` to something
        else and use patches in the input catalog(s).  cf. `Covariance Estimates`.

    The typical usage pattern is as follows:

        >>> kv = treecorr.KZCorrelation(config)
        >>> kv.process(cat1,cat2)   # Calculate the cross-correlation
        >>> kv.write(file_name)     # Write out to a file.
        >>> xi = kv.xi              # Or access the correlation function directly.

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
    _cls = 'KZCorrelation'
    _letter1 = 'K'
    _letter2 = 'Z'
    _letters = 'KZ'
    _builder = _treecorr.KZCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarZ)
    _xireal = 'xi'
    _xiimag = 'xi_im'

    def finalize(self, vark, varz):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vark (float):   The variance of the scaler field.
            varz (float):   The variance per component of the complex field.
        """
        super().finalize(vark, varz)

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
        xi              The real part of correlation function,
                        :math:`xi(r) = \langle \kappa\, z\rangle`
        xi_im           The imaginary part of correlation function.
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
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, file_type, precision, write_patch_results, write_cov)
