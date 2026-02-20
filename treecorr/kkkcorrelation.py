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
.. module:: kkkcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK
from .corr3base import Corr3


class KKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-scalar-scalar correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of the CMB temperature fluctuations, where "kappa" would really be
        :math:`\Delta T`.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance of :math:`\zeta`, only including the shot noise propagated into
                    the final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.

    The typical usage pattern is as follows:

        >>> kkk = treecorr.KKKCorrelation(config)
        >>> kkk.process(cat)                # Compute the auto-correlation.
        >>> # kkk.process(cat1, cat2, cat3) # ... or the cross-correlation.
        >>> kkk.write(file_name)            # Write out to a file.
        >>> zeta = kkk.zeta                 # Access zeta directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KKKCorrelation'
    _letter1 = 'K'
    _letter2 = 'K'
    _letter3 = 'K'
    _letters = 'KKK'
    _builder = _treecorr.KKKCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_k'
    _sig2 = 'sig_k'
    _sig3 = 'sig_k'
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self.logger.debug('Finished building KKKCorr')

    @property
    def zeta(self):
        if self._z[1].size:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._z[0]

    def finalize(self, vark1, vark2, vark3):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
            vark3 (float):  The variance of the third scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * vark3

        # I don't really understand why the variance is coming out 2x larger than the normal
        # formula for LogSAS.  But with just Gaussian noise, I need to multiply the numerator
        # by two to get the variance estimates to come out right.
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zetar and zetai.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
                        (if rrr is given)
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zetar', 'zetai', 'sigma_zeta']
        else:
            return ['zeta', 'sigma_zeta']

    @property
    def _write_class_data(self):
        if self.bin_type == 'LogMultipole':
            return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]
        else:
            return [self.zeta, np.sqrt(self.varzeta)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if self.bin_type == 'LogMultipole':
            self._z[0] = data['zetar'].reshape(s)
            self._z[1] = data['zetai'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = [data['sigma_zeta'].reshape(s)**2]
