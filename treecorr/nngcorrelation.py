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
.. module:: nngcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .corr3base import Corr3


class GNNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-count-count correlation
    function, where as usual G represents any spin-0 scalar field.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    scalar value.  Use `NGNCorrelation` and `NNGCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> gnn = treecorr.GNNCorrelation(config)
        >>> gnn.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> gnn.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> grr.process(cat1, rand)        # Compute cross-correlation with randoms.
        >>> gdr.process(cat1, cat2, rand)  # Compute cross-correlation with randoms and data
        >>> gnn.write(file_name)           # Write out to a file.
        >>> gnn.calculateZeta(grr=grr, gdr=gdr) # Calculate zeta using randoms
        >>> zeta = gnn.zeta                # Access correlation function
        >>> zetar = gnn.zetar              # Access real and imaginary parts separately
        >>> zetai = gnn.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'GNNCorrelation'
    _letter1 = 'G'
    _letter2 = 'N'
    _letter3 = 'N'
    _letters = 'GNN'
    _builder = _treecorr.GNNCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = 'sig_sn (per component)'
    _sig2 = None
    _sig3 = None

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._grr = None
        self._gdr = None
        self._grd = None
        shape = self.data_shape
        self._z[0:2] = [np.zeros(shape, dtype=float) for _ in range(2)]
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building GNNCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._grr is not None and self._grr is not True:
            ret._grr = self._grr.copy()
        if self._gdr is not None and self._gdr is not True:
            ret._gdr = self._gdr.copy()
        if self._grd is not None and self._grd is not True:
            ret._grd = self._grd.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._grr = None
        self._krd = None
        self._gdr = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imag part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class NGNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-scalar-count correlation
    function, where as usual G represents any spin-0 scalar field.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    scalar value.  Use `GNNCorrelation` and `NNGCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> ngn = treecorr.GNNCorrelation(config)
        >>> ngn.process(cat1, cat2, cat1)  # Compute cross-correlation of two fields.
        >>> ngn.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> rgr.process(rand, cat2, rand)  # Compute cross-correlation with randoms.
        >>> dgr.process(cat1, cat2, rand)  # Compute cross-correlation with randoms and data
        >>> ngn.write(file_name)           # Write out to a file.
        >>> ngn.calculateZeta(rgr=rgr, dgr=dgr) # Calculate zeta using randoms
        >>> zeta = ngn.zeta                # Access correlation function
        >>> zetar = ngn.zetar              # Access real and imaginary parts separately
        >>> zetai = ngn.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NGNCorrelation'
    _letter1 = 'N'
    _letter2 = 'G'
    _letter3 = 'N'
    _letters = 'NGN'
    _builder = _treecorr.NGNCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = None
    _sig2 = 'sig_sn (per component)'
    _sig3 = None

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rgr = None
        self._dgr = None
        self._rgd = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building NGNCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rgr is not None and self._rgr is not True:
            ret._rgr = self._rgr.copy()
        if self._dgr is not None and self._dgr is not True:
            ret._dgr = self._dgr.copy()
        if self._rgd is not None and self._rgd is not True:
            ret._rgd = self._rgd.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._rgr = None
        self._rgd = None
        self._dgr = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imag part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class NNGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-count-scalar correlation
    function, where as usual G represents any spin-0 scalar field.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    scalar value.  Use `GNNCorrelation` and `NGNCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> nng = treecorr.NNGCorrelation(config)
        >>> nng.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> nng.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> rrg.process(rand, cat2)        # Compute cross-correlation with randoms.
        >>> drg.process(cat1, rand, cat2)  # Compute cross-correlation with randoms and data
        >>> nng.write(file_name)           # Write out to a file.
        >>> nng.calculateZeta(rrg=rrg, drg=drg) # Calculate zeta using randoms
        >>> zeta = nng.zeta                # Access correlation function
        >>> zetar = nng.zetar              # Access real and imaginary parts separately
        >>> zetai = nng.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NNGCorrelation'
    _letter1 = 'N'
    _letter2 = 'N'
    _letter3 = 'G'
    _letters = 'NNG'
    _builder = _treecorr.NNGCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = None
    _sig2 = None
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rrg = None
        self._drg = None
        self._rdg = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building NNGCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rrg is not None and self._rrg is not True:
            ret._rrg = self._rrg.copy()
        if self._drg is not None and self._drg is not True:
            ret._drg = self._drg.copy()
        if self._rdg is not None and self._rdg is not True:
            ret._rdg = self._rdg.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._rrg = None
        self._rdg = None
        self._drg = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imag part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2
