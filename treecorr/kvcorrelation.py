# Copyright (c) 2003-2019 by Mike Jarvis
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
from .catalog import calculateVarV, calculateVarK
from .kzcorrelation import BaseKZCorrelation
from .util import make_writer, make_reader
from .config import make_minimal_config


class KVCorrelation(BaseKZCorrelation):
    r"""This class handles the calculation and storage of a 2-point scalar-vector correlation
    function.

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
        xi:         The real component of the correlation function,
                    :math:`\xi(r) = \langle \kappa\, v_R\rangle`.
        xi_im:      The imaginary comonent of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into the final
        correlation.  This does not include sample variance, so it is always an underestimate of
        the actual variance.  To get better estimates, you need to set ``var_method`` to something
        else and use patches in the input catalog(s).  cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> kv = treecorr.KVCorrelation(config)
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
    _cls = 'KVCorrelation'
    _letter = 'V'
    _builder = _treecorr.KVCorr
    _calculateVarZ = staticmethod(calculateVarV)
    _xi = 'xi'
    _xi_im = 'xi_im'

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `KVCorrelation`.  See class doc for details.
        """
        BaseKZCorrelation.__init__(self, config, logger=logger, **kwargs)

    def finalize(self, vark, varv):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vark (float):   The variance of the scaler field.
            varv (float):   The variance per component of the vector field.
        """
        BaseKZCorrelation.finalize(self, vark, varv)

    def process(self, cat1, cat2, *, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True, patch_method='global'):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the K field.
            cat2 (Catalog):     A catalog or list of catalogs for the V field.
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `Corr2.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
            patch_method (str): Which patch method to use. (default: 'global')
        """
        BaseKZCorrelation.process(self, cat1, cat2, metric, num_threads, comm, low_mem,
                                  initialize, finalize, patch_method)

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
        xi              The real component of the correlation function,
                        :math:`xi(r) = \langle \kappa\, v_R\rangle`
        xi_im           The imaginary component of the correlation function.
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
        BaseKZCorrelation.write(self, file_name, file_type, precision,
                                write_patch_results, write_cov)

    @classmethod
    def from_file(cls, file_name, *, file_type=None, logger=None, rng=None):
        """Create a KVCorrelation instance from an output file.

        This should be a file that was written by TreeCorr.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII', 'FITS', or 'HDF').  (default: determine
                                the type automatically from the extension of file_name.)
            logger (Logger):    If desired, a logger object to use for logging. (default: None)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                                random number generation. (default: None)

        Returns:
            corr: A KVCorrelation object, constructed from the information in the file.
        """
        return BaseKZCorrelation.from_file(cls, file_name, file_type, logger, rng)
