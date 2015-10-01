# Copyright (c) 2003-2015 by Mike Jarvis
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
.. module:: nncorrelation
"""

import treecorr
import numpy

# Start by loading up the relevant C functions using ctypes
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))

# some useful aliases
cint = ctypes.c_int
cdouble = ctypes.c_double
cdouble_ptr = ctypes.POINTER(cdouble)
cvoid_ptr = ctypes.c_void_p

_treecorr.BuildNNCorr.restype = cvoid_ptr
_treecorr.BuildNNCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyNNCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNN3D.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNN3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNN3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NNCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    Ojects of this class holds the following attributes:

        :nbins:     The number of bins in logr
        :bin_size:  The size of the bins in logr
        :min_sep:   The minimum separation being considered
        :max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

        :logr:      The nominal center of the bin in log(r) (the natural logarithm of r).
        :meanr:     The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        :meanlogr:  The mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs in each bin.
        :tot:       The total number of pairs processed, which is used to normalize
                    the randoms if they have a different number of pairs.

    If sep_units are given (either in the config dict or as a named kwarg) then logr and meanlogr
    both take r to be in these units.  i.e. exp(logr) will have R in units of sep_units.

    The usage pattern is as follows:

        >>> nn = treecorr.NNCorrelation(config)
        >>> nn.process(cat)         # For auto-correlation.
        >>> nn.process(cat1,cat2)   # For cross-correlation.
        >>> rr.process...           # Likewise for random-random correlations
        >>> dr.process...           # If desired, also do data-random correlations
        >>> rd.process...           # For cross-correlations, also do the reverse.
        >>> nn.write(file_name,rr,dr,rd)         # Write out to a file.
        >>> xi,varxi = nn.calculateXi(rr,dr,rd)  # Or get the correlation function directly.

    :param config:      A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries in addition to those listed
                        in :class:`~treecorr.BinnedCorr2`, which are ignored here. (default: None)
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    See the documentation for :class:`~treecorr.BinnedCorr2` for the list of other allowed kwargs,
    which may be passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.meanr = numpy.zeros(self.nbins, dtype=float)
        self.meanlogr = numpy.zeros(self.nbins, dtype=float)
        self.weight = numpy.zeros(self.nbins, dtype=float)
        self.npairs = numpy.zeros(self.nbins, dtype=float)
        self.tot = 0.
        self._build_corr()
        self.logger.debug('Finished building NNCorr')

    def _build_corr(self):
        meanr = self.meanr.ctypes.data_as(cdouble_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)
        self.corr = _treecorr.BuildNNCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          meanr,meanlogr,weight,npairs);

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            _treecorr.DestroyNNCorr(self.corr)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['corr']
        del d['logger']  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._build_corr()
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,0),
                self.config.get('log_file',None))

    def __repr__(self):
        return 'NNCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanr, meanlogr.

        :param cat:         The catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat.name == '':
            self.logger.info('Starting process NN auto-correlations')
        else:
            self.logger.info('Starting process NN auto-correlations for cat %s.', cat.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")

        self._set_num_threads(num_threads)

        field = cat.getNField(self.min_sep,self.max_sep,self.b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        if field.flat:
            _treecorr.ProcessAutoNNFlat(self.corr, field.data, self.output_dots)
        elif field.perp:
            _treecorr.ProcessAutoNNPerp(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNN3D(self.corr, field.data, self.output_dots)
        self.tot += 0.5 * cat.sumw**2


    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanr, meanlogr.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NN cross-correlations')
        else:
            self.logger.info('Starting process NN cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        self._set_num_threads(num_threads)

        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method,metric,self.max_top)
        f2 = cat2.getNField(self.min_sep,self.max_sep,self.b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        if f1.flat:
            _treecorr.ProcessCrossNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessCrossNNPerp(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNN3D(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += cat1.sumw*cat2.sumw


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the sums into the bins, but does not finalize the calculation.
        After calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NN pairwise-correlations')
        else:
            self.logger.info('Starting process NN pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField(metric)
        f2 = cat2.getNSimpleField(metric)

        if f1.flat:
            _treecorr.ProcessPairwiseNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessPairwiseNNPerp(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseNN3D(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += cat1.weight


    def finalize(self):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanr, meanlogr by dividing by the total weight.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self.meanr[mask1] /= self.sep_units
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = numpy.exp(self.logr[mask2])
        self.meanlogr[mask2] = self.logr[mask2]


    def clear(self):
        """Clear the data vectors
        """
        self.meanr[:] = 0.
        self.meanlogr[:] = 0.
        self.weight[:] = 0.
        self.npairs[:] = 0.
        self.tot = 0.

    def __iadd__(self, other):
        """Add a second NNCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        process_auto and/or process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, NNCorrelation):
            raise AttributeError("Can only add another NNCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NNCorrelation to be added is not compatible with this one.")

        self.meanr[:] += other.meanr[:]
        self.meanlogr[:] += other.meanlogr[:]
        self.weight[:] += other.weight[:]
        self.npairs[:] += other.npairs[:]
        self.tot += other.tot
        return self


    def process(self, cat1, cat2=None, metric=None, num_threads=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:        A catalog or list of catalogs for the first N field.
        :param cat2:        A catalog or list of catalogs for the second N field, if any.
                            (default: None)
        :param metric:      Which metric to use for distance measurements.  Options are:

                            - 'Euclidean' = straight line Euclidean distance between two points.
                              For spherical coordinates (ra,dec without r), this is the chord
                              distance between points on the unit sphere.
                            - 'Rperp' = the perpendicular component of the distance. For two points
                              with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                              distance and :math:`Rparallel = |r1-r2|`, then we define
                              :math:`Rperp^2 = d^2 - Rparallel^2`.

                            (default: 'Euclidean'; this value can also be given in the constructor
                            in the config dict.)

        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            self._process_all_auto(cat1,metric,num_threads)
        else:
            self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize()


    def calculateXi(self, rr, dr=None, rd=None):
        """Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        The rr value is the NNCorrelation function for random points.
        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as dr and rd.

        - If dr is None, the simple correlation function :math:`\\xi = (DD/RR - 1)` is used.
        - if dr is given and rd is None, then :math:`\\xi = (DD - 2DR + RR)/RR` is used.
        - If dr and rd are both given, then :math:`\\xi = (DD - DR - RD + RR)/RR` is used.

        where DD is the data NN correlation function, which is the current object.

        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
                        
        :returns:           (xi, varxi) as a tuple
        """
        # Each random weight value needs to be rescaled by the ratio of total possible pairs.
        if rr.tot == 0:
            raise RuntimeError("rr has tot=0.")

        rrw = self.tot / rr.tot
        if dr is None:
            if rd is None:
                xi = (self.weight - rr.weight * rrw)
            else:
                if rd.tot == 0:
                    raise RuntimeError("rd has tot=0.")
                rdw = self.tot / rd.tot
                xi = (self.weight - 2.*rd.weight * rdw + rr.weight * rrw)
        else:
            if dr.tot == 0:
                raise RuntimeError("dr has tot=0.")
            drw = self.tot / dr.tot
            if rd is None:
                xi = (self.weight - 2.*dr.weight * drw + rr.weight * rrw)
            else:
                if rd.tot == 0:
                    raise RuntimeError("rd has tot=0.")
                rdw = self.tot / rd.tot
                xi = (self.weight - rd.weight * rdw - dr.weight * drw + rr.weight * rrw)
        if numpy.any(rr.weight == 0):
            self.logger.warn("Warning: Some bins for the randoms had no pairs.")
            self.logger.warn("         Probably max_sep is larger than your field.")
        mask1 = rr.weight != 0
        mask2 = rr.weight == 0
        xi[mask1] /= (rr.weight[mask1] * rrw)
        xi[mask2] = 0

        varxi = numpy.zeros_like(rr.weight)
        varxi[mask1] = 1./ (rr.weight[mask1] * rrw)

        return xi, varxi


    def write(self, file_name, rr=None, dr=None, rd=None, file_type=None, prec=None):
        """Write the correlation function to the file, file_name.

        rr is the NNCorrelation function for random points.
        If dr is None, the simple correlation function :math:`\\xi = (DD - RR)/RR` is used.
        if dr is given and rd is None, then :math:`\\xi = (DD - 2DR + RR)/RR` is used.
        If dr and rd are both given, then :math:`\\xi = (DD - DR - RD + RR)/RR` is used.

        Normally, at least rr should be provided, but if this is also None, then only the 
        basic accumulated number of pairs are output (along with the separation columns).

        The output file will include the following columns:

            :R_nom:     The nominal center of the bin in R.
            :meanR:     The mean value :math:`\\langle R\\rangle` of pairs that fell into each bin.
            :meanlogR:  The mean value :math:`\\langle logR\\rangle` of pairs that fell into each
                        bin.

        Then if rr is None:

            :DD:        The total weight of pairs in each bin.
            :npairs:    The total number of pairs in each bin.

        If rr is given, but not the cross-correlations:

            :xi:        The estimator :math:`\\xi = (DD-RR)/RR`.
            :sigma_xi:  The sqrt of the variance estimate of :math:`\\xi`.
            :DD:        The total weight of data pairs (aka DD) in each bin.
            :RR:        The total weight of random pairs (aka RR) in each bin.
            :npairs:    The number of pairs contributing ot each bin.

        If one of dr or rd is given:

            :xi:        The estimator :math:`\\xi = (DD-2DR+RR)/RR`.
            :sigma_xi:  The sqrt of the variance estimate of :math:`\\xi`.
            :DD:        The total weight of DD pairs in each bin.
            :RR:        The total weight of RR pairs in each bin.
            :DR:        The total weight of DR pairs in each bin.
            :npairs:    The number of pairs contributing ot each bin.

        If both dr and rd are given:

            :xi:        The estimator :math:`\\xi = (DD-DR-RD+RR)/RR`.
            :sigma_xi:  The sqrt of the variance estimate of :math:`\\xi`.
            :DD:        The total weight of DD pairs in each bin.
            :RR:        The total weight of RR pairs in each bin.
            :DR:        The total weight of DR pairs in each bin.
            :RD:        The total weight of RD pairs in each bin.
            :npairs:    The number of pairs contributing ot each bin.


        :param file_name:   The name of the file to write to.
        :param rr:          An NNCorrelation object for the random-random pairs. (default: None,
                            in which case, no xi or varxi columns will be output)
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param prec:        For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NN correlations to %s',file_name)
        
        col_names = [ 'R_nom','meanR','meanlogR' ]
        columns = [ numpy.exp(self.logr), self.meanr, self.meanlogr ]
        if rr is None:
            col_names += [ 'DD', 'npairs' ]
            columns += [ self.weight, self.npairs ]
            if dr is not None:
                raise AttributeError("rr must be provided if dr is not None")
            if rd is not None:
                raise AttributeError("rr must be provided if rd is not None")
        else:
            xi, varxi = self.calculateXi(rr,dr,rd)

            col_names += [ 'xi','sigma_xi','DD','RR' ]
            columns += [ xi, numpy.sqrt(varxi),
                         self.weight, rr.weight * (self.tot/rr.tot) ]

            if dr is not None and rd is not None:
                col_names += ['DR','RD']
                columns += [ dr.weight * (self.tot/dr.tot), rd.weight * (self.tot/rd.tot) ]
            elif dr is not None or rd is not None:
                if dr is None: dr = rd
                col_names += ['DR']
                columns += [ dr.weight * (self.tot/dr.tot) ]
            col_names += [ 'npairs' ]
            columns += [ self.npairs ]

        if prec is None:
            prec = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name, col_names, columns, prec=prec, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NNCorrelation object should be constructed with the same configuration 
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading NN correlations from %s',file_name)

        data = treecorr.util.gen_read(file_name, file_type=file_type)
        self.logr = numpy.log(data['R_nom'])
        self.meanr = data['meanR']
        self.meanlogr = data['meanlogR']
        self.weight = data['DD']
        self.npairs = data['npairs']


    def calculateNapSq(self, rr, dr=None, rd=None, m2_uform=None):
        """Calculate the correlary to the aperture mass statistics for counts.

        This is used by NGCorrelation.writeNorm.  See that function and also 
        GGCorrelation.calculateMapSq() for more details.

        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
        :param m2_uform:    Which form to use for the aperture mass.  (default: 'Crittenden';
                            this value can also be given in the constructor in the config dict.)

        :returns: (nsq, varnsq)
        """
        if m2_uform is None:
            m2_uform = treecorr.config.get(self.config,'m2_uform',str,'Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        r = numpy.exp(self.logr)
        s = numpy.outer(1./r, self.meanr)  
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = numpy.exp(-ssq/4.)
            Tp = (32. + ssq*(-16. + ssq)) / 128. * exp_factor
        else:
            Tp = numpy.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*numpy.pi) * (2.-15.*ssqa) * numpy.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*numpy.pi) * sa * numpy.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
        Tp *= ssq

        xi, varxi = self.calculateXi(rr,dr,rd)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxi = Tp.dot(xi)
        nsq = Tpxi * self.bin_size
        varnsq = (Tp**2).dot(varxi) * self.bin_size**2

        return nsq, varnsq


