# Copyright (c) 2003-2014 by Mike Jarvis
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

_treecorr.BuildNGCorr.restype = cvoid_ptr
_treecorr.BuildNGCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyNGCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessCrossNGSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNGSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NGCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-shear correlation
    function.  This is the tangential shear profile around lenses, commonly referred to as
    galaxy-galaxy lensing.

    It holds the following attributes:

        :logr:      The nominal center of the bin in log(r).
        :meanlogr:  The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        :xi:        The correlation function, xi(r) = <gamma_T>.
        :xi_im:     The imaginary part of xi(r).
        :varxi:     The variance of xi, only including the shape noise propagated into the
                    final correlation.  This does not include sample variance, so it is
                    always an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs going into each bin.

    The usage pattern is as follows:

        >>> ng = treecorr.NGCorrelation(config)
        >>> ng.process(cat1,cat2)   # Compute the cross-correlation.
        >>> ng.write(file_name)     # Write out to a file.
        >>> xi = gg.xi              # Or access the correlation function directly.

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.xi = numpy.zeros(self.nbins, dtype=float)
        self.xi_im = numpy.zeros(self.nbins, dtype=float)

        xi = self.xi.ctypes.data_as(cdouble_ptr)
        xi_im = self.xi_im.ctypes.data_as(cdouble_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)

        self.corr = _treecorr.BuildNGCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          xi,xi_im,meanlogr,weight,npairs);
        self.logger.debug('Finished building NGCorr')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyNGCorr(self.corr)


    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        self.logger.info('Starting process NG cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method)
        f2 = cat2.getGField(self.min_sep,self.max_sep,self.b,self.split_method)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossNGSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNGFlat(self.corr, f1.data, f2.data, self.output_dots)


    def process_pairwise(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        self.logger.info('Starting process NG pairwise-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getNSimpleField()
        f2 = cat2.getGSimpleField()

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessPairwiseNGSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseNGFlat(self.corr, f1.data, f2.data, self.output_dots)


    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The process_cross command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        :param varg:    The shear variance per component for the second field.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xi[mask1] /= self.weight[mask1]
        self.xi_im[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg / self.npairs[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xi[:] = 0
        self.xi_im[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0


    def process(self, cat1, cat2):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the N field.
        :param cat2:    A catalog or list of catalogs for the G field.
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")
        if len(cat2) == 0:
            raise ValueError("No catalogs provided for cat2")

        varg = treecorr.calculateVarG(cat2)
        self.logger.info("varg = %f: sig_sn (per component) = %f",varg,math.sqrt(varg))
        self._process_all_cross(cat1,cat2)
        self.finalize(varg)


    def calculateXi(self, rg=None):
        """Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        If rg is None, the simple correlation function <gamma_T> is returned.
        If rg is not None, then a compensated calculation is done: <gamma_T> = (dg - rg)

        :param rg:          An NGCorrelation using random locations as the lenses, if desired. 
                            (default: None)

        :returns:           (xi, xi_im, varxi) as a tuple.
        """
        if rg is None:
            return self.xi, self.xi_im, self.varxi
        else:
            return (self.xi - rg.xi), (self.xi_im - rg.xi_im), (self.varxi + rg.varxi)


    def write(self, file_name, rg=None):
        """Write the correlation function to the file, file_name.

        If rg is None, the simple correlation function <gamma_T> is used.
        If rg is not None, then a compensated calculation is done: <gamma_T> = (dg - rg)

        :param file_name:   The name of the file to write to.
        :param rg:          An NGCorrelation using random locations as the lenses, if desired. 
                            (default: None)
        """
        self.logger.info('Writing NG correlations to %s',file_name)
    
        xi, xi_im, varxi = self.calculateXi(rg)
         
        self.gen_write(
            file_name,
            ['R_nom','<R>','<gamT>','<gamX>','sigma','weight','npairs'],
            [ numpy.exp(self.logr), numpy.exp(self.meanlogr),
              xi, xi_im, numpy.sqrt(varxi), self.weight, self.npairs ] )


    def calculateNMap(self, rg=None, m2_uform=None):
        """Calculate the aperture mass statistics from the correlation function.

        .. math::

            \\langle N M_{ap} \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{R^2} 
            T_\\times\\left(\\frac{r}{R}\\right) \\Re\\xi(r) \\\\
            \\langle N M_{\\times} \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{R^2}
            T_\\times\\left(\\frac{r}{R}\\right) \\Im\\xi(r)

        The m2_uform parameter sets which definition of the aperture mass to use.
        The default is to look in the config dict that was used to build the catalog,
        or use 'Crittenden' if it is not specified.

        If m2_uform == 'Crittenden':

        .. math::

            T_\\times(s) = \\frac{s^2}{128} (12-s^2) \\exp(-s^2/4)

        If m2_uform == 'Schneider':

        .. math::

            T_\\times(s) = \\frac{18}{\\pi} s^2 \\arccos(s/2) -
            \\frac{3}{40\\pi} s^3 \\sqrt{4-s^2} (196 - 74s^2 + 14s^4 - s^6)

        cf. Schneider, et al (2001): http://xxx.lanl.gov/abs/astro-ph/0112441
        These formulae are not in there, but the derivation is similar to the derivations
        of T+ and T- in that paper.

        :param rg:          An NGCorrelation using random locations as the lenses, if desired. 
                            (default: None)
        :param m2_uform:    Which form to use for the aperture mass.  (default: None, in which
                            case it looks in the object's config file for config['mu_uform'],
                            or 'Crittenden' if it is not provided.)

        :returns:           (nmap, nmx, varnmap) as a tuple
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform','Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        r = numpy.exp(self.logr)
        meanr = numpy.exp(self.meanlogr) # Use the actual mean r for each bin
        s = numpy.outer(1./r, meanr)  
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = numpy.exp(-ssq/4.)
            Tx = ssq * (12. - ssq) / 128. * exp_factor
        else:
            Tx = numpy.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tx[s<2.] = 18./numpy.pi * ssqa * numpy.arccos(sa/2.)
        Tx *= ssq

        xi, xi_im, varxi = self.calculateXi(rg)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Txxi = Tx.dot(xi)
        Txxi_im = Tx.dot(xi_im)
        nmap = Txxi * self.bin_size
        nmx = Txxi_im * self.bin_size

        # The variance of each of these is 
        # Var(<NMap>(R)) = int_r=0..2R [s^4 dlogr^2 Tx(s)^2 Var(xi)]
        varnmap = (Tx**2).dot(varxi) * self.bin_size**2

        return nmap, nmx, varnmap


    def writeNMap(self, file_name, rg=None, m2_uform=None):
        """Write the cross correlation of the foreground galaxy counts with the aperture mass
        based on the correlation function to the file, file_name.

        if rg is provided, the compensated calculation will be used for xi.

        See calculateNMap for an explanation of the m2_uform parameter.

        :param file_name:   The name of the file to write to.
        :param rg:          An NGCorrelation using random locations as the lenses, if desired. 
                            (default: None)
        :param m2_uform:    Which form to use for the aperture mass.  (default: None)
        """
        self.logger.info('Writing NMap from NG correlations to %s',file_name)

        nmap, nmx, varnmap = self.calculateNMap(rg=rg, m2_uform=m2_uform)
 
        self.gen_write(
            file_name,
            ['R','<NMap>','<NMx>','sig_nmap'],
            [ numpy.exp(self.logr), nmap, nmx, numpy.sqrt(varnmap) ] )


    def writeNorm(self, file_name, gg, dd, rr, dr=None, rg=None, m2_uform=None):
        """Write the normalized aperture mass cross-correlation to the file, file_name.

        The combination :math:`\\langle N M_{ap}\\rangle^2 / \\langle M_{ap}^2\\rangle
        \\langle N_{ap}^2\\rangle` is related to :math:`r`, the galaxy-mass correlation 
        coefficient.  Similarly, :math:`\\langle N_{ap}^2\\rangle / \\langle M_{ap}^2\\rangle`
        is related to :math:`b`, the galaxy bias parameter.  cf. Hoekstra et al, 2002: 
        http://adsabs.harvard.edu/abs/2002ApJ...577..604H

        This function computes these combinations and outputs them to a file.

        if rg is provided, the compensated calculation will be used for NMap.
        if dr is provided, the compensated calculation will be used for Nap^2.

        See calculateNMap for an explanation of the m2_uform parameter.

        :param file_name:   The name of the file to write to.
        :param gg:          A GGCorrelation object for the shear-shear correlation function
                            of the G field.
        :param dd:          An NNCorrelation object for the count-count correlation function
                            of the N field.
        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rg:          An NGCorrelation using random locations as the lenses, if desired. 
                            (default: None)
        :param m2_uform:    Which form to use for the aperture mass.  (default: None)
        """
        self.logger.info('Writing Norm from NG correlations to %s',file_name)

        nmap, nmx, varnmap = self.calculateNMap(rg=rg, m2_uform=m2_uform)
        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(m2_uform=m2_uform)
        nsq, varnsq = dd.calculateNapSq(rr, dr=dr, m2_uform=m2_uform)

        nmnorm = nmap**2 / (nsq * mapsq)
        varnmnorm = nmnorm**2 * (4. * varnmap / nmap**2 + varnsq / nsq**2 + varmapsq / mapsq**2)
        nnnorm = nsq / mapsq
        varnnnorm = nnnorm**2 * (varnsq / nsq**2 + varmapsq / mapsq**2)
 
        self.gen_write(
            file_name,
            [ 'R',
              '<NMap>','<NMx>','sig_nmap',
              '<Nap^2>','sig_napsq','<Map^2>','sig_mapsq',
              'NMap_norm','sig_norm','N^2/Map^2','sig_nn/mm' ],
            [ numpy.exp(self.logr),
              nmap, nmx, numpy.sqrt(varnmap),
              nsq, numpy.sqrt(varnsq), mapsq, numpy.sqrt(varmapsq), 
              nmnorm, numpy.sqrt(varnmnorm), nnnorm, numpy.sqrt(varnnnorm) ] )

