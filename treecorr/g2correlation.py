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

_treecorr.BuildGGCorr.restype = ctypes.c_void_p
_treecorr.BuildGGCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyGGCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoGGSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cint  ]
_treecorr.ProcessAutoGGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint  ]
_treecorr.ProcessCrossGGSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossGGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseGGSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseGGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class G2Correlation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xip         The correlation function, xi_plus(r).
        xim         The correlation funciton, xi_minus(r).
        xip_im      The imaginary part of xi_plus(r).
        xim_im      The imaginary part of xi_plus(r).
        varxi       The variance of xip and xim, only including the shape noise propagated
                    into the final correlation.  This does not include sample variance, so
                    it is always an underestimate of the actual variance.
        weight      The total weight in each bin.
        npairs      The number of pairs going into each bin.

    The usage pattern is as follows:

        gg = treecorr.G2Correlation(config)
        gg.process(cat1)        # For auto-correlation.
        gg.process(cat1,cat2)   # For cross-correlation.
        gg.write(file_name)     # Write out to a file.
        xip = gg.xip            # Or access the correlation function directly.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.xip = numpy.zeros(self.nbins, dtype=float)
        self.xim = numpy.zeros(self.nbins, dtype=float)
        self.xip_im = numpy.zeros(self.nbins, dtype=float)
        self.xim_im = numpy.zeros(self.nbins, dtype=float)

        xip = self.xip.ctypes.data_as(cdouble_ptr)
        xipi = self.xip_im.ctypes.data_as(cdouble_ptr)
        xim = self.xim.ctypes.data_as(cdouble_ptr)
        ximi = self.xim_im.ctypes.data_as(cdouble_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)

        self.corr = _treecorr.BuildGGCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          xip,xipi,xim,ximi,meanlogr,weight,npairs);
        self.logger.debug('Finished building GGCorr')
 

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyGGCorr(self.corr)


    def process_auto(self, cat1):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Starting process G2 auto-correlations for cat %s.',cat1.name)
        field = cat1.getGField(self.min_sep,self.max_sep,self.b)

        if field.sphere:
            _treecorr.ProcessAutoGGSphere(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoGGFlat(self.corr, field.data, self.output_dots)


    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Starting process G2 cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getGField(self.min_sep,self.max_sep,self.b)
        f2 = cat2.getGField(self.min_sep,self.max_sep,self.b)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossGGSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossGGFlat(self.corr, f1.data, f2.data, self.output_dots)


    def process_pairwise(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Starting process G2 pairwise-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getGSimpleField()
        f2 = cat2.getGSimpleField()

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessPairwiseGGSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseGGFlat(self.corr, f1.data, f2.data, self.output_dots)


    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xip[mask1] /= self.weight[mask1]
        self.xim[mask1] /= self.weight[mask1]
        self.xip_im[mask1] /= self.weight[mask1]
        self.xim_im[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg1 * varg2 / self.npairs[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xip[:] = 0
        self.xim[:] = 0
        self.xip_im[:] = 0
        self.xim_im[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0


    def process(self, cat1, cat2=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise AttributeError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = varg1
            self.logger.info("varg = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            self._process_all_auto(cat1)
        else:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = treecorr.calculateVarG(cat2)
            self.logger.info("varg1 = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            self.logger.info("varg2 = %f: sig_sn (per component) = %f",varg2,math.sqrt(varg2))
            self._process_all_cross(cat1,cat2)
        self.finalize(varg1,varg2)


    def write(self, file_name):
        """Write the correlation function to the file, file_name.
        """
        self.logger.info('Writing G2 correlations to %s',file_name)
        
        self.gen_write(
            file_name,
            ['R_nom','<R>','xi+','xi-','xi+_im','xi-_im','sigma_xi','weight','npairs'],
            [ numpy.exp(self.logr), numpy.exp(self.meanlogr),
              self.xip, self.xim, self.xip_im, self.xim_im, numpy.sqrt(self.varxi),
              self.weight, self.npairs ] )


    def calculateMapSq(self, m2_uform=None):
        """Calculate the aperture mass statistics from the correlation function.

        <Map^2>(R) = int_r=0..rmax [1/2 (r/R)^2 dlogr (T+(r/R) xi+(r) + T-(r/R) xi-(r)) ]
        <Mx^2>(R)  = int_r=0..rmax [1/2 (r/R)^2 dlogr (T+(r/R) xi+(r) - T-(r/R) xi-(r)) ]

        The m2_uform parameter sets which definition of the aperture mass to use.
        The default is to look in the config dict that was used to build the catalog,
        or use 'Crittenden' if it is not specified.

        If m2_uform == 'Crittenden':

            U(r) = 1/2Pi (1-r^2) exp(-r^2/2)
            Q(r) = 1/4Pi r^2 exp(-r^2/2)
            T+(s) = (s^4 - 16s^2 + 32)/128 exp(-s^2/4)
            T-(s) = s^4/128 exp(-s^2/4)
            rmax = infinity

        If m2_uform == 'Schneider':

            U(r) = 9/Pi (1-r^2) (1/3-r^2)
            Q(r) = 6/Pi r^2 (1-r^2)
            T+(s) = 12/5Pi (2-15s^2) arccos(s/2)
                      + 1/(100Pi) s sqrt(4-s^2) (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8)
            T-(s) = 3/70Pi s^3 (4-s^2)^(7/2)
            rmax = 2R

        cf Schneider, et al (2001): http://xxx.lanl.gov/abs/astro-ph/0112441

        returns (mapsq, mapsq_im, mxsq, mxsq_im, varmapsq)
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
            Tp = (32. + ssq*(-16. + ssq)) / 128. * exp_factor
            Tm = ssq * ssq / 128. * exp_factor
        else:
            Tp = numpy.zeros_like(s)
            Tm = numpy.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*numpy.pi) * (2.-15.*ssqa) * numpy.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*numpy.pi) * sa * numpy.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
            Tm[s<2.] = 3./(70.*numpy.pi) * sa * ssqa * (4.-ssqa)**3.5
        Tp *= ssq
        Tm *= ssq

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxip = Tp.dot(self.xip)
        Tmxim = Tm.dot(self.xim)
        mapsq = (Tpxip + Tmxim) * 0.5 * self.bin_size
        mxsq = (Tpxip - Tmxim) * 0.5 * self.bin_size
        Tpxip_im = Tp.dot(self.xip_im)
        Tmxim_im = Tm.dot(self.xim_im)
        mapsq_im = (Tpxip_im + Tmxim_im) * 0.5 * self.bin_size
        mxsq_im = (Tpxip_im - Tmxim_im) * 0.5 * self.bin_size

        # The variance of each of these is 
        # Var(<Map^2>(R)) = int_r=0..2R [1/4 s^4 dlogr^2 (T+(s)^2 + T-(s)^2) Var(xi)]
        varmapsq = (Tp**2 + Tm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return mapsq, mapsq_im, mxsq, mxsq_im, varmapsq


    def calculateGamSq(self, eb=False):
        """Calculate the tophat shear variance from the correlation function.

        <Gam^2>(R) = int_r=0..2R [s^2 dlogr S+(s) xi+(r)]
        <Gam^2_E>(R) = int_r=0..2R [1/2 s^2 dlogr (S+(s) xi+(r) + S-(s) xi-(r)]
        <Gam^2_B>(R) = int_r=0..2R [1/2 s^2 dlogr (S+(s) xi+(r) - S-(s) xi-(r)]

        S+(s) = 1/Pi * (4*arccos(s/2) - s sqrt(4-s^2) )
        S-(s) = (s<=2):  1/(Pi s^4) * ( s sqrt(4-s^2) (6-s^2) - 8(3-s^2) arcsin(s/2) )
                (s>=2):  4(s^2-3)/s^4

        cf Schneider, et al, 2001: http://adsabs.harvard.edu/abs/2002A%26A...389..729S

        The default behavior is not to compute the E/B versions.  They are calculated if
        eb is set to True.

        returns (gamsq, vargamsq)                                if eb == False
                (gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e)  if eb == True
        """
        r = numpy.exp(self.logr)
        meanr = numpy.exp(self.meanlogr) # Use the actual mean r for each bin
        s = numpy.outer(1./r, meanr)  
        ssq = s*s
        Sp = numpy.zeros_like(s)
        sa = s[s<2]
        ssqa = ssq[s<2]
        Sp[s<2.] = 1./numpy.pi * (4.*numpy.arccos(sa/2.) - sa*numpy.sqrt(4.-ssqa))
        Sp *= ssq

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Spxip = Sp.dot(self.xip)
        gamsq = Spxip * self.bin_size
        vargamsq = (Sp**2).dot(self.varxi) * self.bin_size**2

        # Stop here if eb == False
        if not eb: return gamsq, vargamsq

        Sm = numpy.empty_like(s)
        Sm[s<2.] = 1./(ssqa*numpy.pi) * (sa*numpy.sqrt(4.-ssqa)*(6.-ssqa)
                                              -8.*(3.-ssqa)*numpy.arcsin(sa/2.))
        Sm[s>=2.] = 4.*(ssq[s>=2]-3.)/ssq[s>=2]
        # This already includes the extra ssq factor.

        Smxim = Sm.dot(self.xim)
        gamsq_e = (Spxip + Smxim) * 0.5 * self.bin_size
        gamsq_b = (Spxip - Smxim) * 0.5 * self.bin_size
        vargamsq_e = (Sp**2 + Sm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e


    def writeMapSq(self, file_name, m2_uform=None):
        """Write the aperture mass statistics based on the correlation function to the
        file, file_name.

        See calculateMapSq for an explanation of the m2_uform parameter.
        """
        self.logger.info('Writing Map^2 from G2 correlations to %s',file_name)

        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = self.calculateMapSq(m2_uform=m2_uform)
        gamsq, vargamsq = self.calculateGamSq()

        self.gen_write(
            file_name,
            ['R','<Map^2>','<Mx^2>','<MMx>(a)','<MMx>(b)','sig_map','<Gam^2>','sig_gam'],
            [ numpy.exp(self.logr),
              mapsq, mxsq, mapsq_im, -mxsq_im, numpy.sqrt(varmapsq),
              gamsq, numpy.sqrt(vargamsq) ] )


