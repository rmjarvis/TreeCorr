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

from __future__ import print_function
import numpy as np
import os
import coord
import time
import treecorr

def test_gg_jk():
    # Test the variance estimate for GG correlation with jackknife error estimate.

    def generate_shear_field(nside):
        # Generate a random shear field with a well-defined power spectrum.
        # It generates shears on a grid nside x nside, and returns, x, y, g1, g2
        kvals = np.fft.fftfreq(nside) * 2*np.pi
        kx,ky = np.meshgrid(kvals,kvals)
        k = kx + 1j*ky
        ksq = kx**2 + ky**2

        # Use a random-ish, non-trivial power spectrum.  Needs to be decreasing function of k.
        Pk = 0.3 / (1. + ksq)

        # Make complex gaussian field in k-space.
        f1 = np.random.normal(size=Pk.shape)
        f2 = np.random.normal(size=Pk.shape)
        f = (f1 + 1j*f2) * np.sqrt(0.5)

        # Multiply by the power spectrum to get a realization of a field with this P(k)
        f *= Pk

        # Multiply by exp(2iphi) to get gamma field, rather than kappa.
        ksq[0,0] = 1.  # Avoid division by zero
        exp2iphi = k**2 / ksq
        f *= exp2iphi

        # Inverse fft gives the real-space field.
        gamma = nside * np.fft.ifft2(f)

        # Generate x,y values for the real-space field
        x,y = np.meshgrid(np.linspace(0.,1000.,nside), np.linspace(0.,1000.,nside))

        x = x.ravel()
        y = y.ravel()
        gamma = gamma.ravel()
        return x, y, np.real(gamma), np.imag(gamma)

    # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
    nside = 1000

    # The full simulation needs to run a lot of times to get a good estimate of the variance,
    # but this takes a long time.  So we store the results in the repo.
    # To redo the simulation, just delete the file data/test_gg_jk.fits
    file_name = 'data/test_gg_jk.npz'
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ggs = []
        for run in range(nruns):
            x, y, g1, g2 = generate_shear_field(nside)
            print(run,': ',np.mean(g1),np.std(g1),np.min(g1),np.max(g1))
            cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=20., max_sep=100.)
            gg.process(cat)
            all_ggs.append(gg)

        mean_xip = np.mean([gg.xip for gg in all_ggs], axis=0)
        var_xip = np.var([gg.xip for gg in all_ggs], axis=0)
        mean_xim = np.mean([gg.xim for gg in all_ggs], axis=0)
        var_xim = np.var([gg.xim for gg in all_ggs], axis=0)
        mean_varxip = np.mean([gg.varxip for gg in all_ggs], axis=0)
        mean_varxim = np.mean([gg.varxim for gg in all_ggs], axis=0)

        np.savez(file_name,
                 mean_xip=mean_xip, mean_xim=mean_xim,
                 var_xip=var_xip, var_xim=var_xim,
                 mean_varxip=mean_varxip, mean_varxim=mean_varxim)

    data = np.load(file_name)
    mean_xip = data['mean_xip']
    mean_xim = data['mean_xim']
    var_xip = data['var_xip']
    var_xim = data['var_xim']
    mean_varxip = data['mean_varxip']
    mean_varxim = data['mean_varxim']

    print('mean_xip = ',mean_xip)
    print('mean_xim = ',mean_xim)
    print('mean_varxip = ',mean_varxip)
    print('mean_varxim = ',mean_varxim)
    print('var_xip = ',var_xip)
    print('ratio = ',var_xip / mean_varxip)
    print('var_xim = ',var_xim)
    print('ratio = ',var_xim / mean_varxim)

    # First run with the normal variance estimate, which is too small.
    x, y, g1, g2 = generate_shear_field(nside)
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    gg1 = treecorr.GGCorrelation(bin_size=0.1, min_sep=20., max_sep=100.)
    gg1.process(cat)

    print('xip = ',gg1.xip)
    print('xim = ',gg1.xim)
    print('varxip = ',gg1.varxip)
    print('varxim = ',gg1.varxim)
    print('pullsq for xip = ',(gg1.xip-mean_xip)**2/var_xip)
    print('pullsq for xim = ',(gg1.xim-mean_xim)**2/var_xim)
    print('max pull for xip = ',np.sqrt(np.max((gg1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((gg1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((gg1.xip - mean_xip)**2/var_xip, 25) # within 5 sigma
    np.testing.assert_array_less((gg1.xim - mean_xim)**2/var_xim, 25)
    np.testing.assert_allclose(gg1.varxip, mean_varxip, rtol=0.03)
    np.testing.assert_allclose(gg1.varxim, mean_varxim, rtol=0.03)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxip, var_xip)
    np.testing.assert_array_less(mean_varxim, var_xim)
    np.testing.assert_array_less(gg1.varxip, var_xip)
    np.testing.assert_array_less(gg1.varxim, var_xim)


if __name__ == '__main__':
    test_gg_jk()
