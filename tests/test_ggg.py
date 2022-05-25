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

import numpy as np
import treecorr
import os
import coord
import time

from test_helper import do_pickle, assert_raises, timer, is_ccw, is_ccw_3d

@timer
def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) )
    w = rng.random_sample(ngal)
    g1 = rng.normal(0,0.2, (ngal,) )
    g2 = rng.normal(0,0.2, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    nubins = 5
    nvbins = 5
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins, brute=True)
    ggg.process(cat, num_threads=2)

    true_ntri = np.zeros((nrbins, nubins, 2*nvbins), dtype=int)
    true_weight = np.zeros((nrbins, nubins, 2*nvbins), dtype=float)
    true_gam0 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                d12 = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                d23 = np.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2)
                d31 = np.sqrt((x[k]-x[i])**2 + (y[k]-y[i])**2)

                d3, d2, d1 = sorted([d12, d23, d31])
                rindex = np.floor(np.log(d2/min_sep) / bin_size).astype(int)
                if rindex < 0 or rindex >= nrbins: continue

                if [d1, d2, d3] == [d23, d31, d12]: ii,jj,kk = i,j,k
                elif [d1, d2, d3] == [d23, d12, d31]: ii,jj,kk = i,k,j
                elif [d1, d2, d3] == [d31, d12, d23]: ii,jj,kk = j,k,i
                elif [d1, d2, d3] == [d31, d23, d12]: ii,jj,kk = j,i,k
                elif [d1, d2, d3] == [d12, d23, d31]: ii,jj,kk = k,i,j
                elif [d1, d2, d3] == [d12, d31, d23]: ii,jj,kk = k,j,i
                else: assert False
                # Now use ii, jj, kk rather than i,j,k, to get the indices
                # that correspond to the points in the right order.

                u = d3/d2
                v = (d1-d2)/d3
                if not is_ccw(x[ii],y[ii],x[jj],y[jj],x[kk],y[kk]):
                    v = -v

                uindex = np.floor(u / bin_size).astype(int)
                assert 0 <= uindex < nubins
                vindex = np.floor((v+1) / bin_size).astype(int)
                assert 0 <= vindex < 2*nvbins

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x[i] + x[j] + x[k])/3.
                ceny = (y[i] + y[j] + y[k])/3.

                expmialpha1 = (x[ii]-cenx) - 1j*(y[ii]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x[jj]-cenx) - 1j*(y[jj]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x[kk]-cenx) - 1j*(y[kk]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w[i] * w[j] * w[k]
                g1p = (g1[ii] + 1j*g2[ii]) * expmialpha1**2
                g2p = (g1[jj] + 1j*g2[jj]) * expmialpha2**2
                g3p = (g1[kk] + 1j*g2[kk]) * expmialpha3**2
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[rindex,uindex,vindex] += 1
                true_weight[rindex,uindex,vindex] += www
                true_gam0[rindex,uindex,vindex] += gam0
                true_gam1[rindex,uindex,vindex] += gam1
                true_gam2[rindex,uindex,vindex] += gam2
                true_gam3[rindex,uindex,vindex] += gam3

    pos = true_weight > 0
    true_gam0[pos] /= true_weight[pos]
    true_gam1[pos] /= true_weight[pos]
    true_gam2[pos] /= true_weight[pos]
    true_gam3[pos] /= true_weight[pos]

    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=1.e-5, atol=1.e-8)

    # Check that running via the corr3 script works correctly.
    config = treecorr.config.read_config('configs/ggg_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat.write(config['file_name'])
        t1 = time.time()
        treecorr.corr3(config)
        t2 = time.time()
        print('Time for auto correlation = ',t2-t1)
        data = fitsio.read(config['ggg_file_name'])
        np.testing.assert_allclose(data['r_nom'], ggg.rnom.flatten())
        np.testing.assert_allclose(data['u_nom'], ggg.u.flatten())
        np.testing.assert_allclose(data['v_nom'], ggg.v.flatten())
        np.testing.assert_allclose(data['ntri'], ggg.ntri.flatten())
        np.testing.assert_allclose(data['weight'], ggg.weight.flatten())
        np.testing.assert_allclose(data['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Also check the cross calculation.
    # Here, we get 6x as many triangles, since each triangle is discovered 6 times.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins, brute=True)
    ggg.process(cat, cat, cat, num_threads=2)
    np.testing.assert_array_equal(ggg.ntri, 6*true_ntri)
    np.testing.assert_allclose(ggg.weight, 6*true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    # Using the GGGCrossCorrelation class instead finds each triangle 6 times, but puts
    # them into 6 different accumualtions, so they all end up with the same answer.
    gggc = treecorr.GGGCrossCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                        brute=True)
    t1 = time.time()
    gggc.process(cat, cat, cat, num_threads=2)
    t2 = time.time()
    print('Time for cross correlation = ',t2-t1)
    # All six correlation functions are equivalent to the auto results:
    for g in [gggc.g1g2g3, gggc.g1g3g2, gggc.g2g1g3, gggc.g2g3g1, gggc.g3g1g2, gggc.g3g2g1]:
        np.testing.assert_array_equal(g.ntri, true_ntri)
        np.testing.assert_allclose(g.weight, true_weight, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    # Or with 2 argument version, finds each triangle 3 times.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins, brute=True)
    t1 = time.time()
    ggg.process(cat, cat, num_threads=2)
    t2 = time.time()
    print('Time for 1-2 cross correlation = ',t2-t1)
    np.testing.assert_array_equal(ggg.ntri, 3*true_ntri)
    np.testing.assert_allclose(ggg.weight, 3*true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    # Again, GGGCrossCorrelation gets it right in each permutation.
    t1 = time.time()
    gggc.process(cat, cat, num_threads=2)
    t2 = time.time()
    print('Time for 1-2 cross correlation = ',t2-t1)
    for g in [gggc.g1g2g3, gggc.g1g3g2, gggc.g2g1g3, gggc.g2g3g1, gggc.g3g1g2, gggc.g3g2g1]:
        np.testing.assert_array_equal(g.ntri, true_ntri)
        np.testing.assert_allclose(g.weight, true_weight, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
        np.testing.assert_allclose(g.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    # Repeat with binslop = 0, since the code flow is different from brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  bin_slop=0, max_top=0)
    ggg.process(cat)
    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-3, atol=1.e-4)

    # Check a few basic operations with a GGGCorrelation object.
    do_pickle(ggg)

    ggg2 = ggg.copy()
    ggg2 += ggg
    np.testing.assert_allclose(ggg2.ntri, 2*ggg.ntri)
    np.testing.assert_allclose(ggg2.weight, 2*ggg.weight)
    np.testing.assert_allclose(ggg2.meand1, 2*ggg.meand1)
    np.testing.assert_allclose(ggg2.meand2, 2*ggg.meand2)
    np.testing.assert_allclose(ggg2.meand3, 2*ggg.meand3)
    np.testing.assert_allclose(ggg2.meanlogd1, 2*ggg.meanlogd1)
    np.testing.assert_allclose(ggg2.meanlogd2, 2*ggg.meanlogd2)
    np.testing.assert_allclose(ggg2.meanlogd3, 2*ggg.meanlogd3)
    np.testing.assert_allclose(ggg2.meanu, 2*ggg.meanu)
    np.testing.assert_allclose(ggg2.meanv, 2*ggg.meanv)
    np.testing.assert_allclose(ggg2.gam0r, 2*ggg.gam0r)
    np.testing.assert_allclose(ggg2.gam0i, 2*ggg.gam0i)
    np.testing.assert_allclose(ggg2.gam1r, 2*ggg.gam1r)
    np.testing.assert_allclose(ggg2.gam1i, 2*ggg.gam1i)
    np.testing.assert_allclose(ggg2.gam2r, 2*ggg.gam2r)
    np.testing.assert_allclose(ggg2.gam2i, 2*ggg.gam2i)
    np.testing.assert_allclose(ggg2.gam3r, 2*ggg.gam3r)
    np.testing.assert_allclose(ggg2.gam3i, 2*ggg.gam3i)

    ggg2.clear()
    ggg2 += ggg
    np.testing.assert_allclose(ggg2.ntri, ggg.ntri)
    np.testing.assert_allclose(ggg2.weight, ggg.weight)
    np.testing.assert_allclose(ggg2.meand1, ggg.meand1)
    np.testing.assert_allclose(ggg2.meand2, ggg.meand2)
    np.testing.assert_allclose(ggg2.meand3, ggg.meand3)
    np.testing.assert_allclose(ggg2.meanlogd1, ggg.meanlogd1)
    np.testing.assert_allclose(ggg2.meanlogd2, ggg.meanlogd2)
    np.testing.assert_allclose(ggg2.meanlogd3, ggg.meanlogd3)
    np.testing.assert_allclose(ggg2.meanu, ggg.meanu)
    np.testing.assert_allclose(ggg2.meanv, ggg.meanv)
    np.testing.assert_allclose(ggg2.gam0r, ggg.gam0r)
    np.testing.assert_allclose(ggg2.gam0i, ggg.gam0i)
    np.testing.assert_allclose(ggg2.gam1r, ggg.gam1r)
    np.testing.assert_allclose(ggg2.gam1i, ggg.gam1i)
    np.testing.assert_allclose(ggg2.gam2r, ggg.gam2r)
    np.testing.assert_allclose(ggg2.gam2i, ggg.gam2i)
    np.testing.assert_allclose(ggg2.gam3r, ggg.gam3r)
    np.testing.assert_allclose(ggg2.gam3i, ggg.gam3i)

    ascii_name = 'output/ggg_ascii.txt'
    ggg.write(ascii_name, precision=16)
    ggg3 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
    ggg3.read(ascii_name)
    np.testing.assert_allclose(ggg3.ntri, ggg.ntri)
    np.testing.assert_allclose(ggg3.weight, ggg.weight)
    np.testing.assert_allclose(ggg3.meand1, ggg.meand1)
    np.testing.assert_allclose(ggg3.meand2, ggg.meand2)
    np.testing.assert_allclose(ggg3.meand3, ggg.meand3)
    np.testing.assert_allclose(ggg3.meanlogd1, ggg.meanlogd1)
    np.testing.assert_allclose(ggg3.meanlogd2, ggg.meanlogd2)
    np.testing.assert_allclose(ggg3.meanlogd3, ggg.meanlogd3)
    np.testing.assert_allclose(ggg3.meanu, ggg.meanu)
    np.testing.assert_allclose(ggg3.meanv, ggg.meanv)
    np.testing.assert_allclose(ggg3.gam0r, ggg.gam0r)
    np.testing.assert_allclose(ggg3.gam0i, ggg.gam0i)
    np.testing.assert_allclose(ggg3.gam1r, ggg.gam1r)
    np.testing.assert_allclose(ggg3.gam1i, ggg.gam1i)
    np.testing.assert_allclose(ggg3.gam2r, ggg.gam2r)
    np.testing.assert_allclose(ggg3.gam2i, ggg.gam2i)
    np.testing.assert_allclose(ggg3.gam3r, ggg.gam3r)
    np.testing.assert_allclose(ggg3.gam3i, ggg.gam3i)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/ggg_fits.fits'
        ggg.write(fits_name)
        ggg4 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
        ggg4.read(fits_name)
        np.testing.assert_allclose(ggg4.ntri, ggg.ntri)
        np.testing.assert_allclose(ggg4.weight, ggg.weight)
        np.testing.assert_allclose(ggg4.meand1, ggg.meand1)
        np.testing.assert_allclose(ggg4.meand2, ggg.meand2)
        np.testing.assert_allclose(ggg4.meand3, ggg.meand3)
        np.testing.assert_allclose(ggg4.meanlogd1, ggg.meanlogd1)
        np.testing.assert_allclose(ggg4.meanlogd2, ggg.meanlogd2)
        np.testing.assert_allclose(ggg4.meanlogd3, ggg.meanlogd3)
        np.testing.assert_allclose(ggg4.meanu, ggg.meanu)
        np.testing.assert_allclose(ggg4.meanv, ggg.meanv)
        np.testing.assert_allclose(ggg4.gam0r, ggg.gam0r)
        np.testing.assert_allclose(ggg4.gam0i, ggg.gam0i)
        np.testing.assert_allclose(ggg4.gam1r, ggg.gam1r)
        np.testing.assert_allclose(ggg4.gam1i, ggg.gam1i)
        np.testing.assert_allclose(ggg4.gam2r, ggg.gam2r)
        np.testing.assert_allclose(ggg4.gam2i, ggg.gam2i)
        np.testing.assert_allclose(ggg4.gam3r, ggg.gam3r)
        np.testing.assert_allclose(ggg4.gam3i, ggg.gam3i)

    assert ggg.var_method == 'shot'  # Only option currently.

    with assert_raises(TypeError):
        ggg2 += config
    ggg5 = treecorr.GGGCorrelation(min_sep=min_sep/2, bin_size=bin_size, nbins=nrbins)
    with assert_raises(ValueError):
        ggg2 += ggg5
    ggg6 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size/2, nbins=nrbins)
    with assert_raises(ValueError):
        ggg2 += ggg6
    ggg7 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins*2)
    with assert_raises(ValueError):
        ggg2 += ggg7
    ggg8 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   min_u=0.1)
    with assert_raises(ValueError):
        ggg2 += ggg8
    ggg0 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   max_u=0.1)
    with assert_raises(ValueError):
        ggg2 += ggg0
    ggg10 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   nubins=nrbins*2)
    with assert_raises(ValueError):
        ggg2 += ggg10
    ggg11 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   min_v=0.1)
    with assert_raises(ValueError):
        ggg2 += ggg11
    ggg12 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   max_v=0.1)
    with assert_raises(ValueError):
        ggg2 += ggg12
    ggg13 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                   nvbins=nrbins*2)
    with assert_raises(ValueError):
        ggg2 += ggg13

@timer
def test_direct_spherical():
    # Repeat in spherical coords

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) ) + 200  # Put everything at large y, so small angle on sky
    z = rng.normal(0,s, (ngal,) )
    w = rng.random_sample(ngal)
    g1 = rng.normal(0,0.2, (ngal,) )
    g2 = rng.normal(0,0.2, (ngal,) )
    w = np.ones_like(w)

    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)

    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', w=w, g1=g1, g2=g2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    nubins = 5
    nvbins = 5
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  sep_units='deg', brute=True)
    ggg.process(cat)

    r = np.sqrt(x**2 + y**2 + z**2)
    x /= r;  y /= r;  z /= r
    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    true_ntri = np.zeros((nrbins, nubins, 2*nvbins), dtype=int)
    true_weight = np.zeros((nrbins, nubins, 2*nvbins), dtype=float)
    true_gam0 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)

    rad_min_sep = min_sep * coord.degrees / coord.radians
    c = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra, dec)]
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                d12 = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                d23 = np.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2 + (z[j]-z[k])**2)
                d31 = np.sqrt((x[k]-x[i])**2 + (y[k]-y[i])**2 + (z[k]-z[i])**2)

                d3, d2, d1 = sorted([d12, d23, d31])
                rindex = np.floor(np.log(d2/rad_min_sep) / bin_size).astype(int)
                if rindex < 0 or rindex >= nrbins: continue

                if [d1, d2, d3] == [d23, d31, d12]: ii,jj,kk = i,j,k
                elif [d1, d2, d3] == [d23, d12, d31]: ii,jj,kk = i,k,j
                elif [d1, d2, d3] == [d31, d12, d23]: ii,jj,kk = j,k,i
                elif [d1, d2, d3] == [d31, d23, d12]: ii,jj,kk = j,i,k
                elif [d1, d2, d3] == [d12, d23, d31]: ii,jj,kk = k,i,j
                elif [d1, d2, d3] == [d12, d31, d23]: ii,jj,kk = k,j,i
                else: assert False
                # Now use ii, jj, kk rather than i,j,k, to get the indices
                # that correspond to the points in the right order.

                u = d3/d2
                v = (d1-d2)/d3
                if not is_ccw_3d(x[ii],y[ii],z[ii],x[jj],y[jj],z[jj],x[kk],y[kk],z[kk]):
                    v = -v

                uindex = np.floor(u / bin_size).astype(int)
                assert 0 <= uindex < nubins
                vindex = np.floor((v+1) / bin_size).astype(int)
                assert 0 <= vindex < 2*nvbins

                # Rotate shears to coordinates where line connecting to center is horizontal.
                # Original orientation is where north is up.
                cenx = (x[i] + x[j] + x[k])/3.
                ceny = (y[i] + y[j] + y[k])/3.
                cenz = (z[i] + z[j] + z[k])/3.
                cen = coord.CelestialCoord.from_xyz(cenx,ceny,cenz)

                theta1 = 90*coord.degrees - c[ii].angleBetween(north_pole, cen)
                theta2 = 90*coord.degrees - c[jj].angleBetween(north_pole, cen)
                theta3 = 90*coord.degrees - c[kk].angleBetween(north_pole, cen)
                exp2theta1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
                exp2theta2 = np.cos(2*theta2) + 1j * np.sin(2*theta2)
                exp2theta3 = np.cos(2*theta3) + 1j * np.sin(2*theta3)

                www = w[i] * w[j] * w[k]
                g1p = (g1[ii] + 1j*g2[ii]) * exp2theta1
                g2p = (g1[jj] + 1j*g2[jj]) * exp2theta2
                g3p = (g1[kk] + 1j*g2[kk]) * exp2theta3
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[rindex,uindex,vindex] += 1
                true_weight[rindex,uindex,vindex] += www
                true_gam0[rindex,uindex,vindex] += gam0
                true_gam1[rindex,uindex,vindex] += gam1
                true_gam2[rindex,uindex,vindex] += gam2
                true_gam3[rindex,uindex,vindex] += gam3

    pos = true_weight > 0
    true_gam0[pos] /= true_weight[pos]
    true_gam1[pos] /= true_weight[pos]
    true_gam2[pos] /= true_weight[pos]
    true_gam3[pos] /= true_weight[pos]

    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    # Check that running via the corr3 script works correctly.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        config = treecorr.config.read_config('configs/ggg_direct_spherical.yaml')
        cat.write(config['file_name'])
        treecorr.corr3(config)
        data = fitsio.read(config['ggg_file_name'])
        np.testing.assert_allclose(data['r_nom'], ggg.rnom.flatten())
        np.testing.assert_allclose(data['u_nom'], ggg.u.flatten())
        np.testing.assert_allclose(data['v_nom'], ggg.v.flatten())
        np.testing.assert_allclose(data['ntri'], ggg.ntri.flatten())
        np.testing.assert_allclose(data['weight'], ggg.weight.flatten())
        np.testing.assert_allclose(data['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
        np.testing.assert_allclose(data['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  sep_units='deg', bin_slop=0, max_top=0)
    ggg.process(cat)
    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-3, atol=1.e-4)

@timer
def test_direct_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g1_1 = rng.normal(0,sig_gam, (ngal,) )
    g2_1 = rng.normal(0,sig_gam, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,sig_gam, (ngal,) )
    g2_2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal)
    g1_3 = rng.normal(0,sig_gam, (ngal,) )
    g2_3 = rng.normal(0,sig_gam, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True)
    ggg.process(cat1, cat2, cat3, num_threads=2)

    # Figure out the correct answer for each permutation
    # (We'll need them separately below when we use GGGCrossCorrelation.)
    true_ntri_123 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_132 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_213 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_231 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_312 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_321 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_gam0_123 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_132 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_213 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_231 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_312 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_321 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_123 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_132 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_213 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_231 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_312 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_321 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_123 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_132 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_213 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_231 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_312 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_321 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_123 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_132 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_213 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_231 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_312 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_321 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_123 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_132 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_213 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_231 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_312 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_321 = np.zeros( (nrbins, nubins, 2*nvbins) )
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    max_sep = min_sep * np.exp(nrbins*bin_size)
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                dij = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                dik = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                djk = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = is_ccw(x1[i],y1[i],x2[j],y2[j],x3[k],y3[k])

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x3[k])/3.
                ceny = (y1[i] + y2[j] + y3[k])/3.

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        ii,jj,kk = i,j,k
                        true_ntri = true_ntri_123
                        true_gam0 = true_gam0_123
                        true_gam1 = true_gam1_123
                        true_gam2 = true_gam2_123
                        true_gam3 = true_gam3_123
                        true_weight = true_weight_123
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        g1p, g2p, g3p = g2p, g1p, g3p
                        true_ntri = true_ntri_213
                        true_gam0 = true_gam0_213
                        true_gam1 = true_gam1_213
                        true_gam2 = true_gam2_213
                        true_gam3 = true_gam3_213
                        true_weight = true_weight_213
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        g1p, g2p, g3p = g2p, g3p, g1p
                        true_ntri = true_ntri_231
                        true_gam0 = true_gam0_231
                        true_gam1 = true_gam1_231
                        true_gam2 = true_gam2_231
                        true_gam3 = true_gam3_231
                        true_weight = true_weight_231
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        g1p, g2p, g3p = g1p, g3p, g2p
                        true_ntri = true_ntri_132
                        true_gam0 = true_gam0_132
                        true_gam1 = true_gam1_132
                        true_gam2 = true_gam2_132
                        true_gam3 = true_gam3_132
                        true_weight = true_weight_132
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        g1p, g2p, g3p = g3p, g1p, g2p
                        true_ntri = true_ntri_312
                        true_gam0 = true_gam0_312
                        true_gam1 = true_gam1_312
                        true_gam2 = true_gam2_312
                        true_gam3 = true_gam3_312
                        true_weight = true_weight_312
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        g1p, g2p, g3p = g3p, g2p, g1p
                        true_ntri = true_ntri_321
                        true_gam0 = true_gam0_321
                        true_gam1 = true_gam1_321
                        true_gam2 = true_gam2_321
                        true_gam3 = true_gam3_321
                        true_weight = true_weight_321
                        ccw = not ccw

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if r < min_sep or r >= max_sep: continue
                if u < min_u or u >= max_u: continue
                if v < min_v or v >= max_v: continue
                if not ccw:
                    v = -v
                kr = int(np.floor( (np.log(r)-log_min_sep) / bin_size ))
                ku = int(np.floor( (u-min_u) / ubin_size ))
                if v > 0:
                    kv = int(np.floor( (v-min_v) / vbin_size )) + nvbins
                else:
                    kv = int(np.floor( (v-(-max_v)) / vbin_size ))
                assert 0 <= kr < nrbins
                assert 0 <= ku < nubins
                assert 0 <= kv < 2*nvbins

                www = w1[i] * w2[j] * w3[k]
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1
                true_gam2[kr,ku,kv] += gam2
                true_gam3[kr,ku,kv] += gam3

    n_list = [true_ntri_123, true_ntri_132, true_ntri_213, true_ntri_231,
              true_ntri_312, true_ntri_321]
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_gam0_123, true_gam0_132, true_gam0_213, true_gam0_231,
               true_gam0_312, true_gam0_321]
    g1_list = [true_gam1_123, true_gam1_132, true_gam1_213, true_gam1_231,
               true_gam1_312, true_gam1_321]
    g2_list = [true_gam2_123, true_gam2_132, true_gam2_213, true_gam2_231,
               true_gam2_312, true_gam2_321]
    g3_list = [true_gam3_123, true_gam3_132, true_gam3_213, true_gam3_231,
               true_gam3_312, true_gam3_321]

    # With the regular GGGCorrelation class, we end up with the sum of all permutations.
    true_ntri_sum = sum(n_list)
    true_weight_sum = sum(w_list)
    true_gam0_sum = sum(g0_list)
    true_gam1_sum = sum(g1_list)
    true_gam2_sum = sum(g2_list)
    true_gam3_sum = sum(g3_list)
    pos = true_weight_sum > 0
    true_gam0_sum[pos] /= true_weight_sum[pos]
    true_gam1_sum[pos] /= true_weight_sum[pos]
    true_gam2_sum[pos] /= true_weight_sum[pos]
    true_gam3_sum[pos] /= true_weight_sum[pos]
    #print('true_ntri = ',true_ntri_sum)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)
    np.testing.assert_allclose(ggg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam0, true_gam0_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam1, true_gam1_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam2, true_gam2_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam3, true_gam3_sum, rtol=1.e-5)

    # Now normalize each one individually.
    for w,g0,g1,g2,g3 in zip(w_list, g0_list, g1_list, g2_list, g3_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]
        g2[pos] /= w[pos]
        g3[pos] /= w[pos]

    # Repeat with the full CrossCorrelation class, which distinguishes the permutations.
    gggc = treecorr.GGGCrossCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                        min_u=min_u, max_u=max_u, nubins=nubins,
                                        min_v=min_v, max_v=max_v, nvbins=nvbins,
                                        brute=True, verbose=1)
    gggc.process(cat1, cat2, cat3)

    #print('true_ntri_123 = ',true_ntri_123)
    #print('diff = ',gggc.g1g2g3.ntri - true_ntri_123)
    np.testing.assert_array_equal(gggc.g1g2g3.ntri, true_ntri_123)
    np.testing.assert_array_equal(gggc.g1g3g2.ntri, true_ntri_132)
    np.testing.assert_array_equal(gggc.g2g1g3.ntri, true_ntri_213)
    np.testing.assert_array_equal(gggc.g2g3g1.ntri, true_ntri_231)
    np.testing.assert_array_equal(gggc.g3g1g2.ntri, true_ntri_312)
    np.testing.assert_array_equal(gggc.g3g2g1.ntri, true_ntri_321)
    np.testing.assert_allclose(gggc.g1g2g3.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam0, true_gam0_132, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam0, true_gam0_312, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam0, true_gam0_321, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam1, true_gam1_123, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam1, true_gam1_132, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam1, true_gam1_213, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam1, true_gam1_231, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam1, true_gam1_312, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam1, true_gam1_321, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam2, true_gam2_123, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam2, true_gam2_132, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam2, true_gam2_213, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam2, true_gam2_231, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam2, true_gam2_312, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam2, true_gam2_321, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam3, true_gam3_123, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam3, true_gam3_132, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam3, true_gam3_213, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam3, true_gam3_231, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam3, true_gam3_312, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam3, true_gam3_321, rtol=1.e-5)

    # Repeat with binslop = 0
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1)
    ggg.process(cat1, cat2, cat3)
    #print('binslop > 0: ggg.ntri = ',ggg.ntri)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)

    # And again with no top-level recursion
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0)
    ggg.process(cat1, cat2, cat3)
    #print('max_top = 0: ggg.ntri = ',ggg.ntri)
    #print('true_ntri = ',true_ntri_sum)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        ggg.process(cat1, cat3=cat3)

    # Check a few basic operations with a GGGCrossCorrelation object.
    do_pickle(gggc)

    gggc2 = gggc.copy()
    gggc2 += gggc
    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        g2 = getattr(gggc2, perm)
        g1 = getattr(gggc, perm)
        np.testing.assert_allclose(g2.ntri, 2*g1.ntri)
        np.testing.assert_allclose(g2.weight, 2*g1.weight)
        np.testing.assert_allclose(g2.meand1, 2*g1.meand1)
        np.testing.assert_allclose(g2.meand2, 2*g1.meand2)
        np.testing.assert_allclose(g2.meand3, 2*g1.meand3)
        np.testing.assert_allclose(g2.meanlogd1, 2*g1.meanlogd1)
        np.testing.assert_allclose(g2.meanlogd2, 2*g1.meanlogd2)
        np.testing.assert_allclose(g2.meanlogd3, 2*g1.meanlogd3)
        np.testing.assert_allclose(g2.meanu, 2*g1.meanu)
        np.testing.assert_allclose(g2.meanv, 2*g1.meanv)
        np.testing.assert_allclose(g2.gam0, 2*g1.gam0)
        np.testing.assert_allclose(g2.gam1, 2*g1.gam1)
        np.testing.assert_allclose(g2.gam2, 2*g1.gam2)
        np.testing.assert_allclose(g2.gam3, 2*g1.gam3)

    gggc2.clear()
    gggc2 += gggc
    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        g2 = getattr(gggc2, perm)
        g1 = getattr(gggc, perm)
        np.testing.assert_allclose(g2.ntri, g1.ntri)
        np.testing.assert_allclose(g2.weight, g1.weight)
        np.testing.assert_allclose(g2.meand1, g1.meand1)
        np.testing.assert_allclose(g2.meand2, g1.meand2)
        np.testing.assert_allclose(g2.meand3, g1.meand3)
        np.testing.assert_allclose(g2.meanlogd1, g1.meanlogd1)
        np.testing.assert_allclose(g2.meanlogd2, g1.meanlogd2)
        np.testing.assert_allclose(g2.meanlogd3, g1.meanlogd3)
        np.testing.assert_allclose(g2.meanu, g1.meanu)
        np.testing.assert_allclose(g2.meanv, g1.meanv)
        np.testing.assert_allclose(g2.gam0, g1.gam0)
        np.testing.assert_allclose(g2.gam1, g1.gam1)
        np.testing.assert_allclose(g2.gam2, g1.gam2)
        np.testing.assert_allclose(g2.gam3, g1.gam3)

    with assert_raises(TypeError):
        gggc2 += {}
    with assert_raises(TypeError):
        gggc2 += ggg

    # Can't add with different config specs
    gggc3 = treecorr.GGGCrossCorrelation(min_sep=min_sep/2, bin_size=bin_size, nbins=nrbins)
    with assert_raises(ValueError):
        gggc2 += gggc3

    # Test I/O
    ascii_name = 'output/gggc_ascii.txt'
    gggc.write(ascii_name, precision=16)
    gggc3 = treecorr.GGGCrossCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
    gggc3.read(ascii_name)
    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        g2 = getattr(gggc3, perm)
        g1 = getattr(gggc, perm)
        np.testing.assert_allclose(g2.ntri, g1.ntri)
        np.testing.assert_allclose(g2.weight, g1.weight)
        np.testing.assert_allclose(g2.meand1, g1.meand1)
        np.testing.assert_allclose(g2.meand2, g1.meand2)
        np.testing.assert_allclose(g2.meand3, g1.meand3)
        np.testing.assert_allclose(g2.meanlogd1, g1.meanlogd1)
        np.testing.assert_allclose(g2.meanlogd2, g1.meanlogd2)
        np.testing.assert_allclose(g2.meanlogd3, g1.meanlogd3)
        np.testing.assert_allclose(g2.meanu, g1.meanu)
        np.testing.assert_allclose(g2.meanv, g1.meanv)
        np.testing.assert_allclose(g2.gam0, g1.gam0)
        np.testing.assert_allclose(g2.gam1, g1.gam1)
        np.testing.assert_allclose(g2.gam2, g1.gam2)
        np.testing.assert_allclose(g2.gam3, g1.gam3)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/gggc_fits.fits'
        gggc.write(fits_name)
        gggc4 = treecorr.GGGCrossCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
        gggc4.read(fits_name)
        for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
            g2 = getattr(gggc4, perm)
            g1 = getattr(gggc, perm)
            np.testing.assert_allclose(g2.ntri, g1.ntri)
            np.testing.assert_allclose(g2.weight, g1.weight)
            np.testing.assert_allclose(g2.meand1, g1.meand1)
            np.testing.assert_allclose(g2.meand2, g1.meand2)
            np.testing.assert_allclose(g2.meand3, g1.meand3)
            np.testing.assert_allclose(g2.meanlogd1, g1.meanlogd1)
            np.testing.assert_allclose(g2.meanlogd2, g1.meanlogd2)
            np.testing.assert_allclose(g2.meanlogd3, g1.meanlogd3)
            np.testing.assert_allclose(g2.meanu, g1.meanu)
            np.testing.assert_allclose(g2.meanv, g1.meanv)
            np.testing.assert_allclose(g2.gam0, g1.gam0)
            np.testing.assert_allclose(g2.gam1, g1.gam1)
            np.testing.assert_allclose(g2.gam2, g1.gam2)
            np.testing.assert_allclose(g2.gam3, g1.gam3)

@timer
def test_direct_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g1_1 = rng.normal(0,sig_gam, (ngal,) )
    g2_1 = rng.normal(0,sig_gam, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,sig_gam, (ngal,) )
    g2_2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True)
    ggg.process(cat1, cat2, num_threads=2)

    # Figure out the correct answer for each permutation
    # (We'll need them separately below when we use GGGCrossCorrelation.)
    true_ntri_122 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_212 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_ntri_221 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_gam0_122 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_212 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_221 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_122 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_212 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_221 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_122 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_212 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_221 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_122 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_212 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam3_221 = np.zeros( (nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_122 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_212 = np.zeros( (nrbins, nubins, 2*nvbins) )
    true_weight_221 = np.zeros( (nrbins, nubins, 2*nvbins) )
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    max_sep = min_sep * np.exp(nrbins*bin_size)
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(j+1,ngal):
                dij = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                dik = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                djk = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = is_ccw(x1[i],y1[i],x2[j],y2[j],x2[k],y2[k])

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x2[k])/3.
                ceny = (y1[i] + y2[j] + y2[k])/3.

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        ii,jj,kk = i,j,k
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam1_122
                        true_gam2 = true_gam2_122
                        true_gam3 = true_gam3_122
                        true_weight = true_weight_122
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        g1p, g2p, g3p = g2p, g1p, g3p
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_gam2 = true_gam2_212
                        true_gam3 = true_gam3_212
                        true_weight = true_weight_212
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        g1p, g2p, g3p = g2p, g3p, g1p
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_gam2 = true_gam2_221
                        true_gam3 = true_gam3_221
                        true_weight = true_weight_221
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        g1p, g2p, g3p = g1p, g3p, g2p
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam1_122
                        true_gam2 = true_gam2_122
                        true_gam3 = true_gam3_122
                        true_weight = true_weight_122
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        g1p, g2p, g3p = g3p, g1p, g2p
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_gam2 = true_gam2_212
                        true_gam3 = true_gam3_212
                        true_weight = true_weight_212
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        g1p, g2p, g3p = g3p, g2p, g1p
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_gam2 = true_gam2_221
                        true_gam3 = true_gam3_221
                        true_weight = true_weight_221
                        ccw = not ccw

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if r < min_sep or r >= max_sep: continue
                if u < min_u or u >= max_u: continue
                if v < min_v or v >= max_v: continue
                if not ccw:
                    v = -v
                kr = int(np.floor( (np.log(r)-log_min_sep) / bin_size ))
                ku = int(np.floor( (u-min_u) / ubin_size ))
                if v > 0:
                    kv = int(np.floor( (v-min_v) / vbin_size )) + nvbins
                else:
                    kv = int(np.floor( (v-(-max_v)) / vbin_size ))
                assert 0 <= kr < nrbins
                assert 0 <= ku < nubins
                assert 0 <= kv < 2*nvbins

                www = w1[i] * w2[j] * w2[k]
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1
                true_gam2[kr,ku,kv] += gam2
                true_gam3[kr,ku,kv] += gam3

    n_list = [true_ntri_122, true_ntri_212, true_ntri_221]
    w_list = [true_weight_122, true_weight_212, true_weight_221]
    g0_list = [true_gam0_122, true_gam0_212, true_gam0_221]
    g1_list = [true_gam1_122, true_gam1_212, true_gam1_221]
    g2_list = [true_gam2_122, true_gam2_212, true_gam2_221]
    g3_list = [true_gam3_122, true_gam3_212, true_gam3_221]

    # With the regular GGGCorrelation class, we end up with the sum of all permutations.
    true_ntri_sum = sum(n_list)
    true_weight_sum = sum(w_list)
    true_gam0_sum = sum(g0_list)
    true_gam1_sum = sum(g1_list)
    true_gam2_sum = sum(g2_list)
    true_gam3_sum = sum(g3_list)
    pos = true_weight_sum > 0
    true_gam0_sum[pos] /= true_weight_sum[pos]
    true_gam1_sum[pos] /= true_weight_sum[pos]
    true_gam2_sum[pos] /= true_weight_sum[pos]
    true_gam3_sum[pos] /= true_weight_sum[pos]
    #print('true_ntri = ',true_ntri_sum)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)
    np.testing.assert_allclose(ggg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam0, true_gam0_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam1, true_gam1_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam2, true_gam2_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam3, true_gam3_sum, rtol=1.e-5)

    # Now normalize each one individually.
    for w,g0,g1,g2,g3 in zip(w_list, g0_list, g1_list, g2_list, g3_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]
        g2[pos] /= w[pos]
        g3[pos] /= w[pos]

    # Repeat with the full CrossCorrelation class, which distinguishes the permutations.
    gggc = treecorr.GGGCrossCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                        min_u=min_u, max_u=max_u, nubins=nubins,
                                        min_v=min_v, max_v=max_v, nvbins=nvbins,
                                        brute=True, verbose=1)
    gggc.process(cat1, cat2)

    #print('true_ntri_122 = ',true_ntri_122)
    #print('diff = ',gggc.g1g2g3.ntri - true_ntri_122)
    np.testing.assert_array_equal(gggc.g1g2g3.ntri, true_ntri_122)
    np.testing.assert_array_equal(gggc.g1g3g2.ntri, true_ntri_122)
    np.testing.assert_array_equal(gggc.g2g1g3.ntri, true_ntri_212)
    np.testing.assert_array_equal(gggc.g2g3g1.ntri, true_ntri_221)
    np.testing.assert_array_equal(gggc.g3g1g2.ntri, true_ntri_212)
    np.testing.assert_array_equal(gggc.g3g2g1.ntri, true_ntri_221)
    np.testing.assert_allclose(gggc.g1g2g3.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam1, true_gam1_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam1, true_gam1_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam1, true_gam1_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam1, true_gam1_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam1, true_gam1_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam1, true_gam1_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam2, true_gam2_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam2, true_gam2_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam2, true_gam2_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam2, true_gam2_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam2, true_gam2_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam2, true_gam2_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam3, true_gam3_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam3, true_gam3_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam3, true_gam3_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam3, true_gam3_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam3, true_gam3_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam3, true_gam3_221, rtol=1.e-5)

    # Repeat with binslop = 0
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1)
    ggg.process(cat1, cat2)
    #print('binslop > 0: ggg.ntri = ',ggg.ntri)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)

    # And again with no top-level recursion
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0)
    ggg.process(cat1, cat2)
    #print('max_top = 0: ggg.ntri = ',ggg.ntri)
    #print('true_ntri = ',true_ntri_sum)
    #print('diff = ',ggg.ntri - true_ntri_sum)
    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)

    # Split into patches to test the list-based version of the code.
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=3)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, npatch=3)

    ggg.process(cat1, cat2, num_threads=2)

    np.testing.assert_array_equal(ggg.ntri, true_ntri_sum)
    np.testing.assert_allclose(ggg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam0, true_gam0_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam1, true_gam1_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam2, true_gam2_sum, rtol=1.e-5)
    np.testing.assert_allclose(ggg.gam3, true_gam3_sum, rtol=1.e-5)

    gggc = treecorr.GGGCrossCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                        min_u=min_u, max_u=max_u, nubins=nubins,
                                        min_v=min_v, max_v=max_v, nvbins=nvbins,
                                        bin_slop=0, verbose=1)
    gggc.process(cat1, cat2)

    np.testing.assert_array_equal(gggc.g1g2g3.ntri, true_ntri_122)
    np.testing.assert_array_equal(gggc.g1g3g2.ntri, true_ntri_122)
    np.testing.assert_array_equal(gggc.g2g1g3.ntri, true_ntri_212)
    np.testing.assert_array_equal(gggc.g2g3g1.ntri, true_ntri_221)
    np.testing.assert_array_equal(gggc.g3g1g2.ntri, true_ntri_212)
    np.testing.assert_array_equal(gggc.g3g2g1.ntri, true_ntri_221)
    np.testing.assert_allclose(gggc.g1g2g3.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam1, true_gam1_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam1, true_gam1_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam1, true_gam1_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam1, true_gam1_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam1, true_gam1_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam1, true_gam1_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam2, true_gam2_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam2, true_gam2_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam2, true_gam2_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam2, true_gam2_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam2, true_gam2_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam2, true_gam2_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g2g3.gam3, true_gam3_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g1g3g2.gam3, true_gam3_122, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g1g3.gam3, true_gam3_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g2g3g1.gam3, true_gam3_221, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g1g2.gam3, true_gam3_212, rtol=1.e-5)
    np.testing.assert_allclose(gggc.g3g2g1.gam3, true_gam3_221, rtol=1.e-5)


@timer
def test_ggg():
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    #
    # Rather than go through the bispectrum, I found it easier to just directly do the
    # integral:
    #
    # Gamma0 = int(int( g(x+x1,y+y1) g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1-Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    # where the positions are measured relative to the centroid (x,y).
    # If we call the positions relative to the centroid:
    #    q1 = x1 + I y1
    #    q2 = x2 + I y2
    #    q3 = -(x1+x2) - I (y1+y2)
    # then the result comes out as
    #
    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    #
    # The other three are a bit more complicated.
    #
    # Gamma1 = int(int( g(x+x1,y+y1)* g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1+Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    #        = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    #
    # Gamm2 and Gamma3 are found from cyclic rotations of q1,q2,q3.

    gamma0 = 0.05
    r0 = 10.
    if __name__ == '__main__':
        ngal = 200000
        L = 30.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        ngal = 10000
        L = 20.*r0
        tol_factor = 5
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    min_sep = 11.
    max_sep = 15.
    nbins = 3
    min_u = 0.7
    max_u = 1.0
    nubins = 3
    min_v = 0.1
    max_v = 0.3
    nvbins = 2

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    ggg.process(cat)

    # Using bin_size=None rather than omiting bin_size is equivalent.
    ggg2 = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_size=None,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    ggg2.process(cat, num_threads=1)
    ggg.process(cat, num_threads=1)
    assert ggg2 == ggg

    # log(<d>) != <logd>, but it should be close:
    print('meanlogd1 - log(meand1) = ',ggg.meanlogd1 - np.log(ggg.meand1))
    print('meanlogd2 - log(meand2) = ',ggg.meanlogd2 - np.log(ggg.meand2))
    print('meanlogd3 - log(meand3) = ',ggg.meanlogd3 - np.log(ggg.meand3))
    print('meand3 / meand2 = ',ggg.meand3 / ggg.meand2)
    print('meanu = ',ggg.meanu)
    print('max diff = ',np.max(np.abs(ggg.meand3/ggg.meand2 -ggg.meanu)))
    print('max rel diff = ',np.max(np.abs((ggg.meand3/ggg.meand2 -ggg.meanu)/ggg.meanu)))
    print('(meand1 - meand2)/meand3 = ',(ggg.meand1-ggg.meand2) / ggg.meand3)
    print('meanv = ',ggg.meanv)
    print('max diff = ',np.max(np.abs((ggg.meand1-ggg.meand2)/ggg.meand3 -np.abs(ggg.meanv))))
    print('max rel diff = ',np.max(np.abs(((ggg.meand1-ggg.meand2)/ggg.meand3-np.abs(ggg.meanv))
                                          / ggg.meanv)))
    np.testing.assert_allclose(ggg.meanlogd1, np.log(ggg.meand1), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd2, np.log(ggg.meand2), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd3, np.log(ggg.meand3), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meand3/ggg.meand2, ggg.meanu, rtol=1.e-5 * tol_factor)
    np.testing.assert_allclose((ggg.meand1-ggg.meand2)/ggg.meand3, np.abs(ggg.meanv),
                               rtol=1.e-5 * tol_factor, atol=1.e-5 * tol_factor)
    np.testing.assert_allclose(ggg.meanlogd3-ggg.meanlogd2, np.log(ggg.meanu),
                               atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(np.log(ggg.meand1-ggg.meand2)-ggg.meanlogd3,
                               np.log(np.abs(ggg.meanv)), atol=2.e-3 * tol_factor)

    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3
    #print('rnom = ',np.exp(ggg.logr))
    #print('unom = ',ggg.u)
    #print('vnom = ',ggg.v)
    #print('d1 = ',d1)
    #print('d2 = ',d2)
    #print('d3 = ',d3)

    # For q1,q2,q3, we can choose an orientation where c1 is at the origin, and d2 is horizontal.
    # Then let s be the "complex vector" from c1 to c3, which is just real.
    s = d2
    # And let t be from c1 to c2. t = |t| e^Iphi
    # |t| = d3
    # cos(phi) = (d2^2+d3^2-d1^2)/(2d2 d3)
    # |t| cos(phi) = (d2^2+d3^2-d1^2)/2d2
    # |t| sin(phi) = sqrt(|t|^2 - (|t|cos(phi))^2)
    tx = (d2**2 + d3**2 - d1**2)/(2.*d2)
    ty = np.sqrt(d3**2 - tx**2)
    # As arranged, if ty > 0, points 1,2,3 are clockwise, which is negative v.
    # So for bins with positive v, we need to flip the direction of ty.
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty

    q1 = (s + t)/3.
    q2 = q1 - t
    q3 = q1 - s
    nq1 = np.abs(q1)**2
    nq2 = np.abs(q2)**2
    nq3 = np.abs(q3)**2
    #print('q1 = ',q1)
    #print('q2 = ',q2)
    #print('q3 = ',q3)

    # The L^2 term in the denominator of true_gam* is the area over which the integral is done.
    # Since the centers of the triangles don't go to the edge of the box, we approximate the
    # correct area by subtracting off 2*mean(qi) from L, which should give a slightly better
    # estimate of the correct area to use here.  (We used 2d2 for the kkk calculation, but this
    # is probably a slightly better estimate of the distance the triangles can get to the edges.)
    L = L - 2.*(q1 + q2 + q3)/3.

    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    # Gamma1 = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    true_gam1 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                  + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                     (2.*q1**2 - q2**2 - q3**2)) ))

    true_gam2 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                  + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                     (2.*q2**2 - q1**2 - q3**2)) ))

    true_gam3 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                  + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                     (2.*q3**2 - q1**2 - q2**2)) ))

    print('ntri = ',ggg.ntri)
    print('gam0 = ',ggg.gam0)
    print('true_gam0 = ',true_gam0)
    print('ratio = ',ggg.gam0 / true_gam0)
    print('diff = ',ggg.gam0 - true_gam0)
    print('max rel diff = ',np.max(np.abs((ggg.gam0 - true_gam0)/true_gam0)))
    # The Gamma0 term is a bit worse than the others.  The accurracy improves as I increase the
    # number of objects, so I think it's just because of the smallish number of galaxies being
    # not super accurate.
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=0.2 * tol_factor, atol=1.e-7)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam0)),
                               np.log(np.abs(true_gam0)), atol=0.2 * tol_factor)

    print('gam1 = ',ggg.gam1)
    print('true_gam1 = ',true_gam1)
    print('ratio = ',ggg.gam1 / true_gam1)
    print('diff = ',ggg.gam1 - true_gam1)
    print('max rel diff = ',np.max(np.abs((ggg.gam1 - true_gam1)/true_gam1)))
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam1)),
                               np.log(np.abs(true_gam1)), atol=0.1 * tol_factor)

    print('gam2 = ',ggg.gam2)
    print('true_gam2 = ',true_gam2)
    print('ratio = ',ggg.gam2 / true_gam2)
    print('diff = ',ggg.gam2 - true_gam2)
    print('max rel diff = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=0.1 * tol_factor)
    print('max rel diff for log = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(np.log(np.abs(ggg.gam2)),
                               np.log(np.abs(true_gam2)), atol=0.1 * tol_factor)

    print('gam3 = ',ggg.gam3)
    print('true_gam3 = ',true_gam3)
    print('ratio = ',ggg.gam3 / true_gam3)
    print('diff = ',ggg.gam3 - true_gam3)
    print('max rel diff = ',np.max(np.abs((ggg.gam3 - true_gam3)/true_gam3)))
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam3)),
                               np.log(np.abs(true_gam3)), atol=0.1 * tol_factor)

    # We check the accuracy of the Map3 calculation below in test_map3.
    # Here we just check that it runs, round trips correctly through an output file,
    # and gives the same answer when run through corr3.

    map3_stats = ggg.calculateMap3()
    map3 = map3_stats[0]
    mx3 = map3_stats[7]
    print('mapsq = ',map3)
    print('mxsq = ',mx3)

    map3_file = 'output/ggg_m3.txt'
    ggg.writeMap3(map3_file, precision=16)
    data = np.genfromtxt(os.path.join('output','ggg_m3.txt'), names=True)
    np.testing.assert_allclose(data['Map3'], map3)
    np.testing.assert_allclose(data['Mx3'], mx3)

    # Check that we get the same result using the corr3 function:
    cat.write(os.path.join('data','ggg_data.dat'))
    config = treecorr.config.read_config('configs/ggg.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','ggg.out'), names=True, skip_header=1)
    np.testing.assert_allclose(corr3_output['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Check m3 output
    corr3_output2 = np.genfromtxt(os.path.join('output','ggg_m3.out'), names=True)
    print('map3 = ',map3)
    print('from corr3 output = ',corr3_output2['Map3'])
    print('ratio = ',corr3_output2['Map3']/map3)
    print('diff = ',corr3_output2['Map3']-map3)
    np.testing.assert_allclose(corr3_output2['Map3'], map3, rtol=1.e-4)

    print('mx3 = ',mx3)
    print('from corr3 output = ',corr3_output2['Mx3'])
    print('ratio = ',corr3_output2['Mx3']/mx3)
    print('diff = ',corr3_output2['Mx3']-mx3)
    np.testing.assert_allclose(corr3_output2['Mx3'], mx3, rtol=1.e-4)

    # OK to have m3 output, but not ggg
    del config['ggg_file_name']
    treecorr.corr3(config)
    corr3_output2 = np.genfromtxt(os.path.join('output','ggg_m3.out'), names=True)
    np.testing.assert_allclose(corr3_output2['Map3'], map3, rtol=1.e-4)
    np.testing.assert_allclose(corr3_output2['Mx3'], mx3, rtol=1.e-4)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','ggg_out1.fits')
        ggg.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(ggg.logr).flatten())
        np.testing.assert_almost_equal(data['u_nom'], ggg.u.flatten())
        np.testing.assert_almost_equal(data['v_nom'], ggg.v.flatten())
        np.testing.assert_almost_equal(data['meand1'], ggg.meand1.flatten())
        np.testing.assert_almost_equal(data['meanlogd1'], ggg.meanlogd1.flatten())
        np.testing.assert_almost_equal(data['meand2'], ggg.meand2.flatten())
        np.testing.assert_almost_equal(data['meanlogd2'], ggg.meanlogd2.flatten())
        np.testing.assert_almost_equal(data['meand3'], ggg.meand3.flatten())
        np.testing.assert_almost_equal(data['meanlogd3'], ggg.meanlogd3.flatten())
        np.testing.assert_almost_equal(data['meanu'], ggg.meanu.flatten())
        np.testing.assert_almost_equal(data['meanv'], ggg.meanv.flatten())
        np.testing.assert_almost_equal(data['gam0r'], ggg.gam0.real.flatten())
        np.testing.assert_almost_equal(data['gam1r'], ggg.gam1.real.flatten())
        np.testing.assert_almost_equal(data['gam2r'], ggg.gam2.real.flatten())
        np.testing.assert_almost_equal(data['gam3r'], ggg.gam3.real.flatten())
        np.testing.assert_almost_equal(data['gam0i'], ggg.gam0.imag.flatten())
        np.testing.assert_almost_equal(data['gam1i'], ggg.gam1.imag.flatten())
        np.testing.assert_almost_equal(data['gam2i'], ggg.gam2.imag.flatten())
        np.testing.assert_almost_equal(data['gam3i'], ggg.gam3.imag.flatten())
        np.testing.assert_almost_equal(data['sigma_gam0'], np.sqrt(ggg.vargam0.flatten()))
        np.testing.assert_almost_equal(data['sigma_gam1'], np.sqrt(ggg.vargam1.flatten()))
        np.testing.assert_almost_equal(data['sigma_gam2'], np.sqrt(ggg.vargam2.flatten()))
        np.testing.assert_almost_equal(data['sigma_gam3'], np.sqrt(ggg.vargam3.flatten()))
        np.testing.assert_almost_equal(data['weight'], ggg.weight.flatten())
        np.testing.assert_almost_equal(data['ntri'], ggg.ntri.flatten())

        # Check the read function
        # Note: These don't need the flatten.
        # The read function should reshape them to the right shape.
        ggg2 = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=1)
        ggg2.read(out_file_name1)
        np.testing.assert_almost_equal(ggg2.logr, ggg.logr)
        np.testing.assert_almost_equal(ggg2.u, ggg.u)
        np.testing.assert_almost_equal(ggg2.v, ggg.v)
        np.testing.assert_almost_equal(ggg2.meand1, ggg.meand1)
        np.testing.assert_almost_equal(ggg2.meanlogd1, ggg.meanlogd1)
        np.testing.assert_almost_equal(ggg2.meand2, ggg.meand2)
        np.testing.assert_almost_equal(ggg2.meanlogd2, ggg.meanlogd2)
        np.testing.assert_almost_equal(ggg2.meand3, ggg.meand3)
        np.testing.assert_almost_equal(ggg2.meanlogd3, ggg.meanlogd3)
        np.testing.assert_almost_equal(ggg2.meanu, ggg.meanu)
        np.testing.assert_almost_equal(ggg2.meanv, ggg.meanv)
        np.testing.assert_almost_equal(ggg2.gam0, ggg.gam0)
        np.testing.assert_almost_equal(ggg2.gam1, ggg.gam1)
        np.testing.assert_almost_equal(ggg2.gam2, ggg.gam2)
        np.testing.assert_almost_equal(ggg2.gam3, ggg.gam3)
        np.testing.assert_almost_equal(ggg2.vargam0, ggg.vargam0)
        np.testing.assert_almost_equal(ggg2.vargam1, ggg.vargam1)
        np.testing.assert_almost_equal(ggg2.vargam2, ggg.vargam2)
        np.testing.assert_almost_equal(ggg2.vargam3, ggg.vargam3)
        np.testing.assert_almost_equal(ggg2.weight, ggg.weight)
        np.testing.assert_almost_equal(ggg2.ntri, ggg.ntri)
        assert ggg2.coords == ggg.coords
        assert ggg2.metric == ggg.metric
        assert ggg2.sep_units == ggg.sep_units
        assert ggg2.bin_type == ggg.bin_type


@timer
def test_map3():
    # Use the same gamma(r) as in test_gg.
    # This time, rather than use a smaller catalog in the nosetests run, we skip the run
    # in that case and just read in the output file.  This way we can test the Map^2 formulae
    # on the more precise output.
    # The code to make the output file is present here, but disabled normally.

    gamma0 = 0.05
    r0 = 10.
    L = 20.*r0
    cat_name = os.path.join('data','ggg_map.dat')
    out_name = os.path.join('data','ggg_map.out')
    ggg = treecorr.GGGCorrelation(bin_size=0.1, min_sep=1, nbins=47, verbose=2)

    # This takes a few hours to run, so be careful about enabling this.
    if False:
        ngal = 100000

        rng = np.random.RandomState(8675309)
        x = (rng.random_sample(ngal)-0.5) * L
        y = (rng.random_sample(ngal)-0.5) * L
        r2 = (x**2 + y**2)/r0**2
        g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
        g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, verbose=2)
        cat.write(cat_name)
        t0 = time.time()
        ggg.process(cat)
        t1 = time.time()
        print('time for ggg.process = ',t1-t0)
        ggg.write(out_name, precision=16)
    else:
        ggg.read(out_name)

    # Before we check the computed 3pt correlation function, let's use the perfect
    # values for gam0, gam1, gam2, gam3 given the measured mean d1,d2,d3 in each bin.
    # cf. comments in test_ggg above for the math here.
    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3
    u = ggg.meanu
    v = ggg.meanv
    s = d2
    tx = d3*(0.5*u*(1-v**2) - np.abs(v))
    ty = np.sqrt(d3**2 - tx**2)
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty

    q1 = (s + t)/3.
    q2 = q1 - t
    q3 = q1 - s
    nq1 = np.abs(q1)**2
    nq2 = np.abs(q2)**2
    nq3 = np.abs(q3)**2

    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    true_gam1 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                  + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                     (2.*q1**2 - q2**2 - q3**2)) ))

    true_gam2 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                  + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                     (2.*q2**2 - q1**2 - q3**2)) ))

    true_gam3 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                  + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                     (2.*q3**2 - q1**2 - q2**2)) ))

    ggg.gam0r = true_gam0.real
    ggg.gam1r = true_gam1.real
    ggg.gam2r = true_gam2.real
    ggg.gam3r = true_gam3.real
    ggg.gam0i = true_gam0.imag
    ggg.gam1i = true_gam1.imag
    ggg.gam2i = true_gam2.imag
    ggg.gam3i = true_gam3.imag

    # Directly calculate Map(u,v) across the region as:
    # Map(u,v) = int( g(x,y) * ((u-x) -I(v-y))^2 / ((u-x)^2 + (v-y)^2) * Q(u-x, v-y) )
    #          = 1/2 gamma0 r0^4 R^2 / (R^2+r0^2)^5 x
    #                 ((u^2+v^2)^2 - 8 (u^2+v^2) (R^2+r0^2) + 8 (R^2+r0^2)^2) x
    #                 exp(-1/2 (u^2+v^2) / (R^2+r0^2))
    # Then, directly compute <Map^3>:
    # <Map^3> = int(Map(u,v)^3, u=-inf..inf, v=-inf..inf) / L^2
    #         = 2816/243 pi gamma0^3 r0^12 R^6 / (r0^2+R^2)^8 / L^2

    r = ggg.rnom1d
    true_map3 = 2816./243. *np.pi * gamma0**3 * r0**12 * r**6 / (L**2 * (r**2+r0**2)**8)

    ggg_map3 = ggg.calculateMap3()
    map3 = ggg_map3[0]
    print('map3 = ',map3)
    print('true_map3 = ',true_map3)
    print('ratio = ',map3/true_map3)
    print('diff = ',map3-true_map3)
    print('max diff = ',max(abs(map3 - true_map3)))
    np.testing.assert_allclose(map3, true_map3, rtol=0.05, atol=5.e-10)
    for mx in ggg_map3[1:-1]:  # Last one is var, so don't include that.
        #print('mx = ',mx)
        #print('max = ',max(abs(mx)))
        np.testing.assert_allclose(mx, 0., atol=5.e-10)

    # Next check the same calculation, but not with k2=k3=1.
    # Setting these to 1 + 1.e-15 should give basically the same answer,
    # but use different formulae (SKL, rather than JBJ).
    ggg_map3b = ggg.calculateMap3(k2=1+1.e-15, k3=1+1.e-15)
    map3b = ggg_map3b[0]
    print('map3b = ',map3b)
    print('ratio = ',map3b/map3)
    print('diff = ',map3b-map3)
    print('max diff = ',max(abs(map3b - map3)))
    np.testing.assert_allclose(ggg_map3b, ggg_map3, rtol=1.e-15, atol=1.e-20)

    # Other combinations are possible to compute analytically as well, so try out a couple
    ggg_map3c = ggg.calculateMap3(k2=1, k3=2)
    map3c = ggg_map3c[0]
    true_map3c = 1024./243.*np.pi * gamma0**3 * r0**12 * r**6
    true_map3c *= (575*r**8 + 806*r**6*r0**2 + 438*r**4*r0**4 + 110*r0**6*r**2 + 11*r0**8)
    true_map3c /= L**2 * (3*r**2+r0**2)**7 * (r**2+r0**2)**5
    print('map3c = ',map3c)
    print('true_map3 = ',true_map3c)
    print('ratio = ',map3c/true_map3c)
    print('diff = ',map3c-true_map3c)
    print('max diff = ',max(abs(map3c - true_map3c)))
    np.testing.assert_allclose(map3c, true_map3c, rtol=0.1, atol=1.e-9)

    # (This is the same but in a different order)
    ggg_map3d = np.array(ggg.calculateMap3(k2=2, k3=1))
    np.testing.assert_allclose(ggg_map3d[[0,2,1,3,5,4,6,7,8]], ggg_map3c)

    ggg_map3e = ggg.calculateMap3(k2=1.5, k3=2)
    map3e = ggg_map3e[0]
    true_map3e = 442368.*np.pi * gamma0**3 * r0**12 * r**6
    true_map3e *= (1800965*r**12 + 4392108*r**10*r0**2 + 4467940*r**8*r0**4 + 2429536*r**6*r0**6 +
                   745312*r**4*r0**8 + 122496*r0**10*r**2 + 8448*r0**12)
    true_map3e /= L**2 * (61*r**4 + 58*r0**2*r**2 + 12*r0**4)**7
    print('map3e = ',map3e)
    print('true_map3 = ',true_map3e)
    print('ratio = ',map3e/true_map3e)
    print('diff = ',map3e-true_map3e)
    print('max diff = ',max(abs(map3e - true_map3e)))
    np.testing.assert_allclose(map3e, true_map3e, rtol=0.1, atol=1.e-9)

    # Repeat now with the real ggg results.  The tolerance needs to be a bit larger now.
    # We also increase the "true" value a bit because the effective L^2 is a bit lower.
    # I could make a hand wavy justification for this factor, but it's actually just an
    # empirical observation of about how much too large the prediction is in practice.
    true_map3 *= (1 + 4*r0/L)

    ggg.read(out_name)
    ggg_map3 = ggg.calculateMap3()
    map3, mapmapmx, mapmxmap, mxmapmap, mxmxmap, mxmapmx, mapmxmx, mx3, var = ggg_map3
    print('R = ',ggg.rnom1d)
    print('map3 = ',map3)
    print('true_map3 = ',true_map3)
    print('ratio = ',map3/true_map3)
    print('diff = ',map3-true_map3)
    print('max diff = ',max(abs(map3 - true_map3)))
    np.testing.assert_allclose(map3, true_map3, rtol=0.1, atol=2.e-9)
    for mx in (mapmapmx, mapmxmap, mxmapmap, mxmxmap, mxmapmx, mapmxmx, mx3):
        #print('mx = ',mx)
        #print('max = ',max(abs(mx)))
        np.testing.assert_allclose(mx, 0., atol=2.e-9)

    map3_file = 'output/ggg_m3.txt'
    ggg.writeMap3(map3_file, precision=16)
    data = np.genfromtxt(os.path.join('output','ggg_m3.txt'), names=True)
    np.testing.assert_allclose(data['Map3'], map3)
    np.testing.assert_allclose(data['Mx3'], mx3)

    # We can also just target the range where we expect good results.
    mask = (ggg.rnom1d > 5) & (ggg.rnom1d < 30)
    R = ggg.rnom1d[mask]
    map3 = ggg.calculateMap3(R=R)[0]
    print('R = ',R)
    print('map3 = ',map3)
    print('true_map3 = ',true_map3[mask])
    print('ratio = ',map3/true_map3[mask])
    print('diff = ',map3-true_map3[mask])
    print('max diff = ',max(abs(map3 - true_map3[mask])))
    np.testing.assert_allclose(map3, true_map3[mask], rtol=0.1)

    map3_file = 'output/ggg_m3b.txt'
    ggg.writeMap3(map3_file, R=R, precision=16)
    data = np.genfromtxt(os.path.join('output','ggg_m3b.txt'), names=True)
    np.testing.assert_allclose(data['Map3'], map3)

    # Finally add some tests where the B-mode is expected to be non-zero.
    # The easiest way to do this is to have gamma0 be complex.
    # Then the real part drives the E-mode, and the imaginary part the B-mode.
    # Note: The following tests detect the error in the code that was corrected by
    #       Laila Linke, which she and Sven Heydenreich found by comparing to direct
    #       Map^3 measurements of the Millenium simulation.  cf. PR #128.
    #       The code prior to Laila's fix fails these tests.
    gamma0 = 0.03 + 0.04j
    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    true_gam1 = ((-2.*np.pi * gamma0 * np.abs(gamma0)**2)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                  + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                     (2.*q1**2 - q2**2 - q3**2)) ))

    true_gam2 = ((-2.*np.pi * gamma0 * np.abs(gamma0)**2)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                  + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                     (2.*q2**2 - q1**2 - q3**2)) ))

    true_gam3 = ((-2.*np.pi * gamma0 * np.abs(gamma0)**2)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                  + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                     (2.*q3**2 - q1**2 - q2**2)) ))

    temp = 2816./243. * np.pi * r0**12 * r**6 / (L**2 * (r**2+r0**2)**8)
    true_map3 = temp * gamma0.real**3
    true_map2mx = temp * gamma0.real**2 * gamma0.imag
    true_mapmx2 = temp * gamma0.real * gamma0.imag**2
    true_mx3 = temp * gamma0.imag**3

    ggg.gam0r = true_gam0.real
    ggg.gam1r = true_gam1.real
    ggg.gam2r = true_gam2.real
    ggg.gam3r = true_gam3.real
    ggg.gam0i = true_gam0.imag
    ggg.gam1i = true_gam1.imag
    ggg.gam2i = true_gam2.imag
    ggg.gam3i = true_gam3.imag

    ggg_map3 = ggg.calculateMap3()

    # The E-mode is unchanged from before.
    map3 = ggg_map3[0]
    print('map3 = ',map3)
    print('true_map3 = ',true_map3)
    print('ratio = ',map3/true_map3)
    print('diff = ',map3-true_map3)
    print('max diff = ',max(abs(map3 - true_map3)))
    np.testing.assert_allclose(map3, true_map3, rtol=0.05, atol=5.e-10)

    # But now the rest of them are non-zero.
    for map2mx in ggg_map3[1:4]:
        print('map2mx = ',map2mx)
        print('true_map2mx = ',true_map2mx)
        print('ratio = ',map2mx/true_map2mx)
        print('diff = ',map2mx-true_map2mx)
        print('max diff = ',max(abs(map2mx - true_map2mx)))
        np.testing.assert_allclose(map2mx, true_map2mx, rtol=0.05, atol=5.e-10)
    for mapmx2 in ggg_map3[4:7]:
        print('mapmx2 = ',mapmx2)
        print('true_mapmx2 = ',true_mapmx2)
        print('ratio = ',mapmx2/true_mapmx2)
        print('diff = ',mapmx2-true_mapmx2)
        print('max diff = ',max(abs(mapmx2 - true_mapmx2)))
        np.testing.assert_allclose(mapmx2, true_mapmx2, rtol=0.05, atol=5.e-10)
    mx3 = ggg_map3[7]
    print('mx3 = ',mx3)
    print('true_mx3 = ',true_mx3)
    print('ratio = ',mx3/true_mx3)
    print('diff = ',mx3-true_mx3)
    print('max diff = ',max(abs(mx3 - true_mx3)))
    np.testing.assert_allclose(mx3, true_mx3, rtol=0.05, atol=5.e-10)

    # Now k = 1,1,2
    temp = 1024./243.*np.pi * r0**12 * r**6
    temp *= (575*r**8 + 806*r**6*r0**2 + 438*r**4*r0**4 + 110*r0**6*r**2 + 11*r0**8)
    temp /= L**2 * (3*r**2+r0**2)**7 * (r**2+r0**2)**5
    true_map3c = temp * gamma0.real**3
    true_map2mxc = temp * gamma0.real**2 * gamma0.imag
    true_mapmx2c = temp * gamma0.real * gamma0.imag**2
    true_mx3c = temp * gamma0.imag**3

    ggg_map3c = ggg.calculateMap3(k2=1, k3=2)
    map3c = ggg_map3c[0]
    print('map3c = ',map3c)
    print('true_map3 = ',true_map3c)
    print('ratio = ',map3c/true_map3c)
    print('diff = ',map3c-true_map3c)
    print('max diff = ',max(abs(map3c - true_map3c)))
    np.testing.assert_allclose(map3c, true_map3c, rtol=0.1, atol=1.e-9)
    for map2mx in ggg_map3c[1:4]:
        print('map2mx = ',map2mx)
        print('true_map2mx = ',true_map2mxc)
        print('ratio = ',map2mx/true_map2mxc)
        print('diff = ',map2mx-true_map2mxc)
        print('max diff = ',max(abs(map2mx - true_map2mxc)))
        np.testing.assert_allclose(map2mx, true_map2mxc, rtol=0.05, atol=1.e-9)
    for mapmx2 in ggg_map3c[4:7]:
        print('mapmx2 = ',mapmx2)
        print('true_mapmx2 = ',true_mapmx2c)
        print('ratio = ',mapmx2/true_mapmx2c)
        print('diff = ',mapmx2-true_mapmx2c)
        print('max diff = ',max(abs(mapmx2 - true_mapmx2c)))
        np.testing.assert_allclose(mapmx2, true_mapmx2c, rtol=0.05, atol=1.e-9)
    mx3 = ggg_map3c[7]
    print('mx3 = ',mx3)
    print('true_mx3 = ',true_mx3c)
    print('ratio = ',mx3/true_mx3c)
    print('diff = ',mx3-true_mx3c)
    print('max diff = ',max(abs(mx3 - true_mx3c)))
    np.testing.assert_allclose(mx3, true_mx3c, rtol=0.05, atol=1.e-9)

    # (This is the same but in a different order)
    ggg_map3d = np.array(ggg.calculateMap3(k2=2, k3=1))
    np.testing.assert_allclose(ggg_map3d[[0,2,1,3,5,4,6,7,8]], ggg_map3c)

    ggg_map3e = ggg.calculateMap3(k2=1.5, k3=2)
    temp = 442368.*np.pi * r0**12 * r**6
    temp *= (1800965*r**12 + 4392108*r**10*r0**2 + 4467940*r**8*r0**4 + 2429536*r**6*r0**6 +
                   745312*r**4*r0**8 + 122496*r0**10*r**2 + 8448*r0**12)
    temp /= L**2 * (61*r**4 + 58*r0**2*r**2 + 12*r0**4)**7
    true_map3e = temp * gamma0.real**3
    true_map2mxe = temp * gamma0.real**2 * gamma0.imag
    true_mapmx2e = temp * gamma0.real * gamma0.imag**2
    true_mx3e = temp * gamma0.imag**3

    map3e = ggg_map3e[0]
    print('map3e = ',map3e)
    print('true_map3 = ',true_map3e)
    print('ratio = ',map3e/true_map3e)
    print('diff = ',map3e-true_map3e)
    print('max diff = ',max(abs(map3e - true_map3e)))
    np.testing.assert_allclose(map3e, true_map3e, rtol=0.1, atol=1.e-9)
    for map2mx in ggg_map3e[1:4]:
        print('map2mx = ',map2mx)
        print('true_map2mx = ',true_map2mxe)
        print('ratio = ',map2mx/true_map2mxe)
        print('diff = ',map2mx-true_map2mxe)
        print('max diff = ',max(abs(map2mx - true_map2mxe)))
        np.testing.assert_allclose(map2mx, true_map2mxe, rtol=0.05, atol=1.e-9)
    for mapmx2 in ggg_map3e[4:7]:
        print('mapmx2 = ',mapmx2)
        print('true_mapmx2 = ',true_mapmx2e)
        print('ratio = ',mapmx2/true_mapmx2e)
        print('diff = ',mapmx2-true_mapmx2e)
        print('max diff = ',max(abs(mapmx2 - true_mapmx2e)))
        np.testing.assert_allclose(mapmx2, true_mapmx2e, rtol=0.05, atol=1.e-9)
    mx3 = ggg_map3e[7]
    print('mx3 = ',mx3)
    print('true_mx3 = ',true_mx3e)
    print('ratio = ',mx3/true_mx3e)
    print('diff = ',mx3-true_mx3e)
    print('max diff = ',max(abs(mx3 - true_mx3e)))
    np.testing.assert_allclose(mx3, true_mx3e, rtol=0.05, atol=1.e-9)



@timer
def test_grid():
    # Same as test_ggg, but the input is on a grid, where Sven Heydenreich reported getting nans.

    gamma0 = 0.05
    r0 = 10.
    nside = 50
    L = 20.*r0

    x,y = np.meshgrid(np.linspace(-L,L,nside), np.linspace(-L,L,nside))
    x = x.ravel()
    y = y.ravel()
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    min_sep = 0.05*r0
    max_sep = 5*r0
    nbins = 10
    nubins = 10
    nvbins = 11

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nubins=nubins, nvbins=nvbins,
                                  verbose=1)
    ggg.process(cat)

    # log(<d>) != <logd>, but it should be close:
    np.testing.assert_allclose(ggg.meanlogd1, np.log(ggg.meand1), rtol=1.e-2)
    np.testing.assert_allclose(ggg.meanlogd2, np.log(ggg.meand2), rtol=1.e-2)
    np.testing.assert_allclose(ggg.meanlogd3, np.log(ggg.meand3), rtol=1.e-2)
    np.testing.assert_allclose(ggg.meand3/ggg.meand2, ggg.meanu, atol=1.e-2)
    np.testing.assert_allclose((ggg.meand1-ggg.meand2)/ggg.meand3, np.abs(ggg.meanv), atol=1.e-2)

    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3

    # See test_ggg for comments about this derivation.
    s = d2
    tx = (d2**2 + d3**2 - d1**2)/(2.*d2)
    tysq = d3**2 - tx**2
    tysq[tysq <= 0] = 0.  # Watch out for invalid triangles that are actually just degenerate.
    ty = np.sqrt(tysq)
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty
    q1 = (s + t)/3.
    q2 = q1 - t
    q3 = q1 - s
    nq1 = np.abs(q1)**2
    nq2 = np.abs(q2)**2
    nq3 = np.abs(q3)**2
    q1[nq1==0] = 1  # With the grid, some triangles are degenerate.  Don't divide by 0.
    nq1[nq1==0] = 1
    L = L - 2.*(q1 + q2 + q3)/3.
    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )
    true_gam1 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                  + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                     (2.*q1**2 - q2**2 - q3**2)) ))
    true_gam2 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                  + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                     (2.*q2**2 - q1**2 - q3**2)) ))
    true_gam3 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                 np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                 (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                  + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                     (2.*q3**2 - q1**2 - q2**2)) ))

    # The bug report was that sometimes gam* could be nan.
    idx = np.where(np.isnan(ggg.gam0) | np.isnan(ggg.gam1) |
                   np.isnan(ggg.gam2) | np.isnan(ggg.gam2) |
                   np.isnan(true_gam0) | np.isnan(true_gam1) |
                   np.isnan(true_gam2) | np.isnan(true_gam3))
    if len(idx[0]) > 0:
        print('values where nan:')
        print('idx = ',idx)
        print('ntri = ',ggg.ntri[idx])
        print('d1 = ',ggg.meand1[idx])
        print('d2 = ',ggg.meand2[idx])
        print('d3 = ',ggg.meand3[idx])
        print('u = ',ggg.meanu[idx])
        print('v = ',ggg.meanv[idx])
        print('rnom = ',ggg.rnom[idx])
        print('unom = ',ggg.u[idx])
        print('vnom = ',ggg.v[idx])
        print('gam0 = ',ggg.gam0[idx])
        print('gam1 = ',ggg.gam1[idx])
        print('gam2 = ',ggg.gam2[idx])
        print('gam3 = ',ggg.gam3[idx])
        print('true_gam0 = ',true_gam0[idx])
        print('true_gam1 = ',true_gam1[idx])
        print('true_gam2 = ',true_gam2[idx])
        print('true_gam3 = ',true_gam3[idx])
        print('q1 = ',q1[idx])
        print('q2 = ',q2[idx])
        print('q3 = ',q3[idx])
        print('nq1 = ',nq1[idx])
        print('nq2 = ',nq2[idx])
        print('nq3 = ',nq3[idx])
    assert not np.any(np.isnan(ggg.gam0))
    assert not np.any(np.isnan(ggg.gam1))
    assert not np.any(np.isnan(ggg.gam2))
    assert not np.any(np.isnan(ggg.gam3))
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=0.2, atol=1.e-6)
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=0.2, atol=1.e-6)
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=0.2, atol=1.e-6)
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=0.2, atol=1.e-6)

    map3_stats = ggg.calculateMap3()
    map3 = map3_stats[0]
    mx3 = map3_stats[7]
    assert not np.any(np.isnan(map3))
    assert not np.any(np.isnan(mx3))


@timer
def test_vargam():
    # Test that the shot noise estimate of vargam is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 300
    nruns = 50000

    file_name = 'data/test_vargam_ggg.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_gggs = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
            g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
            # This time, add some shape noise (different each run).
            g1 += rng.normal(0, 0.3, size=ngal)
            g2 += rng.normal(0, 0.3, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
            ggg = treecorr.GGGCorrelation(bin_size=0.5, min_sep=30., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1)
            ggg.process(cat)
            all_gggs.append(ggg)

        mean_gam0r = np.mean([ggg.gam0r for ggg in all_gggs], axis=0)
        mean_gam0i = np.mean([ggg.gam0i for ggg in all_gggs], axis=0)
        mean_gam1r = np.mean([ggg.gam1r for ggg in all_gggs], axis=0)
        mean_gam1i = np.mean([ggg.gam1i for ggg in all_gggs], axis=0)
        mean_gam2r = np.mean([ggg.gam2r for ggg in all_gggs], axis=0)
        mean_gam2i = np.mean([ggg.gam2i for ggg in all_gggs], axis=0)
        mean_gam3r = np.mean([ggg.gam3r for ggg in all_gggs], axis=0)
        mean_gam3i = np.mean([ggg.gam3i for ggg in all_gggs], axis=0)
        var_gam0r = np.var([ggg.gam0r for ggg in all_gggs], axis=0)
        var_gam0i = np.var([ggg.gam0i for ggg in all_gggs], axis=0)
        var_gam1r = np.var([ggg.gam1r for ggg in all_gggs], axis=0)
        var_gam1i = np.var([ggg.gam1i for ggg in all_gggs], axis=0)
        var_gam2r = np.var([ggg.gam2r for ggg in all_gggs], axis=0)
        var_gam2i = np.var([ggg.gam2i for ggg in all_gggs], axis=0)
        var_gam3r = np.var([ggg.gam3r for ggg in all_gggs], axis=0)
        var_gam3i = np.var([ggg.gam3i for ggg in all_gggs], axis=0)
        mean_vargam0 = np.mean([ggg.vargam0 for ggg in all_gggs], axis=0)
        mean_vargam1 = np.mean([ggg.vargam1 for ggg in all_gggs], axis=0)
        mean_vargam2 = np.mean([ggg.vargam2 for ggg in all_gggs], axis=0)
        mean_vargam3 = np.mean([ggg.vargam3 for ggg in all_gggs], axis=0)

        all_map = [ggg.calculateMap3() for ggg in all_gggs]
        var_map3 = np.var([m[0] for m in all_map], axis=0)
        var_mx3 = np.var([m[7] for m in all_map], axis=0)
        mean_varmap = np.mean([m[8] for m in all_map], axis=0)

        np.savez(file_name,
                 mean_gam0r=mean_gam0r, var_gam0r=var_gam0r,
                 mean_gam0i=mean_gam0i, var_gam0i=var_gam0i,
                 mean_gam1r=mean_gam1r, var_gam1r=var_gam1r,
                 mean_gam1i=mean_gam1i, var_gam1i=var_gam1i,
                 mean_gam2r=mean_gam2r, var_gam2r=var_gam2r,
                 mean_gam2i=mean_gam2i, var_gam2i=var_gam2i,
                 mean_gam3r=mean_gam3r, var_gam3r=var_gam3r,
                 mean_gam3i=mean_gam3i, var_gam3i=var_gam3i,
                 mean_vargam0=mean_vargam0,
                 mean_vargam1=mean_vargam1,
                 mean_vargam2=mean_vargam2,
                 mean_vargam3=mean_vargam3,
                 var_map3=var_map3, var_mx3=var_mx3, mean_varmap=mean_varmap)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_gam0r = data['mean_gam0r']
    mean_gam0i = data['mean_gam0i']
    var_gam0r = data['var_gam0r']
    var_gam0i = data['var_gam0i']
    mean_vargam0 = data['mean_vargam0']
    #print('mean_gam0r = ',mean_gam0r)
    #print('mean_gam0i = ',mean_gam0i)
    #print('mean_vargam0 = ',mean_vargam0)
    #print('var_gam0r = ',var_gam0r)
    #print('ratio = ',var_gam0r / mean_vargam0)
    print('max relerr for gam0r = ',np.max(np.abs((var_gam0r - mean_vargam0)/var_gam0r)))
    #print('var_gam0i = ',var_gam0i)
    #print('ratio = ',var_gam0i / mean_vargam0)
    print('max relerr for gam0i = ',np.max(np.abs((var_gam0i - mean_vargam0)/var_gam0i)))
    np.testing.assert_allclose(mean_vargam0, var_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_vargam0, var_gam0i, rtol=0.03)

    mean_gam1r = data['mean_gam1r']
    mean_gam1i = data['mean_gam1i']
    var_gam1r = data['var_gam1r']
    var_gam1i = data['var_gam1i']
    mean_vargam1 = data['mean_vargam1']
    #print('mean_gam1r = ',mean_gam1r)
    #print('mean_gam1i = ',mean_gam1i)
    #print('mean_vargam1 = ',mean_vargam1)
    #print('var_gam1r = ',var_gam1r)
    #print('ratio = ',var_gam1r / mean_vargam1)
    print('max relerr for gam1r = ',np.max(np.abs((var_gam1r - mean_vargam1)/var_gam1r)))
    #print('var_gam1i = ',var_gam1i)
    #print('ratio = ',var_gam1i / mean_vargam1)
    print('max relerr for gam1i = ',np.max(np.abs((var_gam1i - mean_vargam1)/var_gam1i)))
    np.testing.assert_allclose(mean_vargam1, var_gam1r, rtol=0.03)
    np.testing.assert_allclose(mean_vargam1, var_gam1i, rtol=0.03)

    mean_gam2r = data['mean_gam2r']
    mean_gam2i = data['mean_gam2i']
    var_gam2r = data['var_gam2r']
    var_gam2i = data['var_gam2i']
    mean_vargam2 = data['mean_vargam2']
    #print('mean_gam2r = ',mean_gam2r)
    #print('mean_gam2i = ',mean_gam2i)
    #print('mean_vargam2 = ',mean_vargam2)
    #print('var_gam2r = ',var_gam2r)
    #print('ratio = ',var_gam2r / mean_vargam2)
    print('max relerr for gam2r = ',np.max(np.abs((var_gam2r - mean_vargam2)/var_gam2r)))
    #print('var_gam2i = ',var_gam2i)
    #print('ratio = ',var_gam2i / mean_vargam2)
    print('max relerr for gam2i = ',np.max(np.abs((var_gam2i - mean_vargam2)/var_gam2i)))
    np.testing.assert_allclose(mean_vargam2, var_gam2r, rtol=0.03)
    np.testing.assert_allclose(mean_vargam2, var_gam2i, rtol=0.03)

    mean_gam3r = data['mean_gam3r']
    mean_gam3i = data['mean_gam3i']
    var_gam3r = data['var_gam3r']
    var_gam3i = data['var_gam3i']
    mean_vargam3 = data['mean_vargam3']
    #print('mean_gam3r = ',mean_gam3r)
    #print('mean_gam3i = ',mean_gam3i)
    #print('mean_vargam3 = ',mean_vargam3)
    #print('var_gam3r = ',var_gam3r)
    #print('ratio = ',var_gam3r / mean_vargam3)
    print('max relerr for gam3r = ',np.max(np.abs((var_gam3r - mean_vargam3)/var_gam3r)))
    #print('var_gam3i = ',var_gam3i)
    #print('ratio = ',var_gam3i / mean_vargam3)
    print('max relerr for gam3i = ',np.max(np.abs((var_gam3i - mean_vargam3)/var_gam3i)))
    np.testing.assert_allclose(mean_vargam3, var_gam3r, rtol=0.03)
    np.testing.assert_allclose(mean_vargam3, var_gam3i, rtol=0.03)

    var_map3 = data['var_map3']
    var_mx3 = data['var_mx3']
    mean_varmap = data['mean_varmap']
    print('mean_varmap = ',mean_varmap)
    print('var_map3 = ',var_map3)
    print('ratio = ',var_map3 / mean_varmap)
    print('max relerr for map3 = ',np.max(np.abs((var_map3 - mean_varmap)/var_map3)))
    print('var_mx3 = ',var_mx3)
    print('ratio = ',var_mx3 / mean_varmap)
    print('max relerr for mx3 = ',np.max(np.abs((var_mx3 - mean_varmap)/var_mx3)))
    np.testing.assert_allclose(mean_varmap, var_map3, rtol=0.03)
    np.testing.assert_allclose(mean_varmap, var_mx3, rtol=0.03)


    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
    g1 += rng.normal(0, 0.3, size=ngal)
    g2 += rng.normal(0, 0.3, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ggg = treecorr.GGGCorrelation(bin_size=0.5, min_sep=30., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1)
    ggg.process(cat)
    print('single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ggg.vargam0 - var_gam0r)/var_gam0r)))
    print('ratio = ',ggg.vargam0 / var_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ggg.vargam0 - var_gam0i)/var_gam0i)))
    print('ratio = ',ggg.vargam0 / var_gam0i)
    np.testing.assert_allclose(ggg.vargam0, var_gam0r, rtol=0.3)
    np.testing.assert_allclose(ggg.vargam0, var_gam0i, rtol=0.3)

    print('max relerr for gam1r = ',np.max(np.abs((ggg.vargam1 - var_gam1r)/var_gam1r)))
    print('ratio = ',ggg.vargam1 / var_gam1r)
    print('max relerr for gam1i = ',np.max(np.abs((ggg.vargam1 - var_gam1i)/var_gam1i)))
    print('ratio = ',ggg.vargam1 / var_gam1i)
    np.testing.assert_allclose(ggg.vargam1, var_gam1r, rtol=0.3)
    np.testing.assert_allclose(ggg.vargam1, var_gam1i, rtol=0.3)

    print('max relerr for gam2r = ',np.max(np.abs((ggg.vargam2 - var_gam2r)/var_gam2r)))
    print('ratio = ',ggg.vargam2 / var_gam2r)
    print('max relerr for gam2i = ',np.max(np.abs((ggg.vargam2 - var_gam2i)/var_gam2i)))
    print('ratio = ',ggg.vargam2 / var_gam2i)
    np.testing.assert_allclose(ggg.vargam2, var_gam2r, rtol=0.3)
    np.testing.assert_allclose(ggg.vargam2, var_gam2i, rtol=0.3)

    print('max relerr for gam3r = ',np.max(np.abs((ggg.vargam3 - var_gam3r)/var_gam3r)))
    print('ratio = ',ggg.vargam3 / var_gam3r)
    print('max relerr for gam3i = ',np.max(np.abs((ggg.vargam3 - var_gam3i)/var_gam3i)))
    print('ratio = ',ggg.vargam3 / var_gam3i)
    np.testing.assert_allclose(ggg.vargam3, var_gam3r, rtol=0.3)
    np.testing.assert_allclose(ggg.vargam3, var_gam3i, rtol=0.3)

    var_map = ggg.calculateMap3()[8]
    print('max relerr for map3 = ',np.max(np.abs((var_map - var_map3)/var_map3)))
    print('ratio = ',var_map / var_map3)
    np.testing.assert_allclose(var_map, var_map3, rtol=0.3)

    var_map = ggg.calculateMap3(k2=1.+1.e-13, k3=1.+2.e-13)[8]
    print('With k2,k3 != 1')
    print('max relerr for map3 = ',np.max(np.abs((var_map - var_map3)/var_map3)))
    print('ratio = ',var_map / var_map3)
    np.testing.assert_allclose(var_map, var_map3, rtol=0.3)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_direct_cross()
    test_direct_cross12()
    test_ggg()
    test_map3()
    test_grid()
    test_vargam()
