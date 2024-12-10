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

import numpy as np
import treecorr
import os
import coord
import time

from test_helper import do_pickle, assert_raises, timer, is_ccw, is_ccw_3d
from test_helper import get_from_wiki, CaptureLog


@timer
def test_direct_logruv_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(0,sig_gam, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
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

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    kkg.process(cat1, cat2, cat3)

    kgk = treecorr.KGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    kgk.process(cat1, cat3, cat2)

    gkk = treecorr.GKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    gkk.process(cat3, cat1, cat2)

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_321 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_321 = np.zeros((nrbins, nubins, 2*nvbins))
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

                # Rotate shear to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x3[k])/3.
                ceny = (y1[i] + y2[j] + y3[k])/3.

                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_123
                        true_zeta = true_zeta_123
                        true_weight = true_weight_123
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_213
                        true_zeta = true_zeta_213
                        true_weight = true_weight_213
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_231
                        true_zeta = true_zeta_231
                        true_weight = true_weight_231
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_132
                        true_zeta = true_zeta_132
                        true_weight = true_weight_132
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_312
                        true_zeta = true_zeta_312
                        true_weight = true_weight_312
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_321
                        true_zeta = true_zeta_321
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
                zeta = www * k1[i] * k2[j] * g3p

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    true_ntri_sum3 = true_ntri_123 + true_ntri_213
    true_weight_sum3 = true_weight_123 + true_weight_213
    true_zeta_sum3 = true_zeta_123 + true_zeta_213
    true_ntri_sum2 = true_ntri_132 + true_ntri_231
    true_weight_sum2 = true_weight_132 + true_weight_231
    true_zeta_sum2 = true_zeta_132 + true_zeta_231
    true_ntri_sum1 = true_ntri_312 + true_ntri_321
    true_weight_sum1 = true_weight_312 + true_weight_321
    true_zeta_sum1 = true_zeta_312 + true_zeta_321
    pos = true_weight_sum1 > 0
    true_zeta_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_zeta_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_zeta_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    n_list = [true_ntri_123, true_ntri_132, true_ntri_213, true_ntri_231,
              true_ntri_312, true_ntri_321]
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    z_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
              true_zeta_312, true_zeta_321]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)

    kkg.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_213)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-5)

    kgk.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    kgk.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_231)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-5)

    gkk.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)
    gkk.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_321)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K in 1,2
    kkg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)

    kgk.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check bin_slop=0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kkg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kkg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        kkg.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    kkg.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=False)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, patch_method='local')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, patch_method='local')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        kkg.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        kgk.process(cat1p, cat3p, cat2p, patch_method='nonlocal')
    with assert_raises(ValueError):
        gkk.process(cat3p, cat1p, cat2p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross21():
    # Check the 2-1 cross correlation

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
    k2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    kkg.process(cat2, cat2, cat1)

    kgk = treecorr.KGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    kgk.process(cat2, cat1, cat2)

    gkk = treecorr.GKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')
    gkk.process(cat1, cat2, cat2)

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_221 = np.zeros((nrbins, nubins, 2*nvbins))
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

                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_122
                        true_zeta = true_zeta_122
                        true_weight = true_weight_122
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_212
                        true_zeta = true_zeta_212
                        true_weight = true_weight_212
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_221
                        true_zeta = true_zeta_221
                        true_weight = true_weight_221
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_122
                        true_zeta = true_zeta_122
                        true_weight = true_weight_122
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_212
                        true_zeta = true_zeta_212
                        true_weight = true_weight_212
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_221
                        true_zeta = true_zeta_221
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
                zeta = www * g1p * k2[j] * k2[k]

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    pos = true_weight_221 > 0
    true_zeta_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_zeta_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_zeta_122[pos] /= true_weight_122[pos]

    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: KGK doesn't have a two-argument version.
    kkg.process(cat2, cat1)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1, cat2)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    with assert_raises(ValueError):
        kgk.process(cat2, cat1)
    with assert_raises(ValueError):
        kgk.process(cat1, cat2)
    with assert_raises(ValueError):
        kkg.process(cat1)
    with assert_raises(ValueError):
        kkg.process(cat2)
    with assert_raises(ValueError):
        kgk.process(cat1)
    with assert_raises(ValueError):
        kgk.process(cat2)
    with assert_raises(ValueError):
        gkk.process(cat1)
    with assert_raises(ValueError):
        gkk.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    kkg.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    kkg.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # And again with no top-level recursion
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kkg.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)

    kkg.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)


@timer
def notest_varzeta_logruv():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
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

    file_name = 'data/test_varzeta_kkg_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kkgs = []

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
            kkg = treecorr.KKGCorrelation(bin_size=0.5, min_sep=30., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            kkg.process(cat)
            all_kkgs.append(kkg)

        mean_zetar = np.mean([kkg.zetar for kkg in all_kkgs], axis=0)
        mean_zetai = np.mean([kkg.zetai for kkg in all_kkgs], axis=0)
        var_zetar = np.var([kkg.zetar for kkg in all_kkgs], axis=0)
        var_zetai = np.var([kkg.zetai for kkg in all_kkgs], axis=0)
        mean_varzeta = np.mean([kkg.varzeta for kkg in all_kkgs], axis=0)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_zetar = data['mean_zetar']
    mean_zetai = data['mean_zetai']
    var_zetar = data['var_zetar']
    var_zetai = data['var_zetai']
    mean_varzeta = data['mean_varzeta']
    #print('mean_zetar = ',mean_zetar)
    #print('mean_zetai = ',mean_zetai)
    #print('mean_varzeta = ',mean_varzeta)
    #print('var_zetar = ',var_zetar)
    #print('ratio = ',var_zetar / mean_varzeta)
    print('max relerr for zetar = ',np.max(np.abs((var_zetar - mean_varzeta)/var_zetar)))
    #print('var_zetai = ',var_zetai)
    #print('ratio = ',var_zetai / mean_varzeta)
    print('max relerr for zetai = ',np.max(np.abs((var_zetai - mean_varzeta)/var_zetai)))
    np.testing.assert_allclose(mean_varzeta, var_zetar, rtol=0.03)
    np.testing.assert_allclose(mean_varzeta, var_zetai, rtol=0.03)

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
    kkg = treecorr.KKGCorrelation(bin_size=0.5, min_sep=30., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')

    # Before running process, varzeta and cov area allowed, but all 0.
    np.testing.assert_array_equal(kkg.cov, 0)
    np.testing.assert_array_equal(kkg.varzeta, 0)

    kkg.process(cat)
    print('single run:')
    print('max relerr for zetar = ',np.max(np.abs((kkg.varzeta - var_zetar)/var_zetar)))
    print('ratio = ',kkg.varzeta / var_zetar)
    print('max relerr for zetai = ',np.max(np.abs((kkg.varzeta - var_zetai)/var_zetai)))
    print('ratio = ',kkg.varzeta / var_zetai)
    np.testing.assert_allclose(kkg.varzeta, var_zetar, rtol=0.3)
    np.testing.assert_allclose(kkg.varzeta, var_zetai, rtol=0.3)
    n = len(kkg.varzeta.ravel())
    np.testing.assert_allclose(kkg.cov.diagonal()[0:n], kkg.varzeta.ravel())


@timer
def notest_direct_logsas_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal).astype(np.float32)
    g1_1 = rng.normal(0,0.2, (ngal,) )
    g2_1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal).astype(np.float32)
    g1_2 = rng.normal(0,0.2, (ngal,) )
    g2_2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal).astype(np.float32)
    g1_3 = rng.normal(0,0.2, (ngal,) )
    g2_3 = rng.normal(0,0.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 10.
    nbins = 2
    nphi_bins = 2

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    kkg.process(cat1, cat2, cat3, num_threads=2, algo='triangle')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_weight_123 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_132 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_213 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_231 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_312 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_321 = np.zeros((nbins, nbins, nphi_bins))
    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    bin_size = (log_max_sep - log_min_sep) / nbins
    phi_bin_size = np.pi / nphi_bins
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d1 = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                d2 = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x3[k])/3.
                ceny = (y1[i] + y2[j] + y3[k])/3.

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w3[k]
                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                zeta = www * g1p * g2p * g3p

                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    # 123
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_123[kr2,kr3,kphi] += 1
                        true_weight_123[kr2,kr3,kphi] += www
                        true_zeta_123[kr2,kr3,kphi] += zeta

                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_132[kr3,kr2,kphi] += 1
                        true_weight_132[kr3,kr2,kphi] += www
                        true_zeta_132[kr3,kr2,kphi] += zeta

                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    # 231
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_231[kr3,kr1,kphi] += 1
                        true_weight_231[kr3,kr1,kphi] += www
                        true_zeta_231[kr3,kr1,kphi] += zeta

                    # 213
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_213[kr1,kr3,kphi] += 1
                        true_weight_213[kr1,kr3,kphi] += www
                        true_zeta_213[kr1,kr3,kphi] += zeta

                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    # 312
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_312[kr1,kr2,kphi] += 1
                        true_weight_312[kr1,kr2,kphi] += www
                        true_zeta_312[kr1,kr2,kphi] += zeta

                    # 321
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_321[kr2,kr1,kphi] += 1
                        true_weight_321[kr2,kr1,kphi] += www
                        true_zeta_321[kr2,kr1,kphi] += zeta

    n_list = [true_ntri_123, true_ntri_132, true_ntri_213, true_ntri_231,
              true_ntri_312, true_ntri_321]
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
               true_zeta_312, true_zeta_321]

    true_ntri_sum = sum(n_list)
    true_weight_sum = sum(w_list)
    true_zeta_sum = sum(g0_list)
    pos = true_weight_sum > 0
    true_zeta_sum[pos] /= true_weight_sum[pos]

    # Now normalize each one individually.
    for w,g0,g1,g2,g3 in zip(w_list, g0_list, g1_list, g2_list, g3_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]
        g2[pos] /= w[pos]
        g3[pos] /= w[pos]

    # With ordered=True we get just the ones in the given order.
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_132)
    np.testing.assert_allclose(kkg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_132, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_213)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_231)
    np.testing.assert_allclose(kkg.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_231, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_312)
    np.testing.assert_allclose(kkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_312, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_321)
    np.testing.assert_allclose(kkg.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_321, rtol=1.e-4, atol=1.e-6)

    # With ordered=False, we end up with the sum of all permutations.
    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)

    # Repeat with binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    kkg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4, atol=1.e-6)

    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)

    # And again with no top-level recursion
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    kkg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4, atol=1.e-6)

    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        kkg.process(cat1, cat3=cat3, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    kkg.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-5)


@timer
def notest_direct_logsas_cross21():
    # Check the 2-1 cross correlation
    return

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g1_1 = rng.normal(0,0.2, (ngal,) )
    g2_1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,0.2, (ngal,) )
    g2_2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 1.
    max_sep = 10.
    nbins = 10
    nphi_bins = 10

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    kkg.process(cat1, cat2, num_threads=2, algo='triangle')

    # Figure out the correct answer for each permutation
    true_ntri_112 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_211 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_121 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_112 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_211 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_121 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_weight_112 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_211 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_121 = np.zeros((nbins, nbins, nphi_bins))
    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    bin_size = (log_max_sep - log_min_sep) / nbins
    phi_bin_size = np.pi / nphi_bins
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                if j == k: continue
                d1 = np.sqrt((x1[j]-x2[k])**2 + (y1[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x1[j])**2 + (y1[i]-y1[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x1[j] + x2[k])/3.
                ceny = (y1[i] + y1[j] + y2[k])/3.

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x1[j]-cenx) - 1j*(y1[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w2[k]
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                zeta = www * g1p * g2p * g3p

                # 123
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_122[kr2,kr3,kphi] += 1
                        true_weight_122[kr2,kr3,kphi] += www
                        true_zeta_122[kr2,kr3,kphi] += zeta

                # 231
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_221[kr3,kr1,kphi] += 1
                        true_weight_221[kr3,kr1,kphi] += www
                        true_zeta_221[kr3,kr1,kphi] += zeta

                # 312
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_212[kr1,kr2,kphi] += 1
                        true_weight_212[kr1,kr2,kphi] += www
                        true_zeta_212[kr1,kr2,kphi] += zeta

    n_list = [true_ntri_122, true_ntri_212, true_ntri_221]
    w_list = [true_weight_122, true_weight_212, true_weight_221]
    g0_list = [true_zeta_122, true_zeta_212, true_zeta_221]

    true_ntri_sum = sum(n_list)
    true_weight_sum = sum(w_list)
    true_zeta_sum = sum(g0_list)
    pos = true_weight_sum > 0
    true_zeta_sum[pos] /= true_weight_sum[pos]

    # Now normalize each one individually.
    for w,g0,g1,g2,g3 in zip(w_list, g0_list, g1_list, g2_list, g3_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]
        g2[pos] /= w[pos]
        g3[pos] /= w[pos]

    np.testing.assert_array_equal(kkg.ntri, true_ntri_122)
    np.testing.assert_allclose(kkg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_212)
    np.testing.assert_allclose(kkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    # With ordered=False, we end up with the sum of all permutations.
    kkg.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)

    kkg.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_122)
    np.testing.assert_allclose(kkg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_212)
    np.testing.assert_allclose(kkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2p, cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    kkg.process(cat1p, cat2p, ordered=False, num_threads=2, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)

    kkg.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_122)
    np.testing.assert_allclose(kkg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_212)
    np.testing.assert_allclose(kkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2p, cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    kkg.process(cat1p, cat2p, ordered=False, num_threads=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum)
    np.testing.assert_allclose(kkg.weight, true_weight_sum, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum, rtol=1.e-4, atol=1.e-6)


@timer
def notest_direct_logmultipole_cross():
    # Check the cross correlation with LogMultipole
    if __name__ == '__main__':
        ngal = 100
    else:
        ngal = 50

    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(1,3, (ngal,))
    g1_1 = rng.normal(0,0.2, (ngal,))
    g2_1 = rng.normal(0,0.2, (ngal,))
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    g1_2 = rng.normal(0,0.2, (ngal,))
    g2_2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.uniform(1,3, (ngal,))
    g1_3 = rng.normal(0,0.2, (ngal,))
    g2_3 = rng.normal(0,0.2, (ngal,))
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 100.
    nbins = 5
    max_n = 20

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    kkg.process(cat1, cat2, cat3, num_threads=1)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_zeta_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_123 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_132 = np.zeros((nbins, nbins, 2*max_n+1))
    # Skip all the ones with 2 or 3 first.
    # It's too annoying to get all the shear projections right for each,
    # and the unordered code is well enough tested with KKK and NNN tests.
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d1 = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                d2 = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears according to the x projection.  See Porth et al, Figure 1.
                # g2 is projected to the line from c1 to c2.
                # g3 is projected to the line from c1 to c3.
                # g1 is projected to the average of these two lines.
                expmialpha2 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                expmialpha3 /= abs(expmialpha3)
                expmialpha1 = expmialpha2 + expmialpha3
                expmialpha1 /= abs(expmialpha1)

                www = w1[i] * w2[j] * w3[k]
                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                zeta = www * g1p * g2p * g3p

                # 123, 132
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_123[kr2,kr3,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_123[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_123[kr2,kr3,:] += 1
                    true_zeta_132[kr3,kr2,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_132[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_132[kr3,kr2,:] += 1

    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)

    # Repeat with binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)

    kkg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kkg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_132, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    # First with just one catalog with patches
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=10, rng=rng)

    kkg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kkg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_132, rtol=1.e-4)

    # Now with all 3 patched.
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1.patch_centers)
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1.patch_centers)

    kkg.process(cat1, cat2, cat3, patch_method='local')
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat1, cat3, cat2, patch_method='local')
    np.testing.assert_allclose(kkg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_132, rtol=1.e-4)

    # No tests of accuracy yet, but make sure patch-based covariance works.
    cov = kkg.estimate_cov('sample')
    cov = kkg.estimate_cov('jackknife')

    with assert_raises(ValueError):
        kkg.process(cat1, cat2, cat3, patch_method='global')


@timer
def notest_varzeta():
    # Test that varzeta, etc. are correct (or close) based on actual variance of many runs.

    # Same gamma pattern as in test_kkg().  Although the signal doesn't actually matter at all here.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(gam), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 5000
    nruns = 50000

    file_name = 'data/test_varzeta_kkg.npz'
    if not os.path.isfile(file_name):
        print(file_name)
        all_zeta = []
        all_varzeta = []

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

            cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2)
            kkg = treecorr.KKGCorrelation(nbins=2, min_sep=30., max_sep=50., nphi_bins=20)
            kkg.process(cat)
            all_zeta.append(kkg.zeta)
            all_varzeta.append(kkg.varzeta)

        mean_zeta = np.mean(all_zeta, axis=0)
        var_zetar = np.var(np.real(all_zeta), axis=0)
        var_zetai = np.var(np.imag(all_zeta), axis=0)
        mean_varzeta = np.mean(all_varzeta, axis=0)

        np.savez(file_name,
                 mean_zeta=mean_zeta, var_zetar=var_zetar,
                 var_zetai=var_zetai, mean_varzeta=mean_varzeta)

    data = np.load(file_name)
    mean_zeta = data['mean_zeta']
    var_zetar = data['var_zetar']
    var_zetai = data['var_zetai']
    mean_varzeta = data['mean_varzeta']

    print('nruns = ',nruns)
    print('mean_zeta = ',mean_zeta)
    print('mean_varzeta = ',mean_varzeta)
    print('var_zeta = ',var_zetar,var_zetai)
    print('ratio = ',mean_varzeta / var_zetar)
    print('max relerr = ',np.max(np.abs((var_zetar - mean_varzeta)/var_zetar)))
    np.testing.assert_allclose(mean_varzeta, var_zetar, rtol=0.4)
    np.testing.assert_allclose(mean_varzeta, var_zetai, rtol=0.4)

    # The shot noise variance estimate is actually pretty good everywhere except at equilateral
    # triangles.  The problem is that equilateral triangles get counted multiple times with
    # each point at the primary vertex, but all with the same value.  So they have a higher
    # actual variance than you would estimate from the nominal weight.  If we exclude those
    # triangles we get agreement at much lower rtol.
    i,j,k = np.meshgrid(range(2), range(2), range(20))
    k_eq = int(60/180 * 20)
    noneq = ~((i==j) & (k==k_eq))
    print('noneq ratio = ', mean_varzeta[noneq] / var_zetar[noneq])
    print('max relerr = ',np.max(np.abs((var_zetar[noneq] - mean_varzeta[noneq])/var_zetar[noneq])))
    np.testing.assert_allclose(mean_varzeta[noneq], var_zetar[noneq], rtol=0.1)
    np.testing.assert_allclose(mean_varzeta[noneq], var_zetai[noneq], rtol=0.1)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
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

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2)
    kkg = treecorr.KKGCorrelation(nbins=2, min_sep=30., max_sep=50., nphi_bins=20)

    # Before running process, varxi and cov area allowed, but all 0.
    np.testing.assert_array_equal(kkg.cov, 0)
    np.testing.assert_array_equal(kkg.varzeta, 0)

    kkg.process(cat)
    print('single run:')
    print('zeta ratio = ',kkg.varzeta/var_zetar)
    print('max relerr for zeta = ',np.max(np.abs((kkg.varzeta[noneq] - var_zetar[noneq])/var_zetar[noneq])))
    np.testing.assert_allclose(kkg.varzeta[noneq], var_zetar[noneq], rtol=0.1)
    np.testing.assert_allclose(kkg.varzeta[noneq], var_zetai[noneq], rtol=0.1)

    kkg.process(cat, algo='triangle')
    print('single run with algo=triangle:')
    print('zeta ratio = ',kkg.varzeta/var_zetar)
    print('max relerr for zeta = ',np.max(np.abs((kkg.varzeta[noneq] - var_zetar[noneq])/var_zetar[noneq])))
    np.testing.assert_allclose(kkg.varzeta[noneq], var_zetar[noneq], rtol=0.2)
    np.testing.assert_allclose(kkg.varzeta[noneq], var_zetai[noneq], rtol=0.2)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross21()
    test_varzeta_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross21()
    test_direct_logmultipole_cross()
    test_varzeta()
