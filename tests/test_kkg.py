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
from test_patch3pt import generate_shear_field


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

    kgk = treecorr.KGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gkk = treecorr.GKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

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
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    z_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
              true_zeta_312, true_zeta_321]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    kkg.process(cat1, cat2, cat3)
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

    # With ordered=False, we end up with the sum of both versions where G is in 3
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

    # With these, ordered=False is equivalent to the G vertex being fixed.
    kkg.process(cat1, cat2, cat3, ordered=3)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=1)
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

    # First test with just one catalog using patches
    kkg.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1p, cat3, cat2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3p, cat1, cat2)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3p, cat2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1p, cat2)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2p)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2p)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    # Now all three patched
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

    # Unordered
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

    kkg.process(cat1p, cat2p, cat3p, ordered=3)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=1)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # patch_method=local
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

    kkg.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local')
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

    kgk = treecorr.KGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gkk = treecorr.GKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

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

    kkg.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: KGK doesn't have a two-argument version.
    kkg.process(cat2, cat1)
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
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
def test_varzeta_logruv():
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
        all_kgks = []
        all_gkks = []

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
            k = gamma0 * np.exp(-r2/2.)
            g1 += rng.normal(0, 0.3, size=ngal)
            g2 += rng.normal(0, 0.3, size=ngal)
            k += rng.normal(0, 0.2, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, k=k, g1=g1, g2=g2,
                                   x_units='arcmin', y_units='arcmin')
            kkg = treecorr.KKGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            kgk = treecorr.KGKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            gkk = treecorr.GKKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            kkg.process(cat, cat)
            kgk.process(cat, cat, cat)
            gkk.process(cat, cat)
            all_kkgs.append(kkg)
            all_kgks.append(kgk)
            all_gkks.append(gkk)

        mean_kkg_zetar = np.mean([kkg.zetar for kkg in all_kkgs], axis=0)
        mean_kkg_zetai = np.mean([kkg.zetai for kkg in all_kkgs], axis=0)
        var_kkg_zetar = np.var([kkg.zetar for kkg in all_kkgs], axis=0)
        var_kkg_zetai = np.var([kkg.zetai for kkg in all_kkgs], axis=0)
        mean_kkg_varzeta = np.mean([kkg.varzeta for kkg in all_kkgs], axis=0)
        mean_kgk_zetar = np.mean([kgk.zetar for kgk in all_kgks], axis=0)
        mean_kgk_zetai = np.mean([kgk.zetai for kgk in all_kgks], axis=0)
        var_kgk_zetar = np.var([kgk.zetar for kgk in all_kgks], axis=0)
        var_kgk_zetai = np.var([kgk.zetai for kgk in all_kgks], axis=0)
        mean_kgk_varzeta = np.mean([kgk.varzeta for kgk in all_kgks], axis=0)
        mean_gkk_zetar = np.mean([gkk.zetar for gkk in all_gkks], axis=0)
        mean_gkk_zetai = np.mean([gkk.zetai for gkk in all_gkks], axis=0)
        var_gkk_zetar = np.var([gkk.zetar for gkk in all_gkks], axis=0)
        var_gkk_zetai = np.var([gkk.zetai for gkk in all_gkks], axis=0)
        mean_gkk_varzeta = np.mean([gkk.varzeta for gkk in all_gkks], axis=0)

        np.savez(file_name,
                 mean_kkg_zetar=mean_kkg_zetar,
                 mean_kkg_zetai=mean_kkg_zetai,
                 var_kkg_zetar=var_kkg_zetar,
                 var_kkg_zetai=var_kkg_zetai,
                 mean_kkg_varzeta=mean_kkg_varzeta,
                 mean_kgk_zetar=mean_kgk_zetar,
                 mean_kgk_zetai=mean_kgk_zetai,
                 var_kgk_zetar=var_kgk_zetar,
                 var_kgk_zetai=var_kgk_zetai,
                 mean_kgk_varzeta=mean_kgk_varzeta,
                 mean_gkk_zetar=mean_gkk_zetar,
                 mean_gkk_zetai=mean_gkk_zetai,
                 var_gkk_zetar=var_gkk_zetar,
                 var_gkk_zetai=var_gkk_zetai,
                 mean_gkk_varzeta=mean_gkk_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_kkg_zetar = data['mean_kkg_zetar']
    mean_kkg_zetai = data['mean_kkg_zetai']
    var_kkg_zetar = data['var_kkg_zetar']
    var_kkg_zetai = data['var_kkg_zetai']
    mean_kkg_varzeta = data['mean_kkg_varzeta']
    mean_kgk_zetar = data['mean_kgk_zetar']
    mean_kgk_zetai = data['mean_kgk_zetai']
    var_kgk_zetar = data['var_kgk_zetar']
    var_kgk_zetai = data['var_kgk_zetai']
    mean_kgk_varzeta = data['mean_kgk_varzeta']
    mean_gkk_zetar = data['mean_gkk_zetar']
    mean_gkk_zetai = data['mean_gkk_zetai']
    var_gkk_zetar = data['var_gkk_zetar']
    var_gkk_zetai = data['var_gkk_zetai']
    mean_gkk_varzeta = data['mean_gkk_varzeta']

    print('var_kkg_zetar = ',var_kkg_zetar)
    print('mean kkg_varzeta = ',mean_kkg_varzeta)
    print('ratio = ',var_kkg_zetar.ravel() / mean_kkg_varzeta.ravel())
    print('var_kgk_zetar = ',var_kgk_zetar)
    print('mean kgk_varzeta = ',mean_kgk_varzeta)
    print('ratio = ',var_kgk_zetar.ravel() / mean_kgk_varzeta.ravel())
    print('var_gkk_zetar = ',var_gkk_zetar)
    print('mean gkk_varzeta = ',mean_gkk_varzeta)
    print('ratio = ',var_gkk_zetar.ravel() / mean_gkk_varzeta.ravel())

    print('max relerr for kkg zetar = ',
          np.max(np.abs((var_kkg_zetar - mean_kkg_varzeta)/var_kkg_zetar)))
    print('max relerr for kkg zetai = ',
          np.max(np.abs((var_kkg_zetai - mean_kkg_varzeta)/var_kkg_zetai)))
    np.testing.assert_allclose(mean_kkg_varzeta, var_kkg_zetar, rtol=0.03)
    np.testing.assert_allclose(mean_kkg_varzeta, var_kkg_zetai, rtol=0.03)

    print('max relerr for kgk zetar = ',
          np.max(np.abs((var_kgk_zetar - mean_kgk_varzeta)/var_kgk_zetar)))
    print('max relerr for kgk zetai = ',
          np.max(np.abs((var_kgk_zetai - mean_kgk_varzeta)/var_kgk_zetai)))
    np.testing.assert_allclose(mean_kgk_varzeta, var_kgk_zetar, rtol=0.03)
    np.testing.assert_allclose(mean_kgk_varzeta, var_kgk_zetai, rtol=0.03)

    print('max relerr for gkk zetar = ',
          np.max(np.abs((var_gkk_zetar - mean_gkk_varzeta)/var_gkk_zetar)))
    print('max relerr for gkk zetai = ',
          np.max(np.abs((var_gkk_zetai - mean_gkk_varzeta)/var_gkk_zetai)))
    np.testing.assert_allclose(mean_gkk_varzeta, var_gkk_zetar, rtol=0.03)
    np.testing.assert_allclose(mean_gkk_varzeta, var_gkk_zetai, rtol=0.03)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
    k = gamma0 * np.exp(-r2/2.)
    g1 += rng.normal(0, 0.3, size=ngal)
    g2 += rng.normal(0, 0.3, size=ngal)
    k += rng.normal(0, 0.2, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, k=k, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kkg = treecorr.KKGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')
    kgk = treecorr.KGKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')
    gkk = treecorr.GKKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(kkg.cov, 0)
    np.testing.assert_array_equal(kkg.varzeta, 0)
    np.testing.assert_array_equal(kgk.cov, 0)
    np.testing.assert_array_equal(kgk.varzeta, 0)
    np.testing.assert_array_equal(gkk.cov, 0)
    np.testing.assert_array_equal(gkk.varzeta, 0)

    kkg.process(cat, cat)
    print('KKG single run:')
    print('max relerr for zetar = ',np.max(np.abs((kkg.varzeta - var_kkg_zetar)/var_kkg_zetar)))
    print('ratio = ',kkg.varzeta / var_kkg_zetar)
    print('max relerr for zetai = ',np.max(np.abs((kkg.varzeta - var_kkg_zetai)/var_kkg_zetai)))
    print('ratio = ',kkg.varzeta / var_kkg_zetai)
    print('var_num = ',kkg._var_num)
    print('ntri = ',kkg.ntri)
    np.testing.assert_allclose(kkg.varzeta, var_kkg_zetar, rtol=0.3)
    np.testing.assert_allclose(kkg.varzeta, var_kkg_zetai, rtol=0.3)
    np.testing.assert_allclose(kkg.cov.diagonal(), kkg.varzeta.ravel())

    kgk.process(cat, cat, cat)
    print('KGK single run:')
    print('max relerr for zetar = ',np.max(np.abs((kgk.varzeta - var_kgk_zetar)/var_kgk_zetar)))
    print('ratio = ',kgk.varzeta / var_kgk_zetar)
    print('max relerr for zetai = ',np.max(np.abs((kgk.varzeta - var_kgk_zetai)/var_kgk_zetai)))
    print('ratio = ',kgk.varzeta / var_kgk_zetai)
    np.testing.assert_allclose(kgk.varzeta, var_kgk_zetar, rtol=0.3)
    np.testing.assert_allclose(kgk.varzeta, var_kgk_zetai, rtol=0.3)
    np.testing.assert_allclose(kgk.cov.diagonal(), kgk.varzeta.ravel())

    gkk.process(cat, cat)
    print('GKK single run:')
    print('max relerr for zetar = ',np.max(np.abs((gkk.varzeta - var_gkk_zetar)/var_gkk_zetar)))
    print('ratio = ',gkk.varzeta / var_gkk_zetar)
    print('max relerr for zetai = ',np.max(np.abs((gkk.varzeta - var_gkk_zetai)/var_gkk_zetai)))
    print('ratio = ',gkk.varzeta / var_gkk_zetai)
    np.testing.assert_allclose(gkk.varzeta, var_gkk_zetar, rtol=0.3)
    np.testing.assert_allclose(gkk.varzeta, var_gkk_zetai, rtol=0.3)
    np.testing.assert_allclose(gkk.cov.diagonal(), gkk.varzeta.ravel())



@timer
def test_direct_logsas_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal).astype(np.float32)
    k1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal).astype(np.float32)
    k2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal).astype(np.float32)
    g1_3 = rng.normal(0,0.2, (ngal,) )
    g2_3 = rng.normal(0,0.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 3

    # In this test set, we use the slow triangle algorithm.
    # We'll test the multipole algorithm below.
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

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

                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w3[k]
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                zeta = www * k1[i] * k2[j] * g3p

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

                    # 132
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
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    z_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
              true_zeta_312, true_zeta_321]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    kkg.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)

    kkg.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_213)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-5)

    kgk.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)

    kgk.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_231)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-5)

    gkk.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    gkk.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_321)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where G is in 3
    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-4)

    kgk.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)

    gkk.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    kkg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    kkg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    kkg.process(cat1, cat2, cat3, ordered=3, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-4)
    kgk.process(cat1, cat3, cat2, ordered=2, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3, cat1, cat2, ordered=1, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        kkg.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        kgk.process(cat1, cat3=cat1, algo='triangle')
    with assert_raises(ValueError):
        gkk.process(cat3, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    kkg.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_123)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_132)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_312)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    kkg.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkg.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_sum3, rtol=1.e-5)
    kgk.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(kgk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_sum2, rtol=1.e-5)
    gkk.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        kkg.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        kgk.process(cat1p, cat3p, cat2p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        gkk.process(cat3p, cat1p, cat2p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross21():
    # Check the 2-1 cross correlation

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
    k2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_weight_122 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_212 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_221 = np.zeros((nbins, nbins, nphi_bins))
    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    bin_size = (log_max_sep - log_min_sep) / nbins
    phi_bin_size = np.pi / nphi_bins
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                if j == k: continue
                d1 = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x2[k])/3.
                ceny = (y1[i] + y2[j] + y2[k])/3.

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)

                www = w1[i] * w2[j] * w2[k]
                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                zeta = www * g1p * k2[j] * k2[k]

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

    w_list = [true_weight_122, true_weight_212, true_weight_221]
    z_list = [true_zeta_122, true_zeta_212, true_zeta_221]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    kkg.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)
    kkg.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    kgk.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)

    gkk.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    gkk.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)

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

    # With ordered=False, doesn't do anything difference, since there is no other valid order.
    kkg.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)

    gkk.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)

    kkg.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    kgk.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgk.ntri, true_ntri_212)
    np.testing.assert_allclose(kgk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_212, rtol=1.e-5)
    gkk.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)

    kkg.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkg.ntri, true_ntri_221)
    np.testing.assert_allclose(kkg.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_221, rtol=1.e-5)
    gkk.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkk.ntri, true_ntri_122)
    np.testing.assert_allclose(gkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_122, rtol=1.e-5)


@timer
def test_direct_logmultipole_cross():
    # Check the cross correlation with LogMultipole
    if __name__ == '__main__':
        ngal = 100
    else:
        ngal = 30

    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(1,3, (ngal,))
    k1 = rng.normal(0,0.2, (ngal,))
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    k2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.uniform(1,3, (ngal,))
    g1_3 = rng.normal(0,0.2, (ngal,))
    g2_3 = rng.normal(0,0.2, (ngal,))
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 30.
    nbins = 5
    max_n = 10

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_zeta_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_123 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_132 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_213 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_231 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_312 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_321 = np.zeros((nbins, nbins, 2*max_n+1))
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

                www = w1[i] * w2[j] * w3[k]

                # 123, 132
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c1 to c3.
                    expmialpha3 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * k1[i] * k2[j] * g3p

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

                # 213, 231
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    expmialpha3 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * k1[i] * k2[j] * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_231[kr3,kr1,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_231[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_231[kr3,kr1,:] += 1
                    true_zeta_213[kr1,kr3,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_213[kr1,kr3,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_213[kr1,kr3,:] += 1

                # 312, 321
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g3 is projected to the average of lines from c3 to c1 and c3 to c2.
                    expmialpha1 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * k1[i] * k2[j] * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_312[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_312[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_312[kr1,kr2,:] += 1
                    true_zeta_321[kr2,kr1,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_321[kr2,kr1,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_321[kr2,kr1,:] += 1

    kkg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat2, cat1, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-4)
    kgk.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kgk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-4)
    kgk.process(cat2, cat3, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-4)
    gkk.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gkk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-4)
    gkk.process(cat3, cat2, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-4)

    # Repeat with binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat2, cat1, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-4)

    kgk.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kgk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-4)
    kgk.process(cat2, cat3, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-4)

    gkk.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gkk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-4)
    gkk.process(cat3, cat2, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    kkg.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(kkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-4)

    kgk.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(kgk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-4)
    kgk.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-4)

    gkk.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(gkk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-4)
    gkk.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-4)

    # Now use all three patched
    kkg.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(kkg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_123, rtol=1.e-4)
    kkg.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(kkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_213, rtol=1.e-4)

    kgk.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(kgk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_132, rtol=1.e-4)
    kgk.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(kgk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_231, rtol=1.e-4)

    gkk.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(gkk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_312, rtol=1.e-4)
    gkk.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(gkk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        kkg.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        kgk.process(cat1p, cat3p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        gkk.process(cat3p, cat1p, cat2p, patch_method='global')

    # Test I/O
    for name, corr in zip(['kkg', 'kgk', 'gkk'], [kkg, kgk, gkk]):
        ascii_name = 'output/'+name+'_ascii_logmultipole.txt'
        corr.write(ascii_name, precision=16)
        corr2 = treecorr.Corr3.from_file(ascii_name)
        np.testing.assert_allclose(corr2.ntri, corr.ntri)
        np.testing.assert_allclose(corr2.weight, corr.weight)
        np.testing.assert_allclose(corr2.zeta, corr.zeta)
        np.testing.assert_allclose(corr2.meand1, corr.meand1)
        np.testing.assert_allclose(corr2.meand2, corr.meand2)
        np.testing.assert_allclose(corr2.meand3, corr.meand3)
        np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
        np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
        np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)

@timer
def test_direct_logmultipole_cross21():
    # Check the cross correlation with LogMultipole
    if __name__ == '__main__':
        ngal = 100
    else:
        ngal = 30

    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(1,3, (ngal,))
    k1 = rng.normal(0,0.2, (ngal,))
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    g1_2 = rng.normal(0,0.2, (ngal,))
    g2_2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 10.
    max_sep = 30.
    nbins = 5
    max_n = 8

    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_zeta_112 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_112 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_112 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_121 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_121 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_121 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_211 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_211 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_211 = np.zeros((nbins, nbins, 2*max_n+1))
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            if i == j: continue
            for k in range(ngal):
                d1 = np.sqrt((x1[j]-x2[k])**2 + (y1[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x1[j])**2 + (y1[i]-y1[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                www = w1[i] * w1[j] * w2[k]

                # 112, 121
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c1 to c3.
                    expmialpha3 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    zeta = www * k1[i] * k1[j] * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_112[kr2,kr3,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_112[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_112[kr2,kr3,:] += 1
                    true_zeta_121[kr3,kr2,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_121[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_121[kr3,kr2,:] += 1

                # 211
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g3 is projected to the average of lines from c3 to c1 and c3 to c2.
                    expmialpha1 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x2[k]-x1[j]) - 1j*(y2[k]-y1[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    zeta = www * k1[i] * k1[j] * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_211[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_211[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_211[kr1,kr2,:] += 1

    kkg.process(cat1, cat1, cat2)
    np.testing.assert_allclose(kkg.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_112, rtol=1.e-4)
    kgk.process(cat1, cat2, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_121, rtol=1.e-4)
    # 3 arg version doesn't work for gkk because the gkk processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    kkg.process(cat1, cat2)
    np.testing.assert_allclose(kkg.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_112, rtol=1.e-4)
    gkk.process(cat2, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_211, rtol=1.e-4)

    # Repeat with binslop = 0
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkg.process(cat1, cat2)
    np.testing.assert_allclose(kkg.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_112, rtol=1.e-4)

    kgk.process(cat1, cat2, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_121, rtol=1.e-4)

    gkk.process(cat2, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_211, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    kkg.process(cat1p, cat2)
    np.testing.assert_allclose(kkg.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_112, rtol=1.e-4)

    kgk.process(cat1p, cat2, cat1)
    np.testing.assert_allclose(kgk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_121, rtol=1.e-4)

    gkk.process(cat2p, cat1)
    np.testing.assert_allclose(gkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_211, rtol=1.e-4)

    # Now use both patched
    kkg.process(cat1p, cat2p)
    np.testing.assert_allclose(kkg.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkg.zeta, true_zeta_112, rtol=1.e-4)

    kgk.process(cat1p, cat2p, cat1p)
    np.testing.assert_allclose(kgk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(kgk.zeta, true_zeta_121, rtol=1.e-4)

    gkk.process(cat2p, cat1p)
    np.testing.assert_allclose(gkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gkk.zeta, true_zeta_211, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        kkg.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        kgk.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        gkk.process(cat2p, cat1p, patch_method='global')


@timer
def test_kkg_logsas():
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    # And kappa(r) = kappa0 exp(-r^2/2r0^2)
    #
    # Doing the direct integral yields
    # zeta = int(int( g(x+x1,y+y1) k(x+x2,y+y2) k(x-x1-x2,y-y1-y2) (x1-Iy1)^2/(x1^2+y1^2) ))
    #      = -2pi/3 kappa0^2 gamma0 |q1|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    #
    # where the positions are measured relative to the centroid (x,y).
    # If we call the positions relative to the centroid:
    #    q1 = x1 + I y1
    #    q2 = x2 + I y2
    #    q3 = -(x1+x2) - I (y1+y2)
    #

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    gamma0 = 0.05
    kappa0 = 0.07
    r0 = 10.
    if __name__ == '__main__':
        ngal = 200000
        L = 30. * r0
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        ngal = 5000
        L = 10. * r0
        tol_factor = 3

    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
    k = kappa0 * np.exp(-r2/2.)

    min_sep = 10.
    max_sep = 13.
    nbins = 3
    min_phi = 45
    max_phi = 90
    nphi_bins = 5

    cat = treecorr.Catalog(x=x, y=y, k=k, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kkg = treecorr.KKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')
    kgk = treecorr.KGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')
    gkk = treecorr.GKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')

    for name, corr in zip(['kkg', 'kgk', 'gkk'], [kkg, kgk, gkk]):
        t0 = time.time()
        if name == 'kgk':
            corr.process(cat, cat, cat, algo='triangle')
        else:
            corr.process(cat, cat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        # Compute true zeta based on measured d1,d2,d3 in correlation
        d1 = corr.meand1
        d2 = corr.meand2
        d3 = corr.meand3
        s = d2
        t = d3 * np.exp(1j * corr.meanphi * np.pi/180.)
        q1 = (s + t)/3.
        q2 = q1 - t
        q3 = q1 - s
        nq1 = np.abs(q1)**2
        nq2 = np.abs(q2)**2
        nq3 = np.abs(q3)**2

        L = L - (np.abs(q1) + np.abs(q2) + np.abs(q3))/3.

        true_zeta = (-2.*np.pi * gamma0 * kappa0**2)/(3*L**2) * np.exp(-(nq1+nq2+nq3)/(2.*r0**2))

        if name == 'kkg':
            true_zeta *= nq3
        elif name == 'kgk':
            true_zeta *= nq2
        else:
            true_zeta *= nq1

        print('ntri = ',corr.ntri)
        print('zeta = ',corr.zeta)
        print('true_zeta = ',true_zeta)
        print('ratio = ',corr.zeta / true_zeta)
        print('diff = ',corr.zeta - true_zeta)
        print('max rel diff = ',np.max(np.abs((corr.zeta - true_zeta)/true_zeta)))
        np.testing.assert_allclose(corr.zeta, true_zeta, rtol=0.2 * tol_factor, atol=1.e-7)
        np.testing.assert_allclose(np.log(np.abs(corr.zeta)),
                                   np.log(np.abs(true_zeta)), atol=0.2 * tol_factor)

        # Repeat this using Multipole and then convert to SAS:
        if name == 'kkg':
            cls = treecorr.KKGCorrelation
        elif name == 'kgk':
            cls = treecorr.KGKCorrelation
        else:
            cls = treecorr.GKKCorrelation
        corrm = cls(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=80,
                    sep_units='arcmin', bin_type='LogMultipole')
        t0 = time.time()
        if name == 'kgk':
            corrm.process(cat, cat, cat)
        else:
            corrm.process(cat, cat)
        corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins, phi_units='deg')
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('zeta mean ratio = ',np.mean(corrs.zeta / corr.zeta))
        print('zeta mean diff = ',np.mean(corrs.zeta - corr.zeta))
        # Some of the individual values are a little ratty, but on average, they are quite close.
        np.testing.assert_allclose(corrs.zeta, corr.zeta, rtol=0.2*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.zeta / corr.zeta), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.std(corrs.zeta / corr.zeta), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(corrs.ntri, corr.ntri, rtol=0.03*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd1, corr.meanlogd1, rtol=0.03*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd2, corr.meanlogd2, rtol=0.03*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd3, corr.meanlogd3, rtol=0.03*tol_factor)
        np.testing.assert_allclose(corrs.meanphi, corr.meanphi, rtol=0.03*tol_factor)

        # Error to try to change sep binning with toSAS
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                              phi_units='deg', min_sep=5)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                              phi_units='deg', max_sep=25)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                              phi_units='deg', nbins=20)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                              phi_units='deg', bin_size=0.01, nbins=None)
        # Error if non-Multipole calls toSAS
        with assert_raises(TypeError):
            corrs.toSAS()

        # All of the above is the default algorithm if process doesn't set algo='triangle'.
        # Check the automatic use of the multipole algorithm from LogSAS.
        corr3 = corr.copy()
        if name == 'kgk':
            corr3.process(cat, cat, cat, algo='multipole', max_n=80)
        else:
            corr3.process(cat, cat, algo='multipole', max_n=80)
        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.zeta, corrs.zeta)

        # Check that we get the same result using the corr3 functin:
        # (This implicitly uses the multipole algorithm.)
        cat.write(os.path.join('data',name+'_data_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['zetar'], corr3.zetar.flatten(), rtol=1.e-3, atol=0)
        np.testing.assert_allclose(corr3_output['zetai'], corr3.zetai.flatten(), rtol=1.e-3,
                                   atol=0)

        if name == 'kgk':
            # Invalid to omit file_name2
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)
        else:
            # Invalid to call cat2 file_name3 rather than file_name2
            config['file_name3'] = config['file_name2']
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)

        # Check the fits write option
        try:
            import fitsio
        except ImportError:
            pass
        else:
            out_file_name = os.path.join('output','corr_kkg_logsas.fits')
            corr.write(out_file_name)
            data = fitsio.read(out_file_name)
            np.testing.assert_allclose(data['d2_nom'], np.exp(corr.logd2).flatten())
            np.testing.assert_allclose(data['d3_nom'], np.exp(corr.logd3).flatten())
            np.testing.assert_allclose(data['phi_nom'], corr.phi.flatten())
            np.testing.assert_allclose(data['meand1'], corr.meand1.flatten())
            np.testing.assert_allclose(data['meanlogd1'], corr.meanlogd1.flatten())
            np.testing.assert_allclose(data['meand2'], corr.meand2.flatten())
            np.testing.assert_allclose(data['meanlogd2'], corr.meanlogd2.flatten())
            np.testing.assert_allclose(data['meand3'], corr.meand3.flatten())
            np.testing.assert_allclose(data['meanlogd3'], corr.meanlogd3.flatten())
            np.testing.assert_allclose(data['meanphi'], corr.meanphi.flatten())
            np.testing.assert_allclose(data['zetar'], corr.zeta.real.flatten())
            np.testing.assert_allclose(data['zetai'], corr.zeta.imag.flatten())
            np.testing.assert_allclose(data['sigma_zeta'], np.sqrt(corr.varzeta.flatten()))
            np.testing.assert_allclose(data['weight'], corr.weight.flatten())
            np.testing.assert_allclose(data['ntri'], corr.ntri.flatten())

            # Check the read function
            # Note: These don't need the flatten.
            # The read function should reshape them to the right shape.
            corr2 = treecorr.Corr3.from_file(out_file_name)
            np.testing.assert_allclose(corr2.logd2, corr.logd2)
            np.testing.assert_allclose(corr2.logd3, corr.logd3)
            np.testing.assert_allclose(corr2.phi, corr.phi)
            np.testing.assert_allclose(corr2.meand1, corr.meand1)
            np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
            np.testing.assert_allclose(corr2.meand2, corr.meand2)
            np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
            np.testing.assert_allclose(corr2.meand3, corr.meand3)
            np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)
            np.testing.assert_allclose(corr2.meanphi, corr.meanphi)
            np.testing.assert_allclose(corr2.zeta, corr.zeta)
            np.testing.assert_allclose(corr2.varzeta, corr.varzeta)
            np.testing.assert_allclose(corr2.weight, corr.weight)
            np.testing.assert_allclose(corr2.ntri, corr.ntri)
            assert corr2.coords == corr.coords
            assert corr2.metric == corr.metric
            assert corr2.sep_units == corr.sep_units
            assert corr2.bin_type == corr.bin_type

@timer
def test_varzeta():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 5000
    nruns = 50000

    file_name = 'data/test_varzeta_kkg.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kkgs = []
        all_kgks = []
        all_gkks = []

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
            k = gamma0 * np.exp(-r2/2.)
            g1 += rng.normal(0, 0.3, size=ngal)
            g2 += rng.normal(0, 0.3, size=ngal)
            k += rng.normal(0, 0.2, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, k=k, g1=g1, g2=g2)
            kkg = treecorr.KKGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            kgk = treecorr.KGKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            gkk = treecorr.GKKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            kkg.process(cat, cat)
            kgk.process(cat, cat, cat)
            gkk.process(cat, cat)
            all_kkgs.append(kkg)
            all_kgks.append(kgk)
            all_gkks.append(gkk)

        mean_kkg_zetar = np.mean([kkg.zetar for kkg in all_kkgs], axis=0)
        mean_kkg_zetai = np.mean([kkg.zetai for kkg in all_kkgs], axis=0)
        var_kkg_zetar = np.var([kkg.zetar for kkg in all_kkgs], axis=0)
        var_kkg_zetai = np.var([kkg.zetai for kkg in all_kkgs], axis=0)
        mean_kkg_varzeta = np.mean([kkg.varzeta for kkg in all_kkgs], axis=0)
        mean_kgk_zetar = np.mean([kgk.zetar for kgk in all_kgks], axis=0)
        mean_kgk_zetai = np.mean([kgk.zetai for kgk in all_kgks], axis=0)
        var_kgk_zetar = np.var([kgk.zetar for kgk in all_kgks], axis=0)
        var_kgk_zetai = np.var([kgk.zetai for kgk in all_kgks], axis=0)
        mean_kgk_varzeta = np.mean([kgk.varzeta for kgk in all_kgks], axis=0)
        mean_gkk_zetar = np.mean([gkk.zetar for gkk in all_gkks], axis=0)
        mean_gkk_zetai = np.mean([gkk.zetai for gkk in all_gkks], axis=0)
        var_gkk_zetar = np.var([gkk.zetar for gkk in all_gkks], axis=0)
        var_gkk_zetai = np.var([gkk.zetai for gkk in all_gkks], axis=0)
        mean_gkk_varzeta = np.mean([gkk.varzeta for gkk in all_gkks], axis=0)

        np.savez(file_name,
                 mean_kkg_zetar=mean_kkg_zetar,
                 mean_kkg_zetai=mean_kkg_zetai,
                 var_kkg_zetar=var_kkg_zetar,
                 var_kkg_zetai=var_kkg_zetai,
                 mean_kkg_varzeta=mean_kkg_varzeta,
                 mean_kgk_zetar=mean_kgk_zetar,
                 mean_kgk_zetai=mean_kgk_zetai,
                 var_kgk_zetar=var_kgk_zetar,
                 var_kgk_zetai=var_kgk_zetai,
                 mean_kgk_varzeta=mean_kgk_varzeta,
                 mean_gkk_zetar=mean_gkk_zetar,
                 mean_gkk_zetai=mean_gkk_zetai,
                 var_gkk_zetar=var_gkk_zetar,
                 var_gkk_zetai=var_gkk_zetai,
                 mean_gkk_varzeta=mean_gkk_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_kkg_zetar = data['mean_kkg_zetar']
    mean_kkg_zetai = data['mean_kkg_zetai']
    var_kkg_zetar = data['var_kkg_zetar']
    var_kkg_zetai = data['var_kkg_zetai']
    mean_kkg_varzeta = data['mean_kkg_varzeta']
    mean_kgk_zetar = data['mean_kgk_zetar']
    mean_kgk_zetai = data['mean_kgk_zetai']
    var_kgk_zetar = data['var_kgk_zetar']
    var_kgk_zetai = data['var_kgk_zetai']
    mean_kgk_varzeta = data['mean_kgk_varzeta']
    mean_gkk_zetar = data['mean_gkk_zetar']
    mean_gkk_zetai = data['mean_gkk_zetai']
    var_gkk_zetar = data['var_gkk_zetar']
    var_gkk_zetai = data['var_gkk_zetai']
    mean_gkk_varzeta = data['mean_gkk_varzeta']

    print('var_kkg_zetar = ',var_kkg_zetar)
    print('mean kkg_varzeta = ',mean_kkg_varzeta)
    print('ratio = ',var_kkg_zetar.ravel() / mean_kkg_varzeta.ravel())
    print('var_kgk_zetar = ',var_kgk_zetar)
    print('mean kgk_varzeta = ',mean_kgk_varzeta)
    print('ratio = ',var_kgk_zetar.ravel() / mean_kgk_varzeta.ravel())
    print('var_gkk_zetar = ',var_gkk_zetar)
    print('mean gkk_varzeta = ',mean_gkk_varzeta)
    print('ratio = ',var_gkk_zetar.ravel() / mean_gkk_varzeta.ravel())

    print('max relerr for kkg zetar = ',
          np.max(np.abs((var_kkg_zetar - mean_kkg_varzeta)/var_kkg_zetar)))
    print('max relerr for kkg zetai = ',
          np.max(np.abs((var_kkg_zetai - mean_kkg_varzeta)/var_kkg_zetai)))
    np.testing.assert_allclose(mean_kkg_varzeta, var_kkg_zetar, rtol=0.1)
    np.testing.assert_allclose(mean_kkg_varzeta, var_kkg_zetai, rtol=0.1)

    print('max relerr for kgk zetar = ',
          np.max(np.abs((var_kgk_zetar - mean_kgk_varzeta)/var_kgk_zetar)))
    print('max relerr for kgk zetai = ',
          np.max(np.abs((var_kgk_zetai - mean_kgk_varzeta)/var_kgk_zetai)))
    np.testing.assert_allclose(mean_kgk_varzeta, var_kgk_zetar, rtol=0.1)
    np.testing.assert_allclose(mean_kgk_varzeta, var_kgk_zetai, rtol=0.1)

    print('max relerr for gkk zetar = ',
          np.max(np.abs((var_gkk_zetar - mean_gkk_varzeta)/var_gkk_zetar)))
    print('max relerr for gkk zetai = ',
          np.max(np.abs((var_gkk_zetai - mean_gkk_varzeta)/var_gkk_zetai)))
    np.testing.assert_allclose(mean_gkk_varzeta, var_gkk_zetar, rtol=0.1)
    np.testing.assert_allclose(mean_gkk_varzeta, var_gkk_zetai, rtol=0.1)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
    k = gamma0 * np.exp(-r2/2.)
    g1 += rng.normal(0, 0.3, size=ngal)
    g2 += rng.normal(0, 0.3, size=ngal)
    k += rng.normal(0, 0.2, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, k=k, g1=g1, g2=g2)
    kkg = treecorr.KKGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
    kgk = treecorr.KGKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
    gkk = treecorr.GKKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(kkg.cov, 0)
    np.testing.assert_array_equal(kkg.varzeta, 0)
    np.testing.assert_array_equal(kgk.cov, 0)
    np.testing.assert_array_equal(kgk.varzeta, 0)
    np.testing.assert_array_equal(gkk.cov, 0)
    np.testing.assert_array_equal(gkk.varzeta, 0)

    kkg.process(cat, cat)
    print('KKG single run:')
    print('max relerr for zetar = ',np.max(np.abs((kkg.varzeta - var_kkg_zetar)/var_kkg_zetar)))
    print('ratio = ',kkg.varzeta / var_kkg_zetar)
    print('max relerr for zetai = ',np.max(np.abs((kkg.varzeta - var_kkg_zetai)/var_kkg_zetai)))
    print('ratio = ',kkg.varzeta / var_kkg_zetai)
    print('var_num = ',kkg._var_num)
    print('ntri = ',kkg.ntri)
    np.testing.assert_allclose(kkg.varzeta, var_kkg_zetar, rtol=0.2)
    np.testing.assert_allclose(kkg.varzeta, var_kkg_zetai, rtol=0.2)
    np.testing.assert_allclose(kkg.cov.diagonal(), kkg.varzeta.ravel())

    kgk.process(cat, cat, cat)
    print('KGK single run:')
    print('max relerr for zetar = ',np.max(np.abs((kgk.varzeta - var_kgk_zetar)/var_kgk_zetar)))
    print('ratio = ',kgk.varzeta / var_kgk_zetar)
    print('max relerr for zetai = ',np.max(np.abs((kgk.varzeta - var_kgk_zetai)/var_kgk_zetai)))
    print('ratio = ',kgk.varzeta / var_kgk_zetai)
    np.testing.assert_allclose(kgk.varzeta, var_kgk_zetar, rtol=0.2)
    np.testing.assert_allclose(kgk.varzeta, var_kgk_zetai, rtol=0.2)
    np.testing.assert_allclose(kgk.cov.diagonal(), kgk.varzeta.ravel())

    gkk.process(cat, cat)
    print('GKK single run:')
    print('max relerr for zetar = ',np.max(np.abs((gkk.varzeta - var_gkk_zetar)/var_gkk_zetar)))
    print('ratio = ',gkk.varzeta / var_gkk_zetar)
    print('max relerr for zetai = ',np.max(np.abs((gkk.varzeta - var_gkk_zetai)/var_gkk_zetai)))
    print('ratio = ',gkk.varzeta / var_gkk_zetai)
    np.testing.assert_allclose(gkk.varzeta, var_gkk_zetar, rtol=0.2)
    np.testing.assert_allclose(gkk.varzeta, var_gkk_zetai, rtol=0.2)
    np.testing.assert_allclose(gkk.cov.diagonal(), gkk.varzeta.ravel())

@timer
def test_kkg_logsas_jk():
    # Test jackknife covariance estimates for kkg correlations with LogSAS binning.

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    if __name__ == '__main__':
        nhalo = 1000
        nsource = 100000
        npatch = 32
        tol_factor = 1
    else:
        nhalo = 500
        nsource = 10000
        npatch = 12
        tol_factor = 4

    # This test is pretty fragile with respect to the choice of these parameters.
    # It was hard to find a combination where JK worked reasonably well with the size of
    # data set that is feasible for a unit test.  Mostly there need to be not many bins
    # and near-equilateral triangles seem to work better, since they have higher snr.
    # Also, it turned out to be important for the K and G catalogs to not use the same
    # points; I think so there are no degenerate triangles, but I'm not sure exactly what
    # the problem with them is in this case.

    nbins = 2
    min_sep = 10
    max_sep = 16
    nphi_bins = 2
    min_phi = 30
    max_phi = 90

    file_name = 'data/test_kkg_logsas_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_kkg = []
        all_kgk = []
        all_gkk = []
        rng1 = np.random.default_rng()
        for run in range(nruns):
            # It doesn't work as well if the same points are in both, so make a single field
            # with both k and g, but take half the points for k, and the other half for g.
            x, y, g1, g2, k = generate_shear_field(2*nsource, nhalo, rng1)
            x1 = x[::2]
            y1 = y[::2]
            g1 = g1[::2]
            g2 = g2[::2]
            x2 = x[1::2]
            y2 = y[1::2]
            k = k[1::2]
            print(run,': ',np.mean(k),np.std(k),np.std(g1),np.std(g2))
            gcat = treecorr.Catalog(x=x1, y=y1, g1=g1, g2=g2)
            kcat = treecorr.Catalog(x=x2, y=y2, k=k)
            kkg = treecorr.KKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            kkg.process(kcat, gcat)
            all_kkg.append(kkg.zeta.ravel())

            kgk = treecorr.KGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            kgk.process(kcat, gcat, kcat)
            all_kgk.append(kgk.zeta.ravel())

            gkk = treecorr.GKKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            gkk.process(gcat, kcat)
            all_gkk.append(gkk.zeta.ravel())

        mean_kkg = np.mean([zeta for zeta in all_kkg], axis=0)
        var_kkg = np.var([zeta for zeta in all_kkg], axis=0)
        mean_kgk = np.mean([zeta for zeta in all_kgk], axis=0)
        var_kgk = np.var([zeta for zeta in all_kgk], axis=0)
        mean_gkk = np.mean([zeta for zeta in all_gkk], axis=0)
        var_gkk = np.var([zeta for zeta in all_gkk], axis=0)

        np.savez(file_name,
                 mean_kkg=mean_kkg, var_kkg=var_kkg,
                 mean_kgk=mean_kgk, var_kgk=var_kgk,
                 mean_gkk=mean_gkk, var_gkk=var_gkk)

    data = np.load(file_name)
    mean_kkg = data['mean_kkg']
    var_kkg = data['var_kkg']
    mean_kgk = data['mean_kgk']
    var_kgk = data['var_kgk']
    mean_gkk = data['mean_gkk']
    var_gkk = data['var_gkk']
    print('mean kkg = ',mean_kkg)
    print('var kkg = ',var_kkg)
    print('mean kgk = ',mean_kgk)
    print('var kgk = ',var_kgk)
    print('mean gkk = ',mean_gkk)
    print('var gkk = ',var_gkk)

    rng = np.random.default_rng(1234)
    x, y, g1, g2, k = generate_shear_field(2*nsource, nhalo, rng)
    x1 = x[::2]
    y1 = y[::2]
    g1 = g1[::2]
    g2 = g2[::2]
    x2 = x[1::2]
    y2 = y[1::2]
    k = k[1::2]
    gcat = treecorr.Catalog(x=x1, y=y1, g1=g1, g2=g2, npatch=npatch, rng=rng)
    kcat = treecorr.Catalog(x=x2, y=y2, k=k, rng=rng, patch_centers=gcat.patch_centers)

    # First check calculate_xi with all pairs in results dict.
    kkg = treecorr.KKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    kkg.process(kcat, gcat)
    kkg2 = kkg.copy()
    kkg2._calculate_xi_from_pairs(list(kkg.results.keys()), False)
    np.testing.assert_allclose(kkg2.ntri, kkg.ntri, rtol=0.01)
    np.testing.assert_allclose(kkg2.zeta, kkg.zeta, rtol=0.01)
    np.testing.assert_allclose(kkg2.varzeta, kkg.varzeta, rtol=0.01)

    kgk = treecorr.KGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    kgk.process(kcat, gcat, kcat)
    kgk2 = kgk.copy()
    kgk2._calculate_xi_from_pairs(list(kgk.results.keys()), False)
    np.testing.assert_allclose(kgk2.ntri, kgk.ntri, rtol=0.01)
    np.testing.assert_allclose(kgk2.zeta, kgk.zeta, rtol=0.01)
    np.testing.assert_allclose(kgk2.varzeta, kgk.varzeta, rtol=0.01)

    gkk = treecorr.GKKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    gkk.process(gcat, kcat)
    gkk2 = gkk.copy()
    gkk2._calculate_xi_from_pairs(list(gkk.results.keys()), False)
    np.testing.assert_allclose(gkk2.ntri, gkk.ntri, rtol=0.01)
    np.testing.assert_allclose(gkk2.zeta, gkk.zeta, rtol=0.01)
    np.testing.assert_allclose(gkk2.varzeta, gkk.varzeta, rtol=0.01)

    # Next check jackknife covariance estimate
    cov_kkg = kkg.estimate_cov('jackknife')
    print('kkg var ratio = ',np.diagonal(cov_kkg)/var_kkg)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_kkg))-np.log(var_kkg))))
    np.testing.assert_allclose(np.diagonal(cov_kkg), var_kkg, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_kkg)), np.log(var_kkg), atol=0.5*tol_factor)

    cov_kgk = kgk.estimate_cov('jackknife')
    print('kgk var ratio = ',np.diagonal(cov_kgk)/var_kgk)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_kgk))-np.log(var_kgk))))
    np.testing.assert_allclose(np.diagonal(cov_kgk), var_kgk, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_kgk)), np.log(var_kgk), atol=0.5*tol_factor)

    cov_gkk = gkk.estimate_cov('jackknife')
    print('gkk var ratio = ',np.diagonal(cov_gkk)/var_gkk)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_gkk))-np.log(var_gkk))))
    np.testing.assert_allclose(np.diagonal(cov_gkk), var_gkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_gkk)), np.log(var_gkk), atol=0.7*tol_factor)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_kkg.dat')
    kkg.write(file_name, write_patch_results=True)
    kkg2 = treecorr.Corr3.from_file(file_name)
    cov2 = kkg2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_kkg)

    file_name = os.path.join('output','test_write_results_kgk.dat')
    kgk.write(file_name, write_patch_results=True)
    kgk2 = treecorr.Corr3.from_file(file_name)
    cov2 = kgk2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_kgk)

    file_name = os.path.join('output','test_write_results_gkk.dat')
    gkk.write(file_name, write_patch_results=True)
    gkk2 = treecorr.Corr3.from_file(file_name)
    cov2 = gkk2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_gkk)

    # Check jackknife using LogMultipole
    print('Using LogMultipole:')
    kkgm = treecorr.KKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    kkgm.process(kcat, gcat)
    fm = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                 nphi_bins=nphi_bins).zeta.ravel()
    cov = kkgm.estimate_cov('jackknife', func=fm)
    print('kkg max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkg))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkg, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkg), atol=0.5*tol_factor)

    kgkm = treecorr.KGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    kgkm.process(kcat, gcat, kcat)
    cov = kgkm.estimate_cov('jackknife', func=fm)
    print('kgk max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kgk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kgk, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kgk), atol=0.5*tol_factor)

    gkkm = treecorr.GKKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    gkkm.process(gcat, kcat)
    cov = gkkm.estimate_cov('jackknife', func=fm)
    print('gkk max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_gkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gkk), atol=0.7*tol_factor)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross21()
    test_varzeta_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross21()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross21()
    test_kkg_logsas()
    test_varzeta()
    test_kkg_logsas_jk()
