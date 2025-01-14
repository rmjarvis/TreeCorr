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

    kgg = treecorr.KGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gkg = treecorr.GKGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ggk = treecorr.GGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
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
    true_gam0_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
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

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                www = w1[i] * w2[j] * w3[k]
                gam0 = www * k1[i] * g2p * g3p
                gam1 = www * k1[i] * np.conjugate(g2p) * g3p

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_123
                        true_gam0 = true_gam0_123
                        true_gam1 = true_gam2_123
                        true_weight = true_weight_123
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_213
                        true_gam0 = true_gam0_213
                        true_gam1 = true_gam1_213
                        true_weight = true_weight_213
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_231
                        true_gam0 = true_gam0_231
                        true_gam1 = true_gam1_231
                        true_weight = true_weight_231
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_132
                        true_gam0 = true_gam0_132
                        true_gam1 = true_gam2_132
                        true_weight = true_weight_132
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_312
                        true_gam0 = true_gam0_312
                        true_gam1 = true_gam1_312
                        true_weight = true_weight_312
                        gam1 = np.conjugate(gam1)
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_321
                        true_gam0 = true_gam0_321
                        true_gam1 = true_gam1_321
                        true_weight = true_weight_321
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)

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

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_gam0_sum1 = true_gam0_123 + true_gam0_132
    true_gam1_sum1 = true_gam2_123 + true_gam2_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_gam0_sum2 = true_gam0_213 + true_gam0_312
    true_gam1_sum2 = true_gam1_213 + true_gam1_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_gam0_sum3 = true_gam0_231 + true_gam0_321
    true_gam1_sum3 = true_gam1_231 + true_gam1_321
    pos = true_weight_sum1 > 0
    true_gam0_sum1[pos] /= true_weight_sum1[pos]
    true_gam1_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_gam0_sum2[pos] /= true_weight_sum2[pos]
    true_gam1_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_gam0_sum3[pos] /= true_weight_sum3[pos]
    true_gam1_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_gam0_123, true_gam0_132, true_gam0_213, true_gam0_231,
               true_gam0_312, true_gam0_321]
    g1_list = [true_gam2_123, true_gam2_132, true_gam1_213, true_gam1_231,
               true_gam1_312, true_gam1_321]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    kgg.process(cat1, cat2, cat3)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)

    kgg.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_132)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-5)

    gkg.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    gkg.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_312)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-5)

    ggk.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)
    ggk.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_321)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 1
    kgg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)

    gkg.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # Check bin_slop=0
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kgg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kgg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # With these, ordered=False is equivalent to the K vertex being fixed.
    kgg.process(cat1, cat2, cat3, ordered=1)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=2)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=3)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        kgg.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    kgg.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2p, cat1, cat3)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2p, cat3, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1p, cat3)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3p, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3p)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1p)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    # Now all three patched
    kgg.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    # Unordered
    kgg.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=False)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=1)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=2)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=3)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # patch_method=local
    kgg.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, patch_method='local')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, patch_method='local')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        kgg.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        gkg.process(cat2p, cat1p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        ggk.process(cat2p, cat3p, cat1p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross12():
    # Check the 1-2 cross correlation

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

    kgg = treecorr.KGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gkg = treecorr.GKGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ggk = treecorr.GGKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_gam0_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
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

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2

                www = w1[i] * w2[j] * w2[k]
                gam0 = www * k1[i] * g2p * g3p
                gam1 = www * k1[i] * np.conjugate(g2p) * g3p

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam2_122
                        true_weight = true_weight_122
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_weight = true_weight_212
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_weight = true_weight_221
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam2_122
                        true_weight = true_weight_122
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_weight = true_weight_212
                        gam1 = np.conjugate(gam1)
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_weight = true_weight_221
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)

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

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1

    pos = true_weight_221 > 0
    true_gam0_221[pos] /= true_weight_221[pos]
    true_gam1_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_gam0_212[pos] /= true_weight_212[pos]
    true_gam1_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_gam0_122[pos] /= true_weight_122[pos]
    true_gam2_122[pos] /= true_weight_122[pos]

    kgg.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: GKG doesn't have a two-argument version.
    kgg.process(cat1, cat2)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2, cat1)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    with assert_raises(ValueError):
        gkg.process(cat1, cat2)
    with assert_raises(ValueError):
        gkg.process(cat2, cat1)
    with assert_raises(ValueError):
        kgg.process(cat1)
    with assert_raises(ValueError):
        kgg.process(cat2)
    with assert_raises(ValueError):
        gkg.process(cat1)
    with assert_raises(ValueError):
        gkg.process(cat2)
    with assert_raises(ValueError):
        ggk.process(cat1)
    with assert_raises(ValueError):
        ggk.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    kgg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    # Repeat with binslop = 0
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    kgg.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    # And again with no top-level recursion
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    kgg.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)

    kgg.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)


@timer
def test_vargam_logruv():
    # Test that the shot noise estimate of vargam0 is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(1234)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 300
    nruns = 50000

    file_name = 'data/test_vargam_kgg_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kggs = []
        all_gkgs = []
        all_ggks = []

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
            kgg = treecorr.KGGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            gkg = treecorr.GKGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            ggk = treecorr.GGKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                          sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                          bin_type='LogRUV')
            kgg.process(cat, cat)
            gkg.process(cat, cat, cat)
            ggk.process(cat, cat)
            all_kggs.append(kgg)
            all_gkgs.append(gkg)
            all_ggks.append(ggk)

        mean_kgg_gam0r = np.mean([kgg.gam0r for kgg in all_kggs], axis=0)
        mean_kgg_gam0i = np.mean([kgg.gam0i for kgg in all_kggs], axis=0)
        mean_kgg_gam2r = np.mean([kgg.gam2r for kgg in all_kggs], axis=0)
        mean_kgg_gam2i = np.mean([kgg.gam2i for kgg in all_kggs], axis=0)
        var_kgg_gam0r = np.var([kgg.gam0r for kgg in all_kggs], axis=0)
        var_kgg_gam0i = np.var([kgg.gam0i for kgg in all_kggs], axis=0)
        var_kgg_gam2r = np.var([kgg.gam2r for kgg in all_kggs], axis=0)
        var_kgg_gam2i = np.var([kgg.gam2i for kgg in all_kggs], axis=0)
        mean_kgg_vargam0 = np.mean([kgg.vargam0 for kgg in all_kggs], axis=0)
        mean_kgg_vargam2 = np.mean([kgg.vargam2 for kgg in all_kggs], axis=0)
        mean_gkg_gam0r = np.mean([gkg.gam0r for gkg in all_gkgs], axis=0)
        mean_gkg_gam0i = np.mean([gkg.gam0i for gkg in all_gkgs], axis=0)
        mean_gkg_gam1r = np.mean([gkg.gam1r for gkg in all_gkgs], axis=0)
        mean_gkg_gam1i = np.mean([gkg.gam1i for gkg in all_gkgs], axis=0)
        var_gkg_gam0r = np.var([gkg.gam0r for gkg in all_gkgs], axis=0)
        var_gkg_gam0i = np.var([gkg.gam0i for gkg in all_gkgs], axis=0)
        var_gkg_gam1r = np.var([gkg.gam1r for gkg in all_gkgs], axis=0)
        var_gkg_gam1i = np.var([gkg.gam1i for gkg in all_gkgs], axis=0)
        mean_gkg_vargam0 = np.mean([gkg.vargam0 for gkg in all_gkgs], axis=0)
        mean_gkg_vargam1 = np.mean([gkg.vargam1 for gkg in all_gkgs], axis=0)
        mean_ggk_gam0r = np.mean([ggk.gam0r for ggk in all_ggks], axis=0)
        mean_ggk_gam0i = np.mean([ggk.gam0i for ggk in all_ggks], axis=0)
        mean_ggk_gam1r = np.mean([ggk.gam1r for ggk in all_ggks], axis=0)
        mean_ggk_gam1i = np.mean([ggk.gam1i for ggk in all_ggks], axis=0)
        var_ggk_gam0r = np.var([ggk.gam0r for ggk in all_ggks], axis=0)
        var_ggk_gam0i = np.var([ggk.gam0i for ggk in all_ggks], axis=0)
        var_ggk_gam1r = np.var([ggk.gam1r for ggk in all_ggks], axis=0)
        var_ggk_gam1i = np.var([ggk.gam1i for ggk in all_ggks], axis=0)
        mean_ggk_vargam0 = np.mean([ggk.vargam0 for ggk in all_ggks], axis=0)
        mean_ggk_vargam1 = np.mean([ggk.vargam1 for ggk in all_ggks], axis=0)

        np.savez(file_name,
                 mean_kgg_gam0r=mean_kgg_gam0r,
                 mean_kgg_gam0i=mean_kgg_gam0i,
                 mean_kgg_gam2r=mean_kgg_gam2r,
                 mean_kgg_gam2i=mean_kgg_gam2i,
                 var_kgg_gam0r=var_kgg_gam0r,
                 var_kgg_gam0i=var_kgg_gam0i,
                 var_kgg_gam2r=var_kgg_gam2r,
                 var_kgg_gam2i=var_kgg_gam2i,
                 mean_kgg_vargam0=mean_kgg_vargam0,
                 mean_kgg_vargam2=mean_kgg_vargam2,
                 mean_gkg_gam0r=mean_gkg_gam0r,
                 mean_gkg_gam0i=mean_gkg_gam0i,
                 mean_gkg_gam1r=mean_gkg_gam1r,
                 mean_gkg_gam1i=mean_gkg_gam1i,
                 var_gkg_gam0r=var_gkg_gam0r,
                 var_gkg_gam0i=var_gkg_gam0i,
                 var_gkg_gam1r=var_gkg_gam1r,
                 var_gkg_gam1i=var_gkg_gam1i,
                 mean_gkg_vargam0=mean_gkg_vargam0,
                 mean_gkg_vargam1=mean_gkg_vargam1,
                 mean_ggk_gam0r=mean_ggk_gam0r,
                 mean_ggk_gam0i=mean_ggk_gam0i,
                 mean_ggk_gam1r=mean_ggk_gam1r,
                 mean_ggk_gam1i=mean_ggk_gam1i,
                 var_ggk_gam0r=var_ggk_gam0r,
                 var_ggk_gam0i=var_ggk_gam0i,
                 var_ggk_gam1r=var_ggk_gam1r,
                 var_ggk_gam1i=var_ggk_gam1i,
                 mean_ggk_vargam0=mean_ggk_vargam0,
                 mean_ggk_vargam1=mean_ggk_vargam1)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_kgg_gam0r = data['mean_kgg_gam0r']
    mean_kgg_gam0i = data['mean_kgg_gam0i']
    mean_kgg_gam2r = data['mean_kgg_gam2r']
    mean_kgg_gam2i = data['mean_kgg_gam2i']
    var_kgg_gam0r = data['var_kgg_gam0r']
    var_kgg_gam0i = data['var_kgg_gam0i']
    var_kgg_gam2r = data['var_kgg_gam2r']
    var_kgg_gam2i = data['var_kgg_gam2i']
    mean_kgg_vargam0 = data['mean_kgg_vargam0']
    mean_kgg_vargam2 = data['mean_kgg_vargam2']
    mean_gkg_gam0r = data['mean_gkg_gam0r']
    mean_gkg_gam0i = data['mean_gkg_gam0i']
    mean_gkg_gam1r = data['mean_gkg_gam1r']
    mean_gkg_gam1i = data['mean_gkg_gam1i']
    var_gkg_gam0r = data['var_gkg_gam0r']
    var_gkg_gam0i = data['var_gkg_gam0i']
    var_gkg_gam1r = data['var_gkg_gam1r']
    var_gkg_gam1i = data['var_gkg_gam1i']
    mean_gkg_vargam0 = data['mean_gkg_vargam0']
    mean_gkg_vargam1 = data['mean_gkg_vargam1']
    mean_ggk_gam0r = data['mean_ggk_gam0r']
    mean_ggk_gam0i = data['mean_ggk_gam0i']
    mean_ggk_gam1r = data['mean_ggk_gam1r']
    mean_ggk_gam1i = data['mean_ggk_gam1i']
    var_ggk_gam0r = data['var_ggk_gam0r']
    var_ggk_gam0i = data['var_ggk_gam0i']
    var_ggk_gam1r = data['var_ggk_gam1r']
    var_ggk_gam1i = data['var_ggk_gam1i']
    mean_ggk_vargam0 = data['mean_ggk_vargam0']
    mean_ggk_vargam1 = data['mean_ggk_vargam1']

    print('var_kgg_gam0r = ',var_kgg_gam0r)
    print('mean kgg_vargam0 = ',mean_kgg_vargam0)
    print('ratio = ',var_kgg_gam0r.ravel() / mean_kgg_vargam0.ravel())
    print('var_gkg_gam0r = ',var_gkg_gam0r)
    print('mean gkg_vargam0 = ',mean_gkg_vargam0)
    print('ratio = ',var_gkg_gam0r.ravel() / mean_gkg_vargam0.ravel())
    print('var_ggk_gam0r = ',var_ggk_gam0r)
    print('mean ggk_vargam0 = ',mean_ggk_vargam0)
    print('ratio = ',var_ggk_gam0r.ravel() / mean_ggk_vargam0.ravel())

    print('max relerr for kgg gam0r = ',
          np.max(np.abs((var_kgg_gam0r - mean_kgg_vargam0)/var_kgg_gam0r)))
    print('max relerr for kgg gam0i = ',
          np.max(np.abs((var_kgg_gam0i - mean_kgg_vargam0)/var_kgg_gam0i)))
    np.testing.assert_allclose(mean_kgg_vargam0, var_kgg_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_kgg_vargam0, var_kgg_gam0i, rtol=0.03)
    np.testing.assert_allclose(mean_kgg_vargam2, var_kgg_gam2r, rtol=0.03)
    np.testing.assert_allclose(mean_kgg_vargam2, var_kgg_gam2i, rtol=0.03)

    print('max relerr for gkg gam0r = ',
          np.max(np.abs((var_gkg_gam0r - mean_gkg_vargam0)/var_gkg_gam0r)))
    print('max relerr for gkg gam0i = ',
          np.max(np.abs((var_gkg_gam0i - mean_gkg_vargam0)/var_gkg_gam0i)))
    np.testing.assert_allclose(mean_gkg_vargam0, var_gkg_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_gkg_vargam0, var_gkg_gam0i, rtol=0.03)
    np.testing.assert_allclose(mean_gkg_vargam1, var_gkg_gam1r, rtol=0.03)
    np.testing.assert_allclose(mean_gkg_vargam1, var_gkg_gam1i, rtol=0.03)

    print('max relerr for ggk gam0r = ',
          np.max(np.abs((var_ggk_gam0r - mean_ggk_vargam0)/var_ggk_gam0r)))
    print('max relerr for ggk gam0i = ',
          np.max(np.abs((var_ggk_gam0i - mean_ggk_vargam0)/var_ggk_gam0i)))
    np.testing.assert_allclose(mean_ggk_vargam0, var_ggk_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_ggk_vargam0, var_ggk_gam0i, rtol=0.03)
    np.testing.assert_allclose(mean_ggk_vargam1, var_ggk_gam1r, rtol=0.03)
    np.testing.assert_allclose(mean_ggk_vargam1, var_ggk_gam1i, rtol=0.03)

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
    kgg = treecorr.KGGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')
    gkg = treecorr.GKGCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')
    ggk = treecorr.GGKCorrelation(bin_size=0.5, min_sep=50., max_sep=100.,
                                  sep_units='arcmin', nubins=3, nvbins=3, verbose=1,
                                  bin_type='LogRUV')

    # Before running process, vargam0 and cov are allowed, but all 0.
    np.testing.assert_array_equal(kgg.cov, 0)
    np.testing.assert_array_equal(kgg.vargam0, 0)
    np.testing.assert_array_equal(kgg.vargam2, 0)
    np.testing.assert_array_equal(gkg.cov, 0)
    np.testing.assert_array_equal(gkg.vargam0, 0)
    np.testing.assert_array_equal(gkg.vargam1, 0)
    np.testing.assert_array_equal(ggk.cov, 0)
    np.testing.assert_array_equal(ggk.vargam0, 0)
    np.testing.assert_array_equal(ggk.vargam1, 0)

    kgg.process(cat, cat)
    print('KGG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((kgg.vargam0 - var_kgg_gam0r)/var_kgg_gam0r)))
    print('ratio = ',kgg.vargam0 / var_kgg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((kgg.vargam0 - var_kgg_gam0i)/var_kgg_gam0i)))
    print('ratio = ',kgg.vargam0 / var_kgg_gam0i)
    print('var_num = ',kgg._var_num)
    print('ntri = ',kgg.ntri)
    np.testing.assert_allclose(kgg.vargam0, var_kgg_gam0r, rtol=0.3)
    np.testing.assert_allclose(kgg.vargam0, var_kgg_gam0i, rtol=0.3)
    np.testing.assert_allclose(kgg.vargam2, var_kgg_gam2r, rtol=0.3)
    np.testing.assert_allclose(kgg.vargam2, var_kgg_gam2i, rtol=0.3)
    n = kgg.vargam0.size
    np.testing.assert_allclose(kgg.cov.diagonal()[0:n], kgg.vargam0.ravel())
    np.testing.assert_allclose(kgg.cov.diagonal()[n:2*n], kgg.vargam2.ravel())

    gkg.process(cat, cat, cat)
    print('GKG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((gkg.vargam0 - var_gkg_gam0r)/var_gkg_gam0r)))
    print('ratio = ',gkg.vargam0 / var_gkg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((gkg.vargam0 - var_gkg_gam0i)/var_gkg_gam0i)))
    print('ratio = ',gkg.vargam0 / var_gkg_gam0i)
    np.testing.assert_allclose(gkg.vargam0, var_gkg_gam0r, rtol=0.3)
    np.testing.assert_allclose(gkg.vargam0, var_gkg_gam0i, rtol=0.3)
    np.testing.assert_allclose(gkg.vargam1, var_gkg_gam1r, rtol=0.3)
    np.testing.assert_allclose(gkg.vargam1, var_gkg_gam1i, rtol=0.3)
    np.testing.assert_allclose(gkg.cov.diagonal()[0:n], gkg.vargam0.ravel())
    np.testing.assert_allclose(gkg.cov.diagonal()[n:2*n], gkg.vargam1.ravel())

    ggk.process(cat, cat)
    print('GGK single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ggk.vargam0 - var_ggk_gam0r)/var_ggk_gam0r)))
    print('ratio = ',ggk.vargam0 / var_ggk_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ggk.vargam0 - var_ggk_gam0i)/var_ggk_gam0i)))
    print('ratio = ',ggk.vargam0 / var_ggk_gam0i)
    np.testing.assert_allclose(ggk.vargam0, var_ggk_gam0r, rtol=0.3)
    np.testing.assert_allclose(ggk.vargam0, var_ggk_gam0i, rtol=0.3)
    np.testing.assert_allclose(ggk.vargam1, var_ggk_gam1r, rtol=0.3)
    np.testing.assert_allclose(ggk.vargam1, var_ggk_gam1i, rtol=0.3)
    np.testing.assert_allclose(ggk.cov.diagonal()[0:n], ggk.vargam0.ravel())
    np.testing.assert_allclose(ggk.cov.diagonal()[n:2*n], ggk.vargam1.ravel())

    # Check valid aliases
    # KGG: gam0 = gam1, gam2 = conj(gam3)
    # GKG: gam0 = gam2, gam1 = conj(gam3)
    # GGK: gam0 = gam3, gam1 = conj(gam2)
    np.testing.assert_array_equal(kgg.gam0r, np.real(kgg.gam0))
    np.testing.assert_array_equal(kgg.gam0i, np.imag(kgg.gam0))
    np.testing.assert_array_equal(kgg.gam1r, np.real(kgg.gam1))
    np.testing.assert_array_equal(kgg.gam1i, np.imag(kgg.gam1))
    np.testing.assert_array_equal(kgg.gam2r, np.real(kgg.gam2))
    np.testing.assert_array_equal(kgg.gam2i, np.imag(kgg.gam2))
    np.testing.assert_array_equal(kgg.gam3r, np.real(kgg.gam3))
    np.testing.assert_array_equal(kgg.gam3i, np.imag(kgg.gam3))
    np.testing.assert_array_equal(kgg.gam0, kgg.gam1)
    np.testing.assert_array_equal(kgg.gam0r, kgg.gam1r)
    np.testing.assert_array_equal(kgg.gam0i, kgg.gam1i)
    np.testing.assert_array_equal(kgg.gam2, np.conjugate(kgg.gam3))
    np.testing.assert_array_equal(kgg.gam2r, kgg.gam3r)
    np.testing.assert_array_equal(kgg.gam2i, -kgg.gam3i)
    np.testing.assert_array_equal(kgg.vargam0, kgg.vargam1)
    np.testing.assert_array_equal(kgg.vargam2, kgg.vargam3)

    np.testing.assert_array_equal(gkg.gam0r, np.real(gkg.gam0))
    np.testing.assert_array_equal(gkg.gam0i, np.imag(gkg.gam0))
    np.testing.assert_array_equal(gkg.gam1r, np.real(gkg.gam1))
    np.testing.assert_array_equal(gkg.gam1i, np.imag(gkg.gam1))
    np.testing.assert_array_equal(gkg.gam2r, np.real(gkg.gam2))
    np.testing.assert_array_equal(gkg.gam2i, np.imag(gkg.gam2))
    np.testing.assert_array_equal(gkg.gam3r, np.real(gkg.gam3))
    np.testing.assert_array_equal(gkg.gam3i, np.imag(gkg.gam3))
    np.testing.assert_array_equal(gkg.gam0, gkg.gam2)
    np.testing.assert_array_equal(gkg.gam0r, gkg.gam2r)
    np.testing.assert_array_equal(gkg.gam0i, gkg.gam2i)
    np.testing.assert_array_equal(gkg.gam1, np.conjugate(gkg.gam3))
    np.testing.assert_array_equal(gkg.gam1r, gkg.gam3r)
    np.testing.assert_array_equal(gkg.gam1i, -gkg.gam3i)
    np.testing.assert_array_equal(gkg.vargam0, gkg.vargam2)
    np.testing.assert_array_equal(gkg.vargam1, gkg.vargam3)

    np.testing.assert_array_equal(ggk.gam0r, np.real(ggk.gam0))
    np.testing.assert_array_equal(ggk.gam0i, np.imag(ggk.gam0))
    np.testing.assert_array_equal(ggk.gam1r, np.real(ggk.gam1))
    np.testing.assert_array_equal(ggk.gam1i, np.imag(ggk.gam1))
    np.testing.assert_array_equal(ggk.gam2r, np.real(ggk.gam2))
    np.testing.assert_array_equal(ggk.gam2i, np.imag(ggk.gam2))
    np.testing.assert_array_equal(ggk.gam3r, np.real(ggk.gam3))
    np.testing.assert_array_equal(ggk.gam3i, np.imag(ggk.gam3))
    np.testing.assert_array_equal(ggk.gam0, ggk.gam3)
    np.testing.assert_array_equal(ggk.gam0r, ggk.gam3r)
    np.testing.assert_array_equal(ggk.gam0i, ggk.gam3i)
    np.testing.assert_array_equal(ggk.gam1, np.conjugate(ggk.gam2))
    np.testing.assert_array_equal(ggk.gam1r, ggk.gam2r)
    np.testing.assert_array_equal(ggk.gam1i, -ggk.gam2i)
    np.testing.assert_array_equal(ggk.vargam0, ggk.vargam3)
    np.testing.assert_array_equal(ggk.vargam1, ggk.vargam2)


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
    nbins = 5
    nphi_bins = 3

    # In this test set, we use the slow triangle algorithm.
    # We'll test the multipole algorithm below.
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_gam0_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam2_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam2_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
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

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w3[k]
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                gam0 = www * k1[i] * g2p * g3p
                gam1 = www * k1[i] * np.conjugate(g2p) * g3p

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
                        true_gam0_123[kr2,kr3,kphi] += gam0
                        true_gam2_123[kr2,kr3,kphi] += gam1

                    # 132
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_132[kr3,kr2,kphi] += 1
                        true_weight_132[kr3,kr2,kphi] += www
                        true_gam0_132[kr3,kr2,kphi] += gam0
                        true_gam2_132[kr3,kr2,kphi] += np.conjugate(gam1)

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
                        true_gam0_231[kr3,kr1,kphi] += gam0
                        true_gam1_231[kr3,kr1,kphi] += gam1

                    # 213
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_213[kr1,kr3,kphi] += 1
                        true_weight_213[kr1,kr3,kphi] += www
                        true_gam0_213[kr1,kr3,kphi] += gam0
                        true_gam1_213[kr1,kr3,kphi] += gam1

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
                        true_gam0_312[kr1,kr2,kphi] += gam0
                        true_gam1_312[kr1,kr2,kphi] += np.conjugate(gam1)

                    # 321
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_321[kr2,kr1,kphi] += 1
                        true_weight_321[kr2,kr1,kphi] += www
                        true_gam0_321[kr2,kr1,kphi] += gam0
                        true_gam1_321[kr2,kr1,kphi] += np.conjugate(gam1)

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_gam0_sum1 = true_gam0_123 + true_gam0_132
    true_gam1_sum1 = true_gam2_123 + true_gam2_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_gam0_sum2 = true_gam0_213 + true_gam0_312
    true_gam1_sum2 = true_gam1_213 + true_gam1_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_gam0_sum3 = true_gam0_231 + true_gam0_321
    true_gam1_sum3 = true_gam1_231 + true_gam1_321
    pos = true_weight_sum1 > 0
    true_gam0_sum1[pos] /= true_weight_sum1[pos]
    true_gam1_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_gam0_sum2[pos] /= true_weight_sum2[pos]
    true_gam1_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_gam0_sum3[pos] /= true_weight_sum3[pos]
    true_gam1_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_gam0_123, true_gam0_132, true_gam0_213, true_gam0_231,
               true_gam0_312, true_gam0_321]
    g1_list = [true_gam2_123, true_gam2_132, true_gam1_213, true_gam1_231,
               true_gam1_312, true_gam1_321]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    kgg.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)

    kgg.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_132)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-5)

    gkg.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)

    gkg.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_312)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-5)

    ggk.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    ggk.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_321)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 1
    kgg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-4)

    gkg.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)

    ggk.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # Check binslop = 0
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    kgg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    kgg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1, cat2, cat3, ordered=1, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-4)
    gkg.process(cat2, cat1, cat3, ordered=2, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2, cat3, cat1, ordered=3, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        kgg.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        gkg.process(cat1, cat3=cat1, algo='triangle')
    with assert_raises(ValueError):
        ggk.process(cat3, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    kgg.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_123)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_213)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_231)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    kgg.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(kgg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam1_sum1, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gkg.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_sum2, rtol=1.e-5)
    ggk.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        kgg.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        gkg.process(cat2p, cat1p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        ggk.process(cat2p, cat3p, cat1p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,0.2, (ngal,) )
    g2_2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_gam0_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam0_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam0_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam2_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam1_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam1_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
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

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w2[k]
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                gam0 = www * k1[i] * g2p * g3p
                gam1 = www * k1[i] * np.conjugate(g2p) * g3p

                # 123
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                        gam1 = np.conjugate(gam1)
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_122[kr2,kr3,kphi] += 1
                        true_weight_122[kr2,kr3,kphi] += www
                        true_gam0_122[kr2,kr3,kphi] += gam0
                        true_gam2_122[kr2,kr3,kphi] += gam1

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
                        true_gam0_221[kr3,kr1,kphi] += gam0
                        true_gam1_221[kr3,kr1,kphi] += gam1

                # 312
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    gam1 = np.conjugate(gam1)
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_212[kr1,kr2,kphi] += 1
                        true_weight_212[kr1,kr2,kphi] += www
                        true_gam0_212[kr1,kr2,kphi] += gam0
                        true_gam1_212[kr1,kr2,kphi] += gam1

    w_list = [true_weight_122, true_weight_212, true_weight_221]
    g0_list = [true_gam0_122, true_gam0_212, true_gam0_221]
    g1_list = [true_gam2_122, true_gam1_212, true_gam1_221]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    kgg.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4, atol=1.e-6)
    kgg.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4, atol=1.e-6)

    gkg.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-4, atol=1.e-6)

    ggk.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4, atol=1.e-6)
    ggk.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4, atol=1.e-6)

    with assert_raises(ValueError):
        gkg.process(cat1, cat2)
    with assert_raises(ValueError):
        gkg.process(cat2, cat1)
    with assert_raises(ValueError):
        kgg.process(cat1)
    with assert_raises(ValueError):
        kgg.process(cat2)
    with assert_raises(ValueError):
        gkg.process(cat1)
    with assert_raises(ValueError):
        gkg.process(cat2)
    with assert_raises(ValueError):
        ggk.process(cat1)
    with assert_raises(ValueError):
        ggk.process(cat2)

    # With ordered=False, doesn't do anything difference, since there is no other valid order.
    kgg.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)

    ggk.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)

    kgg.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    gkg.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gkg.ntri, true_ntri_212)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-5)
    ggk.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)

    kgg.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kgg.ntri, true_ntri_122)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-5)
    ggk.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggk.ntri, true_ntri_221)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-5)


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
    max_sep = 30.
    nbins = 5
    max_n = 10

    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_gam0_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_123 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_132 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_213 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_231 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_312 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
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
                    # g2 is projected to the line from c1 to c2
                    # g3 is projected to the line from c1 to c3.
                    expmialpha2 = (x1[i]-x2[j]) - 1j*(y1[i]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = (x1[i]-x3[k]) - 1j*(y1[i]-y3[k])
                    expmialpha3 /= abs(expmialpha3)
                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * k1[i] * g2p * g3p
                    gam1 = www * k1[i] * np.conjugate(g2p) * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_123[kr2,kr3,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam2_123[kr2,kr3,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_123[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_123[kr2,kr3,:] += 1
                    true_gam0_132[kr3,kr2,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam2_132[kr3,kr2,:] += np.conjugate(gam1) * np.exp(1j * n1d * phi)
                    true_weight_132[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_132[kr3,kr2,:] += 1

                # 231, 213
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    # g2 is projected to the average of the two lines.
                    expmialpha3 = (x2[j]-x3[k]) - 1j*(y2[j]-y3[k])
                    expmialpha3 /= abs(expmialpha3)
                    expmialpha1 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = expmialpha1 + expmialpha3
                    expmialpha2 /= abs(expmialpha2)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * g2p * k1[i] * g3p
                    gam1 = www * np.conjugate(g2p) * k1[i] * g3p

                    assert 0 <= kr3 < nbins
                    assert 0 <= kr1 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_231[kr3,kr1,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_231[kr3,kr1,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_231[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_231[kr3,kr1,:] += 1
                    true_gam0_213[kr1,kr3,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam1_213[kr1,kr3,:] += gam1 * np.exp(1j * n1d * phi)
                    true_weight_213[kr1,kr3,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_213[kr1,kr3,:] += 1

                # 312, 321
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g2 is projected to the line from c3 to c2.
                    # g3 is projected to the average of the two lines.
                    expmialpha1 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * g3p * k1[i] * g2p
                    gam1 = www * np.conjugate(g3p) * k1[i] * g2p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_312[kr1,kr2,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_312[kr1,kr2,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_312[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_312[kr1,kr2,:] += 1
                    true_gam0_321[kr2,kr1,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam1_321[kr2,kr1,:] += gam1 * np.exp(1j * n1d * phi)
                    true_weight_321[kr2,kr1,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_321[kr2,kr1,:] += 1

    kgg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kgg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-4)
    kgg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-4)
    gkg.process(cat2, cat1, cat3)
    np.testing.assert_allclose(gkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-4)
    gkg.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-4)
    ggk.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-4)
    ggk.process(cat3, cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-4)

    # Repeat with binslop = 0
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kgg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(kgg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-4)
    kgg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-4)

    gkg.process(cat2, cat1, cat3)
    np.testing.assert_allclose(gkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-4)
    gkg.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-4)

    ggk.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-4)
    ggk.process(cat3, cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    kgg.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(kgg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-4)
    kgg.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-4)

    gkg.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(gkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-4)
    gkg.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-4)

    ggk.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-4)
    ggk.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-4)

    # Now use all three patched
    kgg.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(kgg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_123, rtol=1.e-4)
    kgg.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(kgg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_132, rtol=1.e-4)

    gkg.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(gkg.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_213, rtol=1.e-4)
    gkg.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(gkg.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_312, rtol=1.e-4)

    ggk.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(ggk.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_231, rtol=1.e-4)
    ggk.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(ggk.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        kgg.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        gkg.process(cat2p, cat1p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        ggk.process(cat2p, cat3p, cat1p, patch_method='global')

    # Test I/O
    for name, corr in zip(['kgg', 'gkg', 'ggk'], [kgg, gkg, ggk]):
        ascii_name = 'output/'+name+'_ascii_logmultipole.txt'
        corr.write(ascii_name, precision=16)
        corr2 = treecorr.Corr3.from_file(ascii_name)
        np.testing.assert_allclose(corr2.ntri, corr.ntri)
        np.testing.assert_allclose(corr2.weight, corr.weight)
        np.testing.assert_allclose(corr2.gam0, corr.gam0)
        np.testing.assert_allclose(corr2.gam1, corr.gam1)
        np.testing.assert_allclose(corr2.gam2, corr.gam2)
        np.testing.assert_allclose(corr2.gam3, corr.gam3)
        np.testing.assert_allclose(corr2.meand1, corr.meand1)
        np.testing.assert_allclose(corr2.meand2, corr.meand2)
        np.testing.assert_allclose(corr2.meand3, corr.meand3)
        np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
        np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
        np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)


@timer
def test_direct_logmultipole_cross12():
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

    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_gam0_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_122 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_212 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_221 = np.zeros((nbins, nbins, 2*max_n+1))
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                if k == j: continue
                d1 = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                www = w1[i] * w2[j] * w2[k]

                # 122
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # Rotate shears according to the x projection.  See Porth et al, Figure 1.
                    # g2 is projected to the line from c1 to c2.
                    # g3 is projected to the line from c1 to c3.
                    expmialpha2 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * k1[i] * g2p * g3p
                    gam2 = www * k1[i] * np.conjugate(g2p) * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_122[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_122[kr2,kr3,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam2_122[kr2,kr3,:] += gam2 * np.exp(-1j * n1d * phi)
                    true_ntri_122[kr2,kr3,:] += 1.

                # 221
                if d3 >= min_sep and d3 < max_sep and d1 >= min_sep and d1 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    # g2 is projected to the average of the two lines.
                    expmialpha3 = (x2[j]-x2[k]) - 1j*(y2[j]-y2[k])
                    expmialpha3 /= abs(expmialpha3)
                    expmialpha1 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = expmialpha1 + expmialpha3
                    expmialpha2 /= abs(expmialpha2)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * g2p * g3p * k1[i]
                    gam1 = www * np.conjugate(g2p) * g3p * k1[i]

                    assert 0 <= kr3 < nbins
                    assert 0 <= kr1 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_221[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_221[kr3,kr1,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_221[kr3,kr1,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_ntri_221[kr3,kr1,:] += 1.

                # 212
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g2 is projected to the line from c3 to c2.
                    # g3 is projected to the average of the two lines.
                    expmialpha1 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x2[k]-x2[j]) - 1j*(y2[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * g3p * k1[i] * g2p
                    gam1 = www * np.conjugate(g3p) * k1[i] * g2p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_212[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_212[kr1,kr2,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_212[kr1,kr2,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_ntri_212[kr1,kr2,:] += 1.

    gkg.process(cat2, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-4)
    ggk.process(cat2, cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4)
    # 3 arg version doesn't work for kgg because the kgg processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    kgg.process(cat1, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4)
    ggk.process(cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4)

    # Repeat with binslop = 0
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kgg.process(cat1, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4)

    gkg.process(cat2, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-4)

    ggk.process(cat2, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    kgg.process(cat1p, cat2)
    np.testing.assert_allclose(kgg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4)

    gkg.process(cat2p, cat1, cat2)
    np.testing.assert_allclose(gkg.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-4)

    ggk.process(cat2p, cat1)
    np.testing.assert_allclose(ggk.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4)

    # Now use both patched
    kgg.process(cat1p, cat2p)
    np.testing.assert_allclose(kgg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(kgg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(kgg.gam2, true_gam2_122, rtol=1.e-4)

    gkg.process(cat2p, cat1p, cat2p)
    np.testing.assert_allclose(gkg.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gkg.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gkg.gam1, true_gam1_212, rtol=1.e-4)

    ggk.process(cat2p, cat1p)
    np.testing.assert_allclose(ggk.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggk.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggk.gam1, true_gam1_221, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        kgg.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        gkg.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        ggk.process(cat2p, cat1p, patch_method='global')


@timer
def test_kgg_logsas():
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    # And kappa(r) = kappa0 exp(-r^2/2r0^2)
    #
    # Doing the direct integral yields
    # gam0 = int(int( g(x+x1,y+y1) g(x+x2,y+y2) k(x-x1-x2,y-y1-y2) (x1-Iy1)^2/(x1^2+y1^2) (x2-Iy2)^2/(x2^2+y2^2) ))
    #      = 2pi/3 kappa0 gamma0^2 |q1|^2 |q2|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) / r0^2
    #
    # gam1 = int(int( g(x+x1,y+y1)* g(x+x2,y+y2) k(x-x1-x2,y-y1-y2) (x1+Iy1)^2/(x1^2+y1^2) (x2-Iy2)^2/(x2^2+y2^2) ))
    #      = 2pi/3 kappa0 gamma0^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) / r0^2 x
    #           (q1*^2 q2^2 + 8/3 q1* q2 r0^2 + 8/9 r0^4) q1 q2* / (q1* q2)
    #
    # where the positions are measured relative to the centroid (x,y).
    # If we call the positions relative to the centroid:
    #    q1 = x1 + I y1
    #    q2 = x2 + I y2
    #    q3 = -(x1+x2) - I (y1+y2)

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
        ngal = 10000
        L = 10. * r0
        tol_factor = 3

    rng = np.random.RandomState(12345)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
    k = kappa0 * np.exp(-r2/2.)

    min_sep = 12.
    max_sep = 14.
    nbins = 2
    min_phi = 45
    max_phi = 90
    nphi_bins = 5

    cat = treecorr.Catalog(x=x, y=y, k=k, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kgg = treecorr.KGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')
    gkg = treecorr.GKGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')
    ggk = treecorr.GGKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  sep_units='arcmin', phi_units='degrees', bin_type='LogSAS')

    for name, corr in zip(['kgg', 'gkg', 'ggk'], [kgg, gkg, ggk]):
        t0 = time.time()
        if name == 'gkg':
            corr.process(cat, cat, cat, algo='triangle')
        else:
            corr.process(cat, cat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        # Compute true gam0 based on measured d1,d2,d3 in correlation
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
        q1c = np.conjugate(q1)
        q2c = np.conjugate(q2)
        q3c = np.conjugate(q3)

        L = L - (np.abs(q1) + np.abs(q2) + np.abs(q3))/3.

        true_gam0 = (2.*np.pi * gamma0**2 * kappa0)/(3*L**2*r0**2) * np.exp(-(nq1+nq2+nq3)/(2.*r0**2))

        if name == 'kgg':
            true_gam1 = true_gam0 * (nq2*nq3 + 8/3*q2*q3c*r0**2 + 8/9*q2*q3c*r0**4/(q2c*q3))
            true_gam0 *= nq2 * nq3
            g1 = lambda kgg: kgg.gam2
            cls = treecorr.KGGCorrelation
        elif name == 'gkg':
            true_gam1 = true_gam0 * (nq1*nq3 + 8/3*q1*q3c*r0**2 + 8/9*q1*q3c*r0**4/(q1c*q3))
            true_gam0 *= nq1 * nq3
            g1 = lambda gkg: gkg.gam1
            cls = treecorr.GKGCorrelation
        else:
            true_gam1 = true_gam0 * (nq1*nq2 + 8/3*q1*q2c*r0**2 + 8/9*q1*q2c*r0**4/(q1c*q2))
            true_gam0 *= nq1 * nq2
            g1 = lambda ggk: ggk.gam1
            cls = treecorr.GGKCorrelation

        print('ntri = ',corr.ntri)
        print('gam0 = ',corr.gam0)
        print('true_gam0 = ',true_gam0)
        print('ratio = ',corr.gam0 / true_gam0)
        print('diff = ',corr.gam0 - true_gam0)
        print('max rel diff = ',np.max(np.abs((corr.gam0 - true_gam0)/true_gam0)))
        np.testing.assert_allclose(corr.gam0, true_gam0, rtol=0.2 * tol_factor, atol=1.e-7)
        np.testing.assert_allclose(np.log(np.abs(corr.gam0)),
                                   np.log(np.abs(true_gam0)), atol=0.2 * tol_factor)
        print('gam1 = ',g1(corr))
        print('true_gam1 = ',true_gam1)
        print('ratio = ',g1(corr) / true_gam1)
        print('diff = ',g1(corr) - true_gam1)
        print('max rel diff = ',np.max(np.abs((g1(corr) - true_gam1)/true_gam1)))
        np.testing.assert_allclose(g1(corr), true_gam1, rtol=0.2 * tol_factor, atol=1.e-7)
        np.testing.assert_allclose(np.log(np.abs(g1(corr))),
                                   np.log(np.abs(true_gam1)), atol=0.2 * tol_factor)

        # Repeat this using Multipole and then convert to SAS:
        corrm = cls(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=80,
                    sep_units='arcmin', bin_type='LogMultipole')
        t0 = time.time()
        if name == 'gkg':
            corrm.process(cat, cat, cat)
        else:
            corrm.process(cat, cat)
        corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins, phi_units='deg')
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('gam0 mean ratio = ',np.mean(corrs.gam0 / corr.gam0))
        print('gam0 mean diff = ',np.mean(corrs.gam0 - corr.gam0))
        print('gam1 mean ratio = ',np.mean(g1(corrs) / g1(corr)))
        print('gam1 mean diff = ',np.mean(g1(corrs) - g1(corr)))
        # Some of the individual values are a little ratty, but on average, they are quite close.
        np.testing.assert_allclose(corrs.gam0, corr.gam0, rtol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.gam1, corr.gam1, rtol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.gam2, corr.gam2, rtol=0.2*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam0 / corr.gam0), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam1 / corr.gam1), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam2 / corr.gam2), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam0 / corr.gam0), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam1 / corr.gam1), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam2 / corr.gam2), 0., atol=0.08*tol_factor)
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
        if name == 'gkg':
            corr3.process(cat, cat, cat, algo='multipole', max_n=80)
        else:
            corr3.process(cat, cat, algo='multipole', max_n=80)
        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.gam0, corrs.gam0)
        np.testing.assert_allclose(corr3.gam1, corrs.gam1)
        np.testing.assert_allclose(corr3.gam2, corrs.gam2)

        # Check that we get the same result using the corr3 functin:
        # (This implicitly uses the multipole algorithm.)
        cat.write(os.path.join('data',name+'_data_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['gam0r'], corr3.gam0r.flatten(), rtol=1.e-3, atol=0)
        np.testing.assert_allclose(corr3_output['gam0i'], corr3.gam0i.flatten(), rtol=1.e-3, atol=0)
        if name == 'kgg':
            np.testing.assert_allclose(corr3_output['gam2r'], corr3.gam2r.flatten(),
                                       rtol=1.e-3, atol=0)
            np.testing.assert_allclose(corr3_output['gam2i'], corr3.gam2i.flatten(),
                                       rtol=1.e-3, atol=0)
        else:
            np.testing.assert_allclose(corr3_output['gam1r'], corr3.gam1r.flatten(),
                                       rtol=1.e-3, atol=0)
            np.testing.assert_allclose(corr3_output['gam1i'], corr3.gam1i.flatten(),
                                       rtol=1.e-3, atol=0)

        if name == 'gkg':
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
            out_file_name = os.path.join('output','corr_kgg_logsas.fits')
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
            np.testing.assert_allclose(data['gam0r'], corr.gam0.real.flatten())
            np.testing.assert_allclose(data['gam0i'], corr.gam0.imag.flatten())
            np.testing.assert_allclose(data['sigma_gam0'], np.sqrt(corr.vargam0.flatten()))
            if name == 'kgg':
                np.testing.assert_allclose(data['gam2r'], corr.gam2.real.flatten())
                np.testing.assert_allclose(data['gam2i'], corr.gam2.imag.flatten())
                np.testing.assert_allclose(data['sigma_gam2'], np.sqrt(corr.vargam2.flatten()))
            else:
                np.testing.assert_allclose(data['gam1r'], corr.gam1.real.flatten())
                np.testing.assert_allclose(data['gam1i'], corr.gam1.imag.flatten())
                np.testing.assert_allclose(data['sigma_gam1'], np.sqrt(corr.vargam1.flatten()))
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
            np.testing.assert_allclose(corr2.gam0, corr.gam0)
            np.testing.assert_allclose(corr2.gam1, corr.gam1)
            np.testing.assert_allclose(corr2.gam2, corr.gam2)
            np.testing.assert_allclose(corr2.vargam0, corr.vargam0)
            np.testing.assert_allclose(corr2.vargam1, corr.vargam1)
            np.testing.assert_allclose(corr2.vargam2, corr.vargam2)
            np.testing.assert_allclose(corr2.weight, corr.weight)
            np.testing.assert_allclose(corr2.ntri, corr.ntri)
            assert corr2.coords == corr.coords
            assert corr2.metric == corr.metric
            assert corr2.sep_units == corr.sep_units
            assert corr2.bin_type == corr.bin_type

@timer
def test_vargam():
    # Test that the shot noise estimate of vargam0 is close based on actual variance of many runs
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

    file_name = 'data/test_vargam_kgg.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kggs = []
        all_gkgs = []
        all_ggks = []

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
            kgg = treecorr.KGGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            gkg = treecorr.GKGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            ggk = treecorr.GGKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
            kgg.process(cat, cat)
            gkg.process(cat, cat, cat)
            ggk.process(cat, cat)
            all_kggs.append(kgg)
            all_gkgs.append(gkg)
            all_ggks.append(ggk)

        mean_kgg_gam0r = np.mean([kgg.gam0r for kgg in all_kggs], axis=0)
        mean_kgg_gam0i = np.mean([kgg.gam0i for kgg in all_kggs], axis=0)
        mean_kgg_gam2r = np.mean([kgg.gam2r for kgg in all_kggs], axis=0)
        mean_kgg_gam2i = np.mean([kgg.gam2i for kgg in all_kggs], axis=0)
        var_kgg_gam0r = np.var([kgg.gam0r for kgg in all_kggs], axis=0)
        var_kgg_gam0i = np.var([kgg.gam0i for kgg in all_kggs], axis=0)
        var_kgg_gam2r = np.var([kgg.gam2r for kgg in all_kggs], axis=0)
        var_kgg_gam2i = np.var([kgg.gam2i for kgg in all_kggs], axis=0)
        mean_kgg_vargam0 = np.mean([kgg.vargam0 for kgg in all_kggs], axis=0)
        mean_kgg_vargam2 = np.mean([kgg.vargam2 for kgg in all_kggs], axis=0)
        mean_gkg_gam0r = np.mean([gkg.gam0r for gkg in all_gkgs], axis=0)
        mean_gkg_gam0i = np.mean([gkg.gam0i for gkg in all_gkgs], axis=0)
        mean_gkg_gam1r = np.mean([gkg.gam1r for gkg in all_gkgs], axis=0)
        mean_gkg_gam1i = np.mean([gkg.gam1i for gkg in all_gkgs], axis=0)
        var_gkg_gam0r = np.var([gkg.gam0r for gkg in all_gkgs], axis=0)
        var_gkg_gam0i = np.var([gkg.gam0i for gkg in all_gkgs], axis=0)
        var_gkg_gam1r = np.var([gkg.gam1r for gkg in all_gkgs], axis=0)
        var_gkg_gam1i = np.var([gkg.gam1i for gkg in all_gkgs], axis=0)
        mean_gkg_vargam0 = np.mean([gkg.vargam0 for gkg in all_gkgs], axis=0)
        mean_gkg_vargam1 = np.mean([gkg.vargam1 for gkg in all_gkgs], axis=0)
        mean_ggk_gam0r = np.mean([ggk.gam0r for ggk in all_ggks], axis=0)
        mean_ggk_gam0i = np.mean([ggk.gam0i for ggk in all_ggks], axis=0)
        mean_ggk_gam1r = np.mean([ggk.gam1r for ggk in all_ggks], axis=0)
        mean_ggk_gam1i = np.mean([ggk.gam1i for ggk in all_ggks], axis=0)
        var_ggk_gam0r = np.var([ggk.gam0r for ggk in all_ggks], axis=0)
        var_ggk_gam0i = np.var([ggk.gam0i for ggk in all_ggks], axis=0)
        var_ggk_gam1r = np.var([ggk.gam1r for ggk in all_ggks], axis=0)
        var_ggk_gam1i = np.var([ggk.gam1i for ggk in all_ggks], axis=0)
        mean_ggk_vargam0 = np.mean([ggk.vargam0 for ggk in all_ggks], axis=0)
        mean_ggk_vargam1 = np.mean([ggk.vargam1 for ggk in all_ggks], axis=0)

        np.savez(file_name,
                 mean_kgg_gam0r=mean_kgg_gam0r,
                 mean_kgg_gam0i=mean_kgg_gam0i,
                 mean_kgg_gam2r=mean_kgg_gam2r,
                 mean_kgg_gam2i=mean_kgg_gam2i,
                 var_kgg_gam0r=var_kgg_gam0r,
                 var_kgg_gam0i=var_kgg_gam0i,
                 var_kgg_gam2r=var_kgg_gam2r,
                 var_kgg_gam2i=var_kgg_gam2i,
                 mean_kgg_vargam0=mean_kgg_vargam0,
                 mean_kgg_vargam2=mean_kgg_vargam2,
                 mean_gkg_gam0r=mean_gkg_gam0r,
                 mean_gkg_gam0i=mean_gkg_gam0i,
                 mean_gkg_gam1r=mean_gkg_gam1r,
                 mean_gkg_gam1i=mean_gkg_gam1i,
                 var_gkg_gam0r=var_gkg_gam0r,
                 var_gkg_gam0i=var_gkg_gam0i,
                 var_gkg_gam1r=var_gkg_gam1r,
                 var_gkg_gam1i=var_gkg_gam1i,
                 mean_gkg_vargam0=mean_gkg_vargam0,
                 mean_gkg_vargam1=mean_gkg_vargam1,
                 mean_ggk_gam0r=mean_ggk_gam0r,
                 mean_ggk_gam0i=mean_ggk_gam0i,
                 mean_ggk_gam1r=mean_ggk_gam1r,
                 mean_ggk_gam1i=mean_ggk_gam1i,
                 var_ggk_gam0r=var_ggk_gam0r,
                 var_ggk_gam0i=var_ggk_gam0i,
                 var_ggk_gam1r=var_ggk_gam1r,
                 var_ggk_gam1i=var_ggk_gam1i,
                 mean_ggk_vargam0=mean_ggk_vargam0,
                 mean_ggk_vargam1=mean_ggk_vargam1)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_kgg_gam0r = data['mean_kgg_gam0r']
    mean_kgg_gam0i = data['mean_kgg_gam0i']
    mean_kgg_gam2r = data['mean_kgg_gam2r']
    mean_kgg_gam2i = data['mean_kgg_gam2i']
    var_kgg_gam0r = data['var_kgg_gam0r']
    var_kgg_gam0i = data['var_kgg_gam0i']
    var_kgg_gam2r = data['var_kgg_gam2r']
    var_kgg_gam2i = data['var_kgg_gam2i']
    mean_kgg_vargam0 = data['mean_kgg_vargam0']
    mean_kgg_vargam2 = data['mean_kgg_vargam2']
    mean_gkg_gam0r = data['mean_gkg_gam0r']
    mean_gkg_gam0i = data['mean_gkg_gam0i']
    mean_gkg_gam1r = data['mean_gkg_gam1r']
    mean_gkg_gam1i = data['mean_gkg_gam1i']
    var_gkg_gam0r = data['var_gkg_gam0r']
    var_gkg_gam0i = data['var_gkg_gam0i']
    var_gkg_gam1r = data['var_gkg_gam1r']
    var_gkg_gam1i = data['var_gkg_gam1i']
    mean_gkg_vargam0 = data['mean_gkg_vargam0']
    mean_gkg_vargam1 = data['mean_gkg_vargam1']
    mean_ggk_gam0r = data['mean_ggk_gam0r']
    mean_ggk_gam0i = data['mean_ggk_gam0i']
    mean_ggk_gam1r = data['mean_ggk_gam1r']
    mean_ggk_gam1i = data['mean_ggk_gam1i']
    var_ggk_gam0r = data['var_ggk_gam0r']
    var_ggk_gam0i = data['var_ggk_gam0i']
    var_ggk_gam1r = data['var_ggk_gam1r']
    var_ggk_gam1i = data['var_ggk_gam1i']
    mean_ggk_vargam0 = data['mean_ggk_vargam0']
    mean_ggk_vargam1 = data['mean_ggk_vargam1']

    print('var_kgg_gam0r = ',var_kgg_gam0r)
    print('mean kgg_vargam0 = ',mean_kgg_vargam0)
    print('ratio = ',var_kgg_gam0r.ravel() / mean_kgg_vargam0.ravel())
    print('var_gkg_gam0r = ',var_gkg_gam0r)
    print('mean gkg_vargam0 = ',mean_gkg_vargam0)
    print('ratio = ',var_gkg_gam0r.ravel() / mean_gkg_vargam0.ravel())
    print('var_ggk_gam0r = ',var_ggk_gam0r)
    print('mean ggk_vargam0 = ',mean_ggk_vargam0)
    print('ratio = ',var_ggk_gam0r.ravel() / mean_ggk_vargam0.ravel())

    print('max relerr for kgg gam0r = ',
          np.max(np.abs((var_kgg_gam0r - mean_kgg_vargam0)/var_kgg_gam0r)))
    print('max relerr for kgg gam0i = ',
          np.max(np.abs((var_kgg_gam0i - mean_kgg_vargam0)/var_kgg_gam0i)))
    np.testing.assert_allclose(mean_kgg_vargam0, var_kgg_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_kgg_vargam0, var_kgg_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_kgg_vargam2, var_kgg_gam2r, rtol=0.1)
    np.testing.assert_allclose(mean_kgg_vargam2, var_kgg_gam2i, rtol=0.1)

    print('max relerr for gkg gam0r = ',
          np.max(np.abs((var_gkg_gam0r - mean_gkg_vargam0)/var_gkg_gam0r)))
    print('max relerr for gkg gam0i = ',
          np.max(np.abs((var_gkg_gam0i - mean_gkg_vargam0)/var_gkg_gam0i)))
    np.testing.assert_allclose(mean_gkg_vargam0, var_gkg_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_gkg_vargam0, var_gkg_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_gkg_vargam1, var_gkg_gam1r, rtol=0.1)
    np.testing.assert_allclose(mean_gkg_vargam1, var_gkg_gam1i, rtol=0.1)

    print('max relerr for ggk gam0r = ',
          np.max(np.abs((var_ggk_gam0r - mean_ggk_vargam0)/var_ggk_gam0r)))
    print('max relerr for ggk gam0i = ',
          np.max(np.abs((var_ggk_gam0i - mean_ggk_vargam0)/var_ggk_gam0i)))
    np.testing.assert_allclose(mean_ggk_vargam0, var_ggk_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_ggk_vargam0, var_ggk_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_ggk_vargam1, var_ggk_gam1r, rtol=0.1)
    np.testing.assert_allclose(mean_ggk_vargam1, var_ggk_gam1i, rtol=0.1)

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
    kgg = treecorr.KGGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
    gkg = treecorr.GKGCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)
    ggk = treecorr.GGKCorrelation(nbins=3, min_sep=30., max_sep=50., nphi_bins=20)

    # Before running process, vargam0 and cov are allowed, but all 0.
    np.testing.assert_array_equal(kgg.cov, 0)
    np.testing.assert_array_equal(kgg.vargam0, 0)
    np.testing.assert_array_equal(kgg.vargam2, 0)
    np.testing.assert_array_equal(gkg.cov, 0)
    np.testing.assert_array_equal(gkg.vargam0, 0)
    np.testing.assert_array_equal(gkg.vargam1, 0)
    np.testing.assert_array_equal(ggk.cov, 0)
    np.testing.assert_array_equal(ggk.vargam0, 0)
    np.testing.assert_array_equal(ggk.vargam1, 0)

    kgg.process(cat, cat)
    print('KGG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((kgg.vargam0 - var_kgg_gam0r)/var_kgg_gam0r)))
    print('ratio = ',kgg.vargam0 / var_kgg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((kgg.vargam0 - var_kgg_gam0i)/var_kgg_gam0i)))
    print('ratio = ',kgg.vargam0 / var_kgg_gam0i)
    print('var_num = ',kgg._var_num)
    print('ntri = ',kgg.ntri)
    np.testing.assert_allclose(kgg.vargam0, var_kgg_gam0r, rtol=0.2)
    np.testing.assert_allclose(kgg.vargam0, var_kgg_gam0i, rtol=0.2)
    np.testing.assert_allclose(kgg.vargam2, var_kgg_gam2r, rtol=0.2)
    np.testing.assert_allclose(kgg.vargam2, var_kgg_gam2i, rtol=0.2)
    n = kgg.vargam0.size
    np.testing.assert_allclose(kgg.cov.diagonal()[0:n], kgg.vargam0.ravel())
    np.testing.assert_allclose(kgg.cov.diagonal()[n:2*n], kgg.vargam2.ravel())

    gkg.process(cat, cat, cat)
    print('GKG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((gkg.vargam0 - var_gkg_gam0r)/var_gkg_gam0r)))
    print('ratio = ',gkg.vargam0 / var_gkg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((gkg.vargam0 - var_gkg_gam0i)/var_gkg_gam0i)))
    print('ratio = ',gkg.vargam0 / var_gkg_gam0i)
    np.testing.assert_allclose(gkg.vargam0, var_gkg_gam0r, rtol=0.2)
    np.testing.assert_allclose(gkg.vargam0, var_gkg_gam0i, rtol=0.2)
    np.testing.assert_allclose(gkg.vargam1, var_gkg_gam1r, rtol=0.2)
    np.testing.assert_allclose(gkg.vargam1, var_gkg_gam1i, rtol=0.2)
    np.testing.assert_allclose(gkg.cov.diagonal()[0:n], gkg.vargam0.ravel())
    np.testing.assert_allclose(gkg.cov.diagonal()[n:2*n], gkg.vargam1.ravel())

    ggk.process(cat, cat)
    print('GGK single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ggk.vargam0 - var_ggk_gam0r)/var_ggk_gam0r)))
    print('ratio = ',ggk.vargam0 / var_ggk_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ggk.vargam0 - var_ggk_gam0i)/var_ggk_gam0i)))
    print('ratio = ',ggk.vargam0 / var_ggk_gam0i)
    np.testing.assert_allclose(ggk.vargam0, var_ggk_gam0r, rtol=0.2)
    np.testing.assert_allclose(ggk.vargam0, var_ggk_gam0i, rtol=0.2)
    np.testing.assert_allclose(ggk.vargam1, var_ggk_gam1r, rtol=0.2)
    np.testing.assert_allclose(ggk.vargam1, var_ggk_gam1i, rtol=0.2)
    np.testing.assert_allclose(ggk.cov.diagonal()[0:n], ggk.vargam0.ravel())
    np.testing.assert_allclose(ggk.cov.diagonal()[n:2*n], ggk.vargam1.ravel())

@timer
def test_kgg_logsas_jk():
    # Test jackknife covariance estimates for kgg correlations with LogSAS binning.

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    if __name__ == '__main__':
        nhalo = 1000
        nsource = 100000
        npatch = 64
        tol_factor = 1
    else:
        nhalo = 500
        nsource = 10000
        npatch = 20
        tol_factor = 4

    nbins = 2
    min_sep = 10
    max_sep = 16
    nphi_bins = 2
    min_phi = 30
    max_phi = 90

    file_name = 'data/test_kgg_logsas_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_kgg0 = []
        all_kgg1 = []
        all_gkg0 = []
        all_gkg1 = []
        all_ggk0 = []
        all_ggk1 = []
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
            kgg = treecorr.KGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            kgg.process(kcat, gcat)
            all_kgg0.append(kgg.gam0.ravel())
            all_kgg1.append(kgg.gam2.ravel())

            gkg = treecorr.GKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            gkg.process(gcat, kcat, gcat)
            all_gkg0.append(gkg.gam0.ravel())
            all_gkg1.append(gkg.gam1.ravel())

            ggk = treecorr.GGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            ggk.process(gcat, kcat)
            all_ggk0.append(ggk.gam0.ravel())
            all_ggk1.append(ggk.gam1.ravel())

        mean_kgg0 = np.mean([gam0 for gam0 in all_kgg0], axis=0)
        mean_kgg1 = np.mean([gam1 for gam1 in all_kgg1], axis=0)
        var_kgg0 = np.var([gam0 for gam0 in all_kgg0], axis=0)
        var_kgg1 = np.var([gam1 for gam1 in all_kgg1], axis=0)
        mean_gkg0 = np.mean([gam0 for gam0 in all_gkg0], axis=0)
        mean_gkg1 = np.mean([gam1 for gam1 in all_gkg1], axis=0)
        var_gkg0 = np.var([gam0 for gam0 in all_gkg0], axis=0)
        var_gkg1 = np.var([gam1 for gam1 in all_gkg1], axis=0)
        mean_ggk0 = np.mean([gam0 for gam0 in all_ggk0], axis=0)
        mean_ggk1 = np.mean([gam1 for gam1 in all_ggk1], axis=0)
        var_ggk0 = np.var([gam0 for gam0 in all_ggk0], axis=0)
        var_ggk1 = np.var([gam1 for gam1 in all_ggk1], axis=0)

        np.savez(file_name,
                 mean_kgg0=mean_kgg0, var_kgg0=var_kgg0,
                 mean_kgg1=mean_kgg1, var_kgg1=var_kgg1,
                 mean_gkg0=mean_gkg0, var_gkg0=var_gkg0,
                 mean_gkg1=mean_gkg1, var_gkg1=var_gkg1,
                 mean_ggk0=mean_ggk0, var_ggk0=var_ggk0,
                 mean_ggk1=mean_ggk1, var_ggk1=var_ggk1)

    data = np.load(file_name)
    mean_kgg0 = data['mean_kgg0']
    mean_kgg1 = data['mean_kgg1']
    var_kgg0 = data['var_kgg0']
    var_kgg1 = data['var_kgg1']
    mean_gkg0 = data['mean_gkg0']
    mean_gkg1 = data['mean_gkg1']
    var_gkg0 = data['var_gkg0']
    var_gkg1 = data['var_gkg1']
    mean_ggk0 = data['mean_ggk0']
    mean_ggk1 = data['mean_ggk1']
    var_ggk0 = data['var_ggk0']
    var_ggk1 = data['var_ggk1']
    print('mean kgg0 = ',mean_kgg0)
    print('mean kgg1 = ',mean_kgg1)
    print('var kgg0 = ',var_kgg0)
    print('var kgg1 = ',var_kgg1)
    print('mean gkg0 = ',mean_gkg0)
    print('mean gkg1 = ',mean_gkg1)
    print('var gkg0 = ',var_gkg0)
    print('var gkg1 = ',var_gkg1)
    print('mean ggk0 = ',mean_ggk0)
    print('mean ggk1 = ',mean_ggk1)
    print('var ggk0 = ',var_ggk0)
    print('var ggk1 = ',var_ggk1)

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
    kgg = treecorr.KGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    kgg.process(kcat, gcat)
    kgg2 = kgg.copy()
    kgg2._calculate_xi_from_pairs(list(kgg.results.keys()))
    np.testing.assert_allclose(kgg2.ntri, kgg.ntri, rtol=0.01)
    np.testing.assert_allclose(kgg2.gam0, kgg.gam0, rtol=0.01)
    np.testing.assert_allclose(kgg2.gam2, kgg.gam2, rtol=0.01)
    np.testing.assert_allclose(kgg2.vargam0, kgg.vargam0, rtol=0.01)
    np.testing.assert_allclose(kgg2.vargam2, kgg.vargam2, rtol=0.01)

    gkg = treecorr.GKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    gkg.process(gcat, kcat, gcat)
    gkg2 = gkg.copy()
    gkg2._calculate_xi_from_pairs(list(gkg.results.keys()))
    np.testing.assert_allclose(gkg2.ntri, gkg.ntri, rtol=0.01)
    np.testing.assert_allclose(gkg2.gam0, gkg.gam0, rtol=0.01)
    np.testing.assert_allclose(gkg2.gam1, gkg.gam1, rtol=0.01)
    np.testing.assert_allclose(gkg2.vargam0, gkg.vargam0, rtol=0.01)
    np.testing.assert_allclose(gkg2.vargam1, gkg.vargam1, rtol=0.01)

    ggk = treecorr.GGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    ggk.process(gcat, kcat)
    ggk2 = ggk.copy()
    ggk2._calculate_xi_from_pairs(list(ggk.results.keys()))
    np.testing.assert_allclose(ggk2.ntri, ggk.ntri, rtol=0.01)
    np.testing.assert_allclose(ggk2.gam0, ggk.gam0, rtol=0.01)
    np.testing.assert_allclose(ggk2.gam1, ggk.gam1, rtol=0.01)
    np.testing.assert_allclose(ggk2.vargam0, ggk.vargam0, rtol=0.01)
    np.testing.assert_allclose(ggk2.vargam1, ggk.vargam1, rtol=0.01)

    # Next check jackknife covariance estimate
    cov_kgg = kgg.estimate_cov('jackknife')
    n = kgg.vargam0.size
    print('kgg var ratio = ',np.diagonal(cov_kgg)[0:n]/var_kgg0)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_kgg)[0:n])-np.log(var_kgg0))))
    np.testing.assert_allclose(np.diagonal(cov_kgg)[0:n], var_kgg0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_kgg)[0:n]), np.log(var_kgg0), atol=0.5*tol_factor)
    np.testing.assert_allclose(np.diagonal(cov_kgg)[n:2*n], var_kgg1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_kgg)[n:2*n]), np.log(var_kgg1), atol=0.5*tol_factor)

    cov_gkg = gkg.estimate_cov('jackknife')
    print('gkg var ratio = ',np.diagonal(cov_gkg)[0:n]/var_gkg0)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_gkg)[0:n])-np.log(var_gkg0))))
    np.testing.assert_allclose(np.diagonal(cov_gkg)[0:n], var_gkg0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_gkg)[0:n]), np.log(var_gkg0), atol=0.5*tol_factor)
    np.testing.assert_allclose(np.diagonal(cov_gkg)[n:2*n], var_gkg1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_gkg)[n:2*n]), np.log(var_gkg1), atol=0.5*tol_factor)

    cov_ggk = ggk.estimate_cov('jackknife')
    print('ggk var ratio = ',np.diagonal(cov_ggk)[0:n]/var_ggk0)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_ggk)[0:n])-np.log(var_ggk0))))
    np.testing.assert_allclose(np.diagonal(cov_ggk)[0:n], var_ggk0, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_ggk)[0:n]), np.log(var_ggk0), atol=0.7*tol_factor)
    np.testing.assert_allclose(np.diagonal(cov_ggk)[n:2*n], var_ggk1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov_ggk)[n:2*n]), np.log(var_ggk1), atol=0.7*tol_factor)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_kgg.dat')
    kgg.write(file_name, write_patch_results=True)
    kgg2 = treecorr.Corr3.from_file(file_name)
    cov2 = kgg2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_kgg)

    file_name = os.path.join('output','test_write_results_gkg.dat')
    gkg.write(file_name, write_patch_results=True)
    gkg2 = treecorr.Corr3.from_file(file_name)
    cov2 = gkg2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_gkg)

    file_name = os.path.join('output','test_write_results_ggk.dat')
    ggk.write(file_name, write_patch_results=True)
    ggk2 = treecorr.Corr3.from_file(file_name)
    cov2 = ggk2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_ggk)

    # Check jackknife using LogMultipole
    print('Using LogMultipole:')
    kggm = treecorr.KGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    kggm.process(kcat, gcat)
    fm0 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam0.ravel()
    fm1 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam1.ravel()
    fm2 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam2.ravel()
    cov = kggm.estimate_cov('jackknife', func=fm0)
    print('kgg0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kgg0))))
    np.testing.assert_allclose(np.diagonal(cov), var_kgg0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kgg0), atol=0.5*tol_factor)
    cov = kggm.estimate_cov('jackknife', func=fm2)
    print('kgg1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kgg1))))
    np.testing.assert_allclose(np.diagonal(cov), var_kgg1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kgg1), atol=0.5*tol_factor)

    gkgm = treecorr.GKGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    gkgm.process(gcat, kcat, gcat)
    cov = gkgm.estimate_cov('jackknife', func=fm0)
    print('gkg0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gkg0))))
    np.testing.assert_allclose(np.diagonal(cov), var_gkg0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gkg0), atol=0.5*tol_factor)
    cov = gkgm.estimate_cov('jackknife', func=fm1)
    print('gkg1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gkg1))))
    np.testing.assert_allclose(np.diagonal(cov), var_gkg1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gkg1), atol=0.5*tol_factor)

    ggkm = treecorr.GGKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    ggkm.process(gcat, kcat)
    cov = ggkm.estimate_cov('jackknife', func=fm0)
    print('ggk0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggk0))))
    np.testing.assert_allclose(np.diagonal(cov), var_ggk0, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggk0), atol=0.7*tol_factor)
    cov = ggkm.estimate_cov('jackknife', func=fm1)
    print('ggk1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggk1))))
    np.testing.assert_allclose(np.diagonal(cov), var_ggk1, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggk1), atol=0.7*tol_factor)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross12()
    test_vargam_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross12()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross12()
    test_kgg_logsas()
    test_vargam()
    test_kgg_logsas_jk()
