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
    sig_kap = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,sig_kap, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal)
    k3 = rng.normal(0,sig_kap, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    knk = treecorr.KNKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    kkn = treecorr.KKNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
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
    true_zeta_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_321 = np.zeros((nrbins, nubins, 2*nvbins))
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
                zeta = www * k2[j] * k3[k]

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_zeta_sum1 = true_zeta_123 + true_zeta_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_zeta_sum2 = true_zeta_213 + true_zeta_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_zeta_sum3 = true_zeta_231 + true_zeta_321
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

    nkk.process(cat1, cat2, cat3)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)

    nkk.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_132)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-5)

    knk.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    knk.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(knk.ntri, true_ntri_312)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-5)

    kkn.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)
    kkn.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_321)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 3
    nkk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)

    knk.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # Check bin_slop=0
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nkk.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nkk.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # With these, ordered=False is equivalent to the K vertex being fixed.
    nkk.process(cat1, cat2, cat3, ordered=1)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=2)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=3)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nkk.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    nkk.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2p, cat1, cat3)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2p, cat3, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1p, cat3)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3p, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1, cat3p)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1p)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    # Now all three patched
    nkk.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p)
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    # Unordered
    nkk.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=False)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=1)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=2)
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=3)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # patch_method=local
    nkk.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, patch_method='local')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, patch_method='local')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        nkk.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        knk.process(cat2p, cat1p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        kkn.process(cat2p, cat3p, cat1p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    sig_kap = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,sig_kap, (ngal,) )
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

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    knk = treecorr.KNKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    kkn = treecorr.KKNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_221 = np.zeros((nrbins, nubins, 2*nvbins))
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
                zeta = www * k2[j] * k2[k]

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    pos = true_weight_221 > 0
    true_zeta_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_zeta_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_zeta_122[pos] /= true_weight_122[pos]

    nkk.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-5)
    kkn.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: KNK doesn't have a two-argument version.
    nkk.process(cat1, cat2)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2, cat1)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    with assert_raises(ValueError):
        knk.process(cat1, cat2)
    with assert_raises(ValueError):
        knk.process(cat2, cat1)
    with assert_raises(ValueError):
        nkk.process(cat1)
    with assert_raises(ValueError):
        nkk.process(cat2)
    with assert_raises(ValueError):
        knk.process(cat1)
    with assert_raises(ValueError):
        knk.process(cat2)
    with assert_raises(ValueError):
        kkn.process(cat1)
    with assert_raises(ValueError):
        kkn.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    nkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    # Repeat with binslop = 0
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    nkk.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-5)
    kkn.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    # And again with no top-level recursion
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nkk.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-5)
    kkn.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)

    nkk.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)


@timer
def test_varzeta_logruv():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for k, but this pattern doesn't have much 3pt correlation.
    kappa0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(12345)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 30
    nsource = 5000

    file_name = 'data/test_varzeta_nkk_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nkks = []
        all_knks = []
        all_kkns = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x1 = (rng.random_sample(nlens)-0.5) * L
            y1 = (rng.random_sample(nlens)-0.5) * L
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5
            r2 = (x2**2 + y2**2)/r0**2
            k = kappa0 * np.exp(-r2/2.)
            k += rng.normal(0, 0.2, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
            kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k,
                                    x_units='arcmin', y_units='arcmin')
            nkk = treecorr.NKKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            knk = treecorr.KNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            kkn = treecorr.KKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            nkk.process(ncat, kcat)
            knk.process(kcat, ncat, kcat)
            kkn.process(kcat, ncat)
            all_nkks.append(nkk)
            all_knks.append(knk)
            all_kkns.append(kkn)

        mean_nkk_zeta = np.mean([nkk.zeta for nkk in all_nkks], axis=0)
        var_nkk_zeta = np.var([nkk.zeta for nkk in all_nkks], axis=0)
        mean_nkk_varzeta = np.mean([nkk.varzeta for nkk in all_nkks], axis=0)
        mean_knk_zeta = np.mean([knk.zeta for knk in all_knks], axis=0)
        var_knk_zeta = np.var([knk.zeta for knk in all_knks], axis=0)
        mean_knk_varzeta = np.mean([knk.varzeta for knk in all_knks], axis=0)
        mean_kkn_zeta = np.mean([kkn.zeta for kkn in all_kkns], axis=0)
        var_kkn_zeta = np.var([kkn.zeta for kkn in all_kkns], axis=0)
        mean_kkn_varzeta = np.mean([kkn.varzeta for kkn in all_kkns], axis=0)

        np.savez(file_name,
                 mean_nkk_zeta=mean_nkk_zeta,
                 var_nkk_zeta=var_nkk_zeta,
                 mean_nkk_varzeta=mean_nkk_varzeta,
                 mean_knk_zeta=mean_knk_zeta,
                 var_knk_zeta=var_knk_zeta,
                 mean_knk_varzeta=mean_knk_varzeta,
                 mean_kkn_zeta=mean_kkn_zeta,
                 var_kkn_zeta=var_kkn_zeta,
                 mean_kkn_varzeta=mean_kkn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nkk_zeta = data['mean_nkk_zeta']
    var_nkk_zeta = data['var_nkk_zeta']
    mean_nkk_varzeta = data['mean_nkk_varzeta']
    mean_knk_zeta = data['mean_knk_zeta']
    var_knk_zeta = data['var_knk_zeta']
    mean_knk_varzeta = data['mean_knk_varzeta']
    mean_kkn_zeta = data['mean_kkn_zeta']
    var_kkn_zeta = data['var_kkn_zeta']
    mean_kkn_varzeta = data['mean_kkn_varzeta']

    print('var_nkk_zeta = ',var_nkk_zeta)
    print('mean nkk_varzeta = ',mean_nkk_varzeta)
    print('ratio = ',var_nkk_zeta.ravel() / mean_nkk_varzeta.ravel())
    print('var_knk_zeta = ',var_knk_zeta)
    print('mean knk_varzeta = ',mean_knk_varzeta)
    print('ratio = ',var_knk_zeta.ravel() / mean_knk_varzeta.ravel())
    print('var_kkn_zeta = ',var_kkn_zeta)
    print('mean kkn_varzeta = ',mean_kkn_varzeta)
    print('ratio = ',var_kkn_zeta.ravel() / mean_kkn_varzeta.ravel())

    print('max relerr for nkk zeta = ',
          np.max(np.abs((var_nkk_zeta - mean_nkk_varzeta)/var_nkk_zeta)))
    np.testing.assert_allclose(mean_nkk_varzeta, var_nkk_zeta, rtol=0.05)

    print('max relerr for knk zeta = ',
          np.max(np.abs((var_knk_zeta - mean_knk_varzeta)/var_knk_zeta)))
    np.testing.assert_allclose(mean_knk_varzeta, var_knk_zeta, rtol=0.05)

    print('max relerr for kkn zeta = ',
          np.max(np.abs((var_kkn_zeta - mean_kkn_varzeta)/var_kkn_zeta)))
    np.testing.assert_allclose(mean_kkn_varzeta, var_kkn_zeta, rtol=0.05)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    k = kappa0 * np.exp(-r2/2.)
    k += rng.normal(0, 0.2, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
    kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k, x_units='arcmin', y_units='arcmin')
    nkk = treecorr.NKKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')
    knk = treecorr.KNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')
    kkn = treecorr.KKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nkk.cov, 0)
    np.testing.assert_array_equal(nkk.varzeta, 0)
    np.testing.assert_array_equal(knk.cov, 0)
    np.testing.assert_array_equal(knk.varzeta, 0)
    np.testing.assert_array_equal(kkn.cov, 0)
    np.testing.assert_array_equal(kkn.varzeta, 0)

    nkk.process(ncat, kcat)
    print('NKK single run:')
    print('max relerr for zeta = ',np.max(np.abs((nkk.varzeta - var_nkk_zeta)/var_nkk_zeta)))
    print('ratio = ',nkk.varzeta / var_nkk_zeta)
    np.testing.assert_allclose(nkk.varzeta, var_nkk_zeta, rtol=0.7)
    np.testing.assert_allclose(nkk.cov.diagonal(), nkk.varzeta.ravel())

    knk.process(kcat, ncat, kcat)
    print('KNK single run:')
    print('max relerr for zeta = ',np.max(np.abs((knk.varzeta - var_knk_zeta)/var_knk_zeta)))
    print('ratio = ',knk.varzeta / var_knk_zeta)
    np.testing.assert_allclose(knk.varzeta, var_knk_zeta, rtol=0.7)
    np.testing.assert_allclose(knk.cov.diagonal(), knk.varzeta.ravel())

    kkn.process(kcat, ncat)
    print('KKN single run:')
    print('max relerr for zeta = ',np.max(np.abs((kkn.varzeta - var_kkn_zeta)/var_kkn_zeta)))
    print('ratio = ',kkn.varzeta / var_kkn_zeta)
    np.testing.assert_allclose(kkn.varzeta, var_kkn_zeta, rtol=0.7)
    np.testing.assert_allclose(kkn.cov.diagonal(), kkn.varzeta.ravel())



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
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal).astype(np.float32)
    k2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal).astype(np.float32)
    k3 = rng.normal(0,0.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    max_sep = 10.
    nbins = 3
    nphi_bins = 5

    # In this test set, we use the slow triangle algorithm.
    # We'll test the multipole algorithm below.
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_123 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_132 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_213 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_231 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_312 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_321 = np.zeros((nbins, nbins, nphi_bins))
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

                www = w1[i] * w2[j] * w3[k]
                zeta = www * k2[j] * k3[k]

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

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_zeta_sum1 = true_zeta_123 + true_zeta_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_zeta_sum2 = true_zeta_213 + true_zeta_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_zeta_sum3 = true_zeta_231 + true_zeta_321
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

    nkk.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)

    nkk.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_132)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-5)

    knk.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)

    knk.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_312)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-5)

    kkn.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    kkn.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_321)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 3
    nkk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-4)

    knk.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)

    kkn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # Check binslop = 0
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    nkk.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    nkk.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1, cat2, cat3, ordered=1, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-4)
    knk.process(cat2, cat1, cat3, ordered=2, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2, cat3, cat1, ordered=3, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nkk.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        knk.process(cat2, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        kkn.process(cat2, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1p.patch_centers)

    nkk.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_123)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_213)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_231)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    nkk.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_sum1)
    np.testing.assert_allclose(nkk.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_sum1, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_sum2)
    np.testing.assert_allclose(knk.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_sum2, rtol=1.e-5)
    kkn.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(kkn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        nkk.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        knk.process(cat2p, cat1p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        kkn.process(cat2p, cat3p, cat1p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_122 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_212 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_221 = np.zeros((nbins, nbins, nphi_bins))
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

                www = w1[i] * w2[j] * w2[k]
                zeta = www * k2[j] * k2[k]

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

    nkk.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    nkk.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)

    knk.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)

    kkn.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)
    kkn.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    with assert_raises(ValueError):
        knk.process(cat1, cat2)
    with assert_raises(ValueError):
        knk.process(cat2, cat1)
    with assert_raises(ValueError):
        nkk.process(cat1)
    with assert_raises(ValueError):
        nkk.process(cat2)
    with assert_raises(ValueError):
        knk.process(cat1)
    with assert_raises(ValueError):
        knk.process(cat2)
    with assert_raises(ValueError):
        kkn.process(cat1)
    with assert_raises(ValueError):
        kkn.process(cat2)

    # With ordered=False, doesn't do anything difference, since there is no other valid order.
    nkk.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)

    kkn.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1p.patch_centers)

    nkk.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=2.e-5)
    kkn.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    knk.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knk.ntri, true_ntri_212)
    np.testing.assert_allclose(knk.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_212, rtol=2.e-5)
    kkn.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)

    nkk.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkk.ntri, true_ntri_122)
    np.testing.assert_allclose(nkk.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_122, rtol=1.e-5)
    kkn.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(kkn.ntri, true_ntri_221)
    np.testing.assert_allclose(kkn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_221, rtol=1.e-5)


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
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    k2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.uniform(1,3, (ngal,))
    k3 = rng.normal(0,0.2, (ngal,))
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    max_sep = 30.
    nbins = 5
    max_n = 10

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
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
                zeta = www * k2[j] * k3[k]

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

                # 213, 231
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
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

    nkk.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nkk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-4)
    nkk.process(cat1, cat3, cat2)
    np.testing.assert_allclose(nkk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-4)
    knk.process(cat2, cat1, cat3)
    np.testing.assert_allclose(knk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-4)
    knk.process(cat3, cat1, cat2)
    np.testing.assert_allclose(knk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-4)
    kkn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-4)
    kkn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-4)

    # Repeat with binslop = 0
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nkk.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nkk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-4)
    nkk.process(cat1, cat3, cat2)
    np.testing.assert_allclose(nkk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-4)

    knk.process(cat2, cat1, cat3)
    np.testing.assert_allclose(knk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-4)
    knk.process(cat3, cat1, cat2)
    np.testing.assert_allclose(knk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-4)

    kkn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-4)
    kkn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    nkk.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(nkk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-4)
    nkk.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(nkk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-4)

    knk.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(knk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-4)
    knk.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(knk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-4)

    kkn.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-4)
    kkn.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(kkn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-4)

    # Now use all three patched
    nkk.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(nkk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_123, rtol=1.e-4)
    nkk.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(nkk.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_132, rtol=1.e-4)

    knk.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(knk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_213, rtol=1.e-4)
    knk.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(knk.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_312, rtol=1.e-4)

    kkn.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(kkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_231, rtol=1.e-4)
    kkn.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(kkn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        nkk.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        knk.process(cat2p, cat1p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        kkn.process(cat2p, cat3p, cat1p, patch_method='global')

    # Test I/O
    for name, corr in zip(['nkk', 'knk', 'kkn'], [nkk, knk, kkn]):
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
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)

    min_sep = 10.
    max_sep = 30.
    nbins = 5
    max_n = 8

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
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
                zeta = www * k1[i] * k1[j]

                # 112, 121
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
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
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_211[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_211[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_211[kr1,kr2,:] += 1

    kkn.process(cat1, cat1, cat2)
    np.testing.assert_allclose(kkn.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_112, rtol=1.e-4)
    knk.process(cat1, cat2, cat1)
    np.testing.assert_allclose(knk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_121, rtol=1.e-4)
    # 3 arg version doesn't work for kkn because the nkk processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    kkn.process(cat1, cat2)
    np.testing.assert_allclose(kkn.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_112, rtol=1.e-4)
    nkk.process(cat2, cat1)
    np.testing.assert_allclose(nkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_211, rtol=1.e-4)

    # Repeat with binslop = 0
    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    kkn.process(cat1, cat2)
    np.testing.assert_allclose(kkn.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_112, rtol=1.e-4)

    knk.process(cat1, cat2, cat1)
    np.testing.assert_allclose(knk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_121, rtol=1.e-4)

    nkk.process(cat2, cat1)
    np.testing.assert_allclose(nkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_211, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    kkn.process(cat1p, cat2)
    np.testing.assert_allclose(kkn.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_112, rtol=1.e-4)

    knk.process(cat1p, cat2, cat1)
    np.testing.assert_allclose(knk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_121, rtol=1.e-4)

    nkk.process(cat2p, cat1)
    np.testing.assert_allclose(nkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_211, rtol=1.e-4)

    # Now use both patched
    kkn.process(cat1p, cat2p)
    np.testing.assert_allclose(kkn.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(kkn.zeta, true_zeta_112, rtol=1.e-4)

    knk.process(cat1p, cat2p, cat1p)
    np.testing.assert_allclose(knk.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(knk.zeta, true_zeta_121, rtol=1.e-4)

    nkk.process(cat2p, cat1p)
    np.testing.assert_allclose(nkk.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(nkk.zeta, true_zeta_211, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        kkn.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        knk.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        nkk.process(cat2p, cat1p, patch_method='global')


@timer
def test_nkk_logsas():
    # Use compensated gaussian around "lens" centers.
    #
    # kappa(r) = k0 exp(-r^2/r0^2) (1-r^2/r0^2)

    r0 = 10.
    if __name__ == '__main__':
        nlens = 10
        nsource = 500000
        L = 50. * r0
        tol_factor = 1
    else:
        nlens = 5
        nsource = 50000
        L = 20. * r0
        tol_factor = 3

    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L

    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    dx = x2[:,None]-x1[None,:]
    dy = y2[:,None]-y1[None,:]
    r = np.sqrt(dx**2 + dy**2) / r0

    k0 = 0.7
    kappa = k0 * np.exp(-r**2) * (1-r**2)
    kappa = np.sum(kappa, axis=1)
    print('mean kappa = ',np.mean(kappa))

    ncat = treecorr.Catalog(x=x1, y=y1)
    kcat = treecorr.Catalog(x=x2, y=y2, k=kappa)

    min_sep = 3
    max_sep = 9
    nbins = 5
    min_phi = 0.2
    max_phi = np.pi
    nphi_bins = 15

    nkk = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')
    knk = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')
    kkn = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')

    for name, corr in zip(['nkk', 'knk', 'kkn'], [nkk, knk, kkn]):
        t0 = time.time()
        if name == 'nkk':
            corr.process(ncat, kcat, algo='triangle')
        elif name == 'knk':
            corr.process(kcat, ncat, kcat, algo='triangle')
        else:
            corr.process(kcat, ncat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        # Use r1,r2 = distance from N vertex to two K vertices.
        if name == 'nkk':
            r1 = corr.meand2/r0
            r2 = corr.meand3/r0
        elif name == 'knk':
            r1 = corr.meand1/r0
            r2 = corr.meand3/r0
        else:
            r1 = corr.meand1/r0
            r2 = corr.meand2/r0

        # Expected value is just the product of the two kappa values at the distances of
        # the two K vertices to the N vertex.
        # This works well if we limit to the range where both values are positive. (r<r0)
        true_zeta = k0**2 * np.exp(-(r1**2 + r2**2)) * (1-r1**2) * (1-r2**2)

        m = np.where((r1 < 0.9) & (r2 < 0.9))
        print('max diff = ',np.max(np.abs(corr.zeta[m] - true_zeta[m])))
        print('max rel diff = ',np.max(np.abs((corr.zeta[m] - true_zeta[m])/true_zeta[m])))
        print('mean diff = ',np.mean(corr.zeta[m] - true_zeta[m]))
        print('mean zeta = ',np.mean(true_zeta[m]))
        print('mean ratio = ',np.mean(corr.zeta[m]/true_zeta[m]))
        np.testing.assert_allclose(corr.zeta[m], true_zeta[m], rtol=0.15*tol_factor)
        print('ave error = ',np.abs(np.mean(corr.zeta[m] - true_zeta[m])/np.mean(true_zeta[m])))
        assert np.abs(np.mean(corr.zeta[m] - true_zeta[m])) < 0.05 * np.mean(true_zeta[m] * tol_factor)

        # Repeat this using Multipole and then convert to SAS:
        t0 = time.time()
        if name == 'nkk':
            corrm = treecorr.NKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(ncat, kcat)
        elif name == 'knk':
            corrm = treecorr.KNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(kcat, ncat, kcat)
        else:
            corrm = treecorr.KKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(kcat, ncat)
        corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins)
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('max diff = ',np.max(np.abs(corrs.zeta[m] - true_zeta[m])))
        print('max rel diff = ',np.max(np.abs((corrs.zeta[m] - true_zeta[m])/true_zeta[m])))
        print('mean diff = ',np.mean(corrs.zeta[m] - true_zeta[m]))
        print('mean zeta = ',np.mean(true_zeta[m]))
        print('mean ratio = ',np.mean(corrs.zeta[m]/true_zeta[m]))
        print('zeta mean ratio = ',np.mean(corrs.zeta[m] / corr.zeta[m]))
        print('zeta mean diff = ',np.mean(corrs.zeta[m] - corr.zeta[m]))
        np.testing.assert_allclose(corrs.zeta[m], corr.zeta[m], rtol=0.15*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.zeta[m] / corr.zeta[m]), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.std(corrs.zeta[m] / corr.zeta[m]), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd1, corr.meanlogd1, atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd2, corr.meanlogd2, atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd3, corr.meanlogd3, atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanphi, corr.meanphi, rtol=0.1*tol_factor)

        # Error to try to change sep binning with toSAS
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, min_sep=5)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, max_sep=25)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, nbins=20)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, bin_size=0.01, nbins=None)
        # Error if non-Multipole calls toSAS
        with assert_raises(TypeError):
            corrs.toSAS()

        # All of the above is the default algorithm if process doesn't set algo='triangle'.
        # Check the automatic use of the multipole algorithm from LogSAS.
        corr3 = corr.copy()
        t0 = time.time()
        if name == 'nkk':
            corr3.process(ncat, kcat, max_n=120)
        elif name == 'knk':
            corr3.process(kcat, ncat, kcat, max_n=120)
        else:
            corr3.process(kcat, ncat, max_n=120)
        t1 = time.time()
        print(name,'normal process time = ',t1-t0)

        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.zeta, corrs.zeta)

        # Check that we get the same result using the corr3 functin:
        # (This implicitly uses the multipole algorithm.)
        ncat.write(os.path.join('data',name+'_ndata_logsas.dat'))
        kcat.write(os.path.join('data',name+'_kdata_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['zeta'], corr3.zeta.flatten(), rtol=1.e-3, atol=0)

        if name == 'knk':
            # Invalid to omit file_name2
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)
        else:
            # Invalid to call cat2 file_name3 rather than file_name2
            config['file_name3'] = config['file_name2']
            if name == 'nkk':
                config['k_col'] = [0,0,3]
            else:
                config['k_col'] = [3,0,0]
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)

        # Check the fits write option
        try:
            import fitsio
        except ImportError:
            pass
        else:
            out_file_name = os.path.join('output','corr_nkk_logsas.fits')
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
            np.testing.assert_allclose(data['zeta'], corr.zeta.real.flatten())
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

    # Put in a nominal pattern for k, but this pattern doesn't have much 3pt correlation.
    kappa0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(1234)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 100
    nsource = 10000

    file_name = 'data/test_varzeta_nkk.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nkks = []
        all_knks = []
        all_kkns = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x1 = (rng.random_sample(nlens)-0.5) * L
            y1 = (rng.random_sample(nlens)-0.5) * L
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5
            r2 = (x2**2 + y2**2)/r0**2
            k = kappa0 * np.exp(-r2/2.)
            k += rng.normal(0, 0.2, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1)
            kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
            nkk = treecorr.NKKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            knk = treecorr.KNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            kkn = treecorr.KKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            nkk.process(ncat, kcat)
            knk.process(kcat, ncat, kcat)
            kkn.process(kcat, ncat)
            all_nkks.append(nkk)
            all_knks.append(knk)
            all_kkns.append(kkn)

        mean_nkk_zeta = np.mean([nkk.zeta for nkk in all_nkks], axis=0)
        var_nkk_zeta = np.var([nkk.zeta for nkk in all_nkks], axis=0)
        mean_nkk_varzeta = np.mean([nkk.varzeta for nkk in all_nkks], axis=0)
        mean_knk_zeta = np.mean([knk.zeta for knk in all_knks], axis=0)
        var_knk_zeta = np.var([knk.zeta for knk in all_knks], axis=0)
        mean_knk_varzeta = np.mean([knk.varzeta for knk in all_knks], axis=0)
        mean_kkn_zeta = np.mean([kkn.zeta for kkn in all_kkns], axis=0)
        var_kkn_zeta = np.var([kkn.zeta for kkn in all_kkns], axis=0)
        mean_kkn_varzeta = np.mean([kkn.varzeta for kkn in all_kkns], axis=0)

        np.savez(file_name,
                 mean_nkk_zeta=mean_nkk_zeta,
                 var_nkk_zeta=var_nkk_zeta,
                 mean_nkk_varzeta=mean_nkk_varzeta,
                 mean_knk_zeta=mean_knk_zeta,
                 var_knk_zeta=var_knk_zeta,
                 mean_knk_varzeta=mean_knk_varzeta,
                 mean_kkn_zeta=mean_kkn_zeta,
                 var_kkn_zeta=var_kkn_zeta,
                 mean_kkn_varzeta=mean_kkn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nkk_zeta = data['mean_nkk_zeta']
    var_nkk_zeta = data['var_nkk_zeta']
    mean_nkk_varzeta = data['mean_nkk_varzeta']
    mean_knk_zeta = data['mean_knk_zeta']
    var_knk_zeta = data['var_knk_zeta']
    mean_knk_varzeta = data['mean_knk_varzeta']
    mean_kkn_zeta = data['mean_kkn_zeta']
    var_kkn_zeta = data['var_kkn_zeta']
    mean_kkn_varzeta = data['mean_kkn_varzeta']

    print('var_nkk_zeta = ',var_nkk_zeta)
    print('mean nkk_varzeta = ',mean_nkk_varzeta)
    print('ratio = ',var_nkk_zeta.ravel() / mean_nkk_varzeta.ravel())
    print('var_knk_zeta = ',var_knk_zeta)
    print('mean knk_varzeta = ',mean_knk_varzeta)
    print('ratio = ',var_knk_zeta.ravel() / mean_knk_varzeta.ravel())
    print('var_kkn_zeta = ',var_kkn_zeta)
    print('mean kkn_varzeta = ',mean_kkn_varzeta)
    print('ratio = ',var_kkn_zeta.ravel() / mean_kkn_varzeta.ravel())

    print('max relerr for nkk zeta = ',
          np.max(np.abs((var_nkk_zeta - mean_nkk_varzeta)/var_nkk_zeta)))
    np.testing.assert_allclose(mean_nkk_varzeta, var_nkk_zeta, rtol=0.2)

    print('max relerr for knk zeta = ',
          np.max(np.abs((var_knk_zeta - mean_knk_varzeta)/var_knk_zeta)))
    np.testing.assert_allclose(mean_knk_varzeta, var_knk_zeta, rtol=0.1)

    print('max relerr for kkn zeta = ',
          np.max(np.abs((var_kkn_zeta - mean_kkn_varzeta)/var_kkn_zeta)))
    np.testing.assert_allclose(mean_kkn_varzeta, var_kkn_zeta, rtol=0.1)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    k = kappa0 * np.exp(-r2/2.)
    k += rng.normal(0, 0.2, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1)
    kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
    nkk = treecorr.NKKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    knk = treecorr.KNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    kkn = treecorr.KKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nkk.cov, 0)
    np.testing.assert_array_equal(nkk.varzeta, 0)
    np.testing.assert_array_equal(knk.cov, 0)
    np.testing.assert_array_equal(knk.varzeta, 0)
    np.testing.assert_array_equal(kkn.cov, 0)
    np.testing.assert_array_equal(kkn.varzeta, 0)

    nkk.process(ncat, kcat)
    print('NKK single run:')
    print('max relerr for zeta = ',np.max(np.abs((nkk.varzeta - var_nkk_zeta)/var_nkk_zeta)))
    print('ratio = ',nkk.varzeta / var_nkk_zeta)
    np.testing.assert_allclose(nkk.varzeta, var_nkk_zeta, rtol=0.5)
    np.testing.assert_allclose(nkk.cov.diagonal(), nkk.varzeta.ravel())

    knk.process(kcat, ncat, kcat)
    print('KNK single run:')
    print('max relerr for zeta = ',np.max(np.abs((knk.varzeta - var_knk_zeta)/var_knk_zeta)))
    print('ratio = ',knk.varzeta / var_knk_zeta)
    np.testing.assert_allclose(knk.varzeta, var_knk_zeta, rtol=0.5)
    np.testing.assert_allclose(knk.cov.diagonal(), knk.varzeta.ravel())

    kkn.process(kcat, ncat)
    print('KKN single run:')
    print('max relerr for zeta = ',np.max(np.abs((kkn.varzeta - var_kkn_zeta)/var_kkn_zeta)))
    print('ratio = ',kkn.varzeta / var_kkn_zeta)
    np.testing.assert_allclose(kkn.varzeta, var_kkn_zeta, rtol=0.6)
    np.testing.assert_allclose(kkn.cov.diagonal(), kkn.varzeta.ravel())


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross12()
    test_varzeta_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross12()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross12()
    test_nkk_logsas()
    test_varzeta()
