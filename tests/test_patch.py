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
import os
import coord
import time
import treecorr
try:
    import cPickle as pickle
except ImportError:
    import pickle

from test_helper import assert_raises, do_pickle, timer, get_from_wiki, CaptureLog, clear_save

@timer
def test_cat_patches():
    # Test the different ways to set patches in the catalog.

    # Use the same input as test_radec()
    if __name__ == '__main__':
        ngal = 10000
        npatch = 128
        max_top = 7
    else:
        ngal = 1000
        npatch = 8
        max_top = 3
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) ) + 100  # Put everything at large y, so smallish angle on sky
    z = rng.normal(0,s, (ngal,) )
    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)

    # cat0 is the base catalog without patches
    cat0 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad')
    assert cat0.npatch == 1
    assert len(cat0.patches) == 1
    assert cat0.patches[0].ntot == ngal
    assert cat0.patches[0].npatch == 1

    # 1. Make the patches automatically using kmeans
    #    Note: If npatch is a power of two, then the patch determination is completely
    #          deterministic, which is helpful for this test.
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch)
    p2, cen = cat0.getNField(max_top=max_top).run_kmeans(npatch)
    np.testing.assert_array_equal(cat1.patch, p2)
    assert cat1.npatch == npatch
    assert len(cat1.patches) == npatch
    assert np.sum([p.ntot for p in cat1.patches]) == ngal
    assert all([c.npatch == npatch for c in cat1.patches])

    # 2. Optionally can use alt algorithm
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch,
                            kmeans_alt=True)
    p3, cen = cat0.getNField(max_top=max_top).run_kmeans(npatch, alt=True)
    np.testing.assert_array_equal(cat2.patch, p3)
    assert cat2.npatch == npatch
    assert len(cat2.patches) == npatch
    assert all([c.npatch == npatch for c in cat2.patches])

    # 3. Optionally can set different init method
    cat3 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch,
                            kmeans_init='kmeans++')
    # Can't test this equalling a repeat run from cat0, because kmpp has a random aspect to it.
    # But at least check that it isn't equal to the other two versions.
    assert not np.array_equal(cat3.patch, p2)
    assert not np.array_equal(cat3.patch, p3)
    assert cat3.npatch == npatch
    assert all([c.npatch == npatch for c in cat3.patches])
    cat3b = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch,
                             kmeans_init='random')
    assert not np.array_equal(cat3b.patch, p2)
    assert not np.array_equal(cat3b.patch, p3)
    assert not np.array_equal(cat3b.patch, cat3.patch)

    # 4. Pass in patch array explicitly
    cat4 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', patch=p2)
    np.testing.assert_array_equal(cat4.patch, p2)
    assert cat4.npatch == npatch
    assert all([c.npatch == npatch for c in cat4.patches])

    # 5. Read patch from a column in ASCII file
    file_name5 = os.path.join('output','test_cat_patches.dat')
    cat4.write(file_name5)
    cat5 = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                            patch_col=3)
    assert not cat5.loaded
    np.testing.assert_array_equal(cat5.patch, p2)
    assert cat5.loaded   # Now it's loaded, since we accessed cat5.patch.
    assert cat5.npatch == npatch
    assert all([c.npatch == npatch for c in cat5.patches])

    # Just load a single patch from an ASCII file with many patches.
    for i in range(npatch):
        cat = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                               patch_col=3, patch=i)
        assert cat.patch == cat5.patches[i].patch == i
        np.testing.assert_array_equal(cat.x,cat5.patches[i].x)
        np.testing.assert_array_equal(cat.y,cat5.patches[i].y)
        assert cat == cat5.patches[i]

        cata = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                                patch_col=3, patch=i, last_row=ngal//2)
        catb = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                                patch_col=3, patch=i, first_row=ngal//2+1)
        assert cata.patch == i
        np.testing.assert_array_equal(cata.x,cat5.patches[i].x[:cata.nobj])
        np.testing.assert_array_equal(cata.y,cat5.patches[i].y[:cata.nobj])
        np.testing.assert_array_equal(catb.x,cat5.patches[i].x[cata.nobj:])
        np.testing.assert_array_equal(catb.y,cat5.patches[i].y[cata.nobj:])

        # get_patches from a single patch will return a list with just itself.
        assert cata.get_patches(low_mem=False) == [cata]
        assert catb.get_patches(low_mem=True) == [catb]

    # Patches start in an unloaded state (by default)
    cat5b = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                             patch_col=3)
    assert not cat5b.loaded
    cat5b_patches = cat5b.get_patches(low_mem=True)
    assert cat5b.loaded   # Needed to load to get number of patches.
    cat5b._patches = None  # Need this so get_patches doesn't early exit.
    cat5b_patches2 = cat5b.get_patches(low_mem=True)  # Repeat with loaded cat5b should be equiv.
    cat5b._patches = None
    cat5b_patches3 = cat5b.get_patches(low_mem=False)
    cat5b._patches = None
    cat5b_patches4 = cat5b.get_patches()  # Default is False
    for i in range(4):  # Don't bother with all the patches.  4 suffices to check this.
        assert not cat5b_patches[i].loaded   # But single patch not loaded yet.
        assert not cat5b_patches2[i].loaded
        assert cat5b_patches3[i].loaded      # Unless we didn't ask for low memory.
        assert cat5b_patches4[i].loaded
        assert np.all(cat5b_patches[i].patch == i)  # Triggers load of patch.
        np.testing.assert_array_equal(cat5b_patches[i].x, cat5.x[cat5.patch == i])

    # Just load a single patch from an ASCII file with many patches.
    for i in range(4):
        cat = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                               patch_col=3, patch=i)
        assert cat.patch == cat5.patches[i].patch
        np.testing.assert_array_equal(cat.x,cat5.patches[i].x)
        np.testing.assert_array_equal(cat.y,cat5.patches[i].y)
        assert cat == cat5.patches[i]
        assert cat == cat5b_patches[i]

    # 6. Read patch from a column in FITS file
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name6 = os.path.join('output','test_cat_patches.fits')
        cat4.write(file_name6)
        cat6 = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad', patch_col='patch')
        np.testing.assert_array_equal(cat6.patch, p2)
        cat6b = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad', patch_col='patch', patch_ext=1)
        np.testing.assert_array_equal(cat6b.patch, p2)
        assert len(cat6.patches) == npatch
        assert len(cat6b.patches) == npatch

        # Calling get_patches will not force loading of the file.
        cat6c = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad', patch_col='patch')
        assert not cat6c.loaded
        cat6c_patches = cat6c.get_patches(low_mem=True)
        assert cat6c.loaded
        cat6c._patches = None
        cat6c_patches2 = cat6c.get_patches(low_mem=True)
        cat6c._patches = None
        cat6c_patches3 = cat6c.get_patches(low_mem=False)
        cat6c._patches = None
        cat6c_patches4 = cat6c.get_patches()
        for i in range(4):
            assert not cat6c_patches[i].loaded
            assert not cat6c_patches2[i].loaded
            assert cat6c_patches3[i].loaded
            assert cat6c_patches4[i].loaded
            assert np.all(cat6c_patches[i].patch == i)  # Triggers load of patch.
            np.testing.assert_array_equal(cat6c_patches[i].x, cat6.x[cat6.patch == i])

            cat = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad', patch_col='patch', patch=i)
            assert cat.patch == cat6.patches[i].patch
            np.testing.assert_array_equal(cat.x,cat6.patches[i].x)
            np.testing.assert_array_equal(cat.y,cat6.patches[i].y)
            assert cat == cat6.patches[i]
            assert cat == cat6c_patches[i]

            cata = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec', last_row=ngal//2,
                                    ra_units='rad', dec_units='rad', patch_col='patch', patch=i)
            catb = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec', first_row=ngal//2+1,
                                    ra_units='rad', dec_units='rad', patch_col='patch', patch=i)
            assert cata.patch == i
            np.testing.assert_array_equal(cata.x,cat6.patches[i].x[:cata.nobj])
            np.testing.assert_array_equal(cata.y,cat6.patches[i].y[:cata.nobj])
            np.testing.assert_array_equal(catb.x,cat6.patches[i].x[cata.nobj:])
            np.testing.assert_array_equal(catb.y,cat6.patches[i].y[cata.nobj:])

            # get_patches from a single patch will return a list with just itself.
            assert cata.get_patches(low_mem=False) == [cata]
            assert catb.get_patches(low_mem=True) == [catb]

    # 7. Set a single patch number
    cat7 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', patch=3)
    np.testing.assert_array_equal(cat7.patch, 3)
    cat8 = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                             patch_col=3, patch=3)
    np.testing.assert_array_equal(cat8.patch, 3)

    # low_mem=True works if not from a file, but it's not any different
    cat1_patches = cat1.patches
    cat1._patches = None
    assert cat1.get_patches(low_mem=True) == cat1_patches
    cat2_patches = cat2.patches
    cat2._patches = None
    assert cat2.get_patches(low_mem=True) == cat2_patches
    cat9 = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad')
    cat9_patches = cat9.patches
    cat9._patches = None
    assert cat9.get_patches(low_mem=True) == cat9_patches

    # Check serialization with patch
    do_pickle(cat2)
    do_pickle(cat7)

    # Check some invalid parameters
    # npatch if given must be compatible with patch
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=3, patch=p2)
    # Note: npatch larger than what is in patch is ok.
    #       It indicates that this is part of a larger group with more patches.
    treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=300, patch=p2)
    # patch has to have same number of entries
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', patch=p2[:17])
    # npatch=0 is not allowed
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=0)
    # bad option names
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch,
                         kmeans_init='invalid')
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch,
                         kmeans_alt='maybe')
    with assert_raises(ValueError):
        treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                         patch_col='invalid')
    # bad patch col
    with assert_raises(ValueError):
        treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                         patch_col=4)
    # cannot give vector for patch when others are from file name
    # (Should this be revisited?  Allow this?)
    with assert_raises(TypeError):
        treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                         patch=p2)
    try:
        import fitsio
    except ImportError:
        pass
    else:
        # bad patch ext
        with assert_raises(IOError):
            treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                             ra_units='rad', dec_units='rad', patch_col='patch', patch_ext=2)
        # bad patch col name for fits
        with assert_raises(ValueError):
            treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                             ra_units='rad', dec_units='rad', patch_col='patches')

@timer
def test_cat_centers():
    # Test writing patch centers and setting patches from centers.

    if __name__ == '__main__':
        ngal = 100000
        npatch = 128
    else:
        ngal = 1000
        npatch = 8
    s = 10.

    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) ) + 100  # Put everything at large y, so smallish angle on sky
    z = rng.normal(0,s, (ngal,) )
    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)

    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=npatch)
    centers = [(c.x.mean(), c.y.mean(), c.z.mean()) for c in cat1.patches]
    centers /= np.sqrt(np.sum(np.array(centers)**2,axis=1))[:,np.newaxis]
    centers2 = cat1.patch_centers
    print('center0 = ',centers[0])
    print('          ',centers2[0])
    print('center1 = ',centers[1])
    print('          ',centers2[1])
    print('max center difference = ',np.max(np.abs(centers2-centers)))
    for p in range(npatch):
        np.testing.assert_allclose(centers2[p], centers[p], atol=1.e-4)
    centers3 = cat1.get_patch_centers()
    for p in range(npatch):
        np.testing.assert_allclose(centers3[p], centers2[p])

    # Write the centers to a file
    cen_file = os.path.join('output','test_cat_centers.dat')
    cat1.write_patch_centers(cen_file)

    # Read the centers file
    centers3 = cat1.read_patch_centers(cen_file)
    np.testing.assert_allclose(centers3, centers2)

    # Set patches from a centers dict
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                            patch_centers=centers2)
    np.testing.assert_array_equal(cat2.patch, cat1.patch)
    np.testing.assert_array_equal(cat2.patch_centers, centers2)

    # Set patches from file
    cat3 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                            patch_centers=cen_file)
    np.testing.assert_array_equal(cat3.patch, cat1.patch)
    np.testing.assert_array_equal(cat3.patch_centers, centers2)

    # If doing this from a config dict, patch_centers will be found in the config dict.
    config = dict(ra_units='rad', dec_units='rad', patch_centers=cen_file)
    cat4 = treecorr.Catalog(config=config, ra=ra, dec=dec)
    np.testing.assert_array_equal(cat4.patch, cat1.patch)
    np.testing.assert_array_equal(cat4.patch_centers, centers2)

    # If the original catalog had manual patches set, it needs to calculate the centers
    # after the fact, so things aren't perfect, but should be close.
    cat5 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                            patch=cat1.patch)
    np.testing.assert_array_equal(cat5.patch, cat1.patch)
    np.testing.assert_allclose(cat5.patch_centers, centers2, atol=1.e-4)

    cat6 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                            patch_centers=cat5.patch_centers)
    print('n diff = ',np.sum(cat6.patch != cat5.patch))
    assert np.sum(cat6.patch != cat5.patch) < 10
    np.testing.assert_allclose(cat6.patch_centers, cat5.patch_centers)

    # The patch centers from the patch sub-catalogs should match.
    cen5 = [c.patch_centers[0] for c in cat5.patches]
    np.testing.assert_array_equal(cen5, cat5.patch_centers)

    # With weights, things can be a bit farther off of course.
    w=rng.uniform(1,2,len(ra))
    cat7 = treecorr.Catalog(ra=ra, dec=dec, w=w, ra_units='rad', dec_units='rad',
                            patch=cat1.patch)
    cat8 = treecorr.Catalog(ra=ra, dec=dec, w=w, ra_units='rad', dec_units='rad',
                            patch_centers=cat7.patch_centers)
    print('n diff = ',np.sum(cat8.patch != cat7.patch))
    assert np.sum(cat8.patch != cat7.patch) < 200
    np.testing.assert_allclose(cat8.patch_centers, cat7.patch_centers)

    # But given the same patch centers, the weight doesn't change the assigned patches.
    cat8b = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                             patch_centers=cat7.patch_centers)
    np.testing.assert_array_equal(cat8.patch, cat8b.patch)
    np.testing.assert_array_equal(cat8.patch_centers, cat8b.patch_centers)

    # Check flat
    cat9 = treecorr.Catalog(x=x, y=y, npatch=npatch)
    cen_file2 = os.path.join('output','test_cat_centers.txt')
    cat9.write_patch_centers(cen_file2)
    centers9 = cat9.read_patch_centers(cen_file2)
    np.testing.assert_allclose(centers9, cat9.patch_centers)

    cat10 = treecorr.Catalog(x=x, y=y, patch_centers=cen_file2)
    np.testing.assert_array_equal(cat10.patch, cat9.patch)
    np.testing.assert_array_equal(cat10.patch_centers, cat9.patch_centers)

    cat11 = treecorr.Catalog(x=x, y=y, patch=cat9.patch)
    cat12 = treecorr.Catalog(x=x, y=y, patch_centers=cat11.patch_centers)
    print('n diff = ',np.sum(cat12.patch != cat11.patch))
    assert np.sum(cat12.patch != cat11.patch) < 10

    cat13 = treecorr.Catalog(x=x, y=y, w=w, patch=cat9.patch)
    cat14 = treecorr.Catalog(x=x, y=y, w=w, patch_centers=cat13.patch_centers)
    print('n diff = ',np.sum(cat14.patch != cat13.patch))
    assert np.sum(cat14.patch != cat13.patch) < 200
    np.testing.assert_array_equal(cat14.patch_centers, cat13.patch_centers)

    # The patch centers from the patch sub-catalogs should match.
    cen13 = [c.patch_centers[0] for c in cat13.patches]
    np.testing.assert_array_equal(cen13, cat13.patch_centers)

    # Using the full patch centers, you can also just load a single patch.
    for i in range(npatch):
        cat = treecorr.Catalog(x=x, y=y, w=w, patch_centers=cat13.patch_centers, patch=i)
        assert cat.patch == cat14.patches[i].patch
        np.testing.assert_array_equal(cat.x,cat14.patches[i].x)
        np.testing.assert_array_equal(cat.y,cat14.patches[i].y)
        assert cat == cat14.patches[i]

    # Loading from a file with patch_centers can mean that get_patches won't trigger a load.
    file_name15 = os.path.join('output','test_cat_centers_f15.dat')
    cat14.write(file_name15)
    cat15 = treecorr.Catalog(file_name15, x_col=1, y_col=2, w_col=3,
                             patch_centers=cat14.patch_centers)
    assert not cat15.loaded
    cat15_patches = cat15.get_patches(low_mem=True)
    assert not cat15.loaded  # Unlike above (in test_cat_patches) it's still unloaded.
    for i in range(4):  # Don't bother with all the patches.  4 suffices to check this.
        assert not cat15_patches[i].loaded
        assert np.all(cat15_patches[i].patch == i)  # Triggers load of patch.
        np.testing.assert_array_equal(cat15_patches[i].x, cat15.x[cat15.patch == i])

        cat = treecorr.Catalog(file_name15, x_col=1, y_col=2, w_col=3,
                               patch_centers=cat15.patch_centers, patch=i)
        assert cat.patch == cat15.patches[i].patch
        np.testing.assert_array_equal(cat.x,cat15_patches[i].x)
        np.testing.assert_array_equal(cat.y,cat15_patches[i].y)
        assert cat == cat15_patches[i]
        assert cat == cat15.patches[i]

    # Check fits
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name17 = os.path.join('output','test_cat_centers.fits')
        cat8.write(file_name17)
        cat17 = treecorr.Catalog(file_name17, ra_col='ra', dec_col='dec', w_col='w',
                                 ra_units='rad', dec_units='rad',
                                 patch_centers=cat8.patch_centers)
        assert not cat17.loaded
        cat17_patches = cat17.get_patches(low_mem=True)
        assert not cat17.loaded  # Unlike above (in test_cat_patches) it's still unloaded.
        for i in range(4):  # Don't bother with all the patches.  4 suffices to check this.
            assert not cat17_patches[i].loaded
            assert np.all(cat17_patches[i].patch == i)  # Triggers load of patch.
            np.testing.assert_array_equal(cat17_patches[i].ra, cat17.ra[cat17.patch == i])

            cat = treecorr.Catalog(file_name17, ra_col='ra', dec_col='dec', w_col='w',
                                   ra_units='rad', dec_units='rad',
                                   patch_centers=cat8.patch_centers, patch=i)
            assert cat.patch == cat17.patches[i].patch
            np.testing.assert_array_equal(cat.ra,cat17_patches[i].ra)
            np.testing.assert_array_equal(cat.dec,cat17_patches[i].dec)
            assert cat == cat17_patches[i]
            assert cat == cat17.patches[i]

    # Check for some invalid values
    # npatch if given must be compatible with patch_centers
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cen_file, npatch=3)
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cen_file, npatch=13)
    # Can't have both patch_centers and another patch specification
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cen_file, patch=np.ones_like(ra))
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cen_file, patch_col=3)

    # patch_centers is wrong shape
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cen_file2)
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                         patch_centers=cat9.patch_centers)
    with assert_raises(ValueError):
        treecorr.Catalog(x=x, y=y, patch_centers=cen_file)
    with assert_raises(ValueError):
        treecorr.Catalog(x=x, y=y, patch_centers=cat1.patch_centers)

    # Missing some patch numbers
    with assert_raises(RuntimeError):
        c=treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad',
                           patch=np.random.uniform(10,20,len(ra)))
        c.get_patch_centers()


def generate_shear_field(nside, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    # Generate a random shear field with a well-defined power spectrum.
    # It generates shears on a grid nside x nside, and returns, x, y, g1, g2
    kvals = np.fft.fftfreq(nside) * 2*np.pi
    kx,ky = np.meshgrid(kvals,kvals)
    k = kx + 1j*ky
    ksq = kx**2 + ky**2

    # Use a power spectrum with lots of large scale power.
    # The rms shape ends up around 0.2 and min/max are around +-1.
    # Having a lot more large-scale than small-scale power means that sample variance is
    # very important, so the shot noise estimate of the variance is particularly bad.
    Pk = 1.e4 * ksq / (1. + 300.*ksq)**2

    # Make complex gaussian field in k-space.
    f1 = rng.normal(size=Pk.shape)
    f2 = rng.normal(size=Pk.shape)
    f = (f1 + 1j*f2) * np.sqrt(0.5)

    # Make f Hermitian, to correspond to E-mode-only field.
    # Hermitian means f(-k) = conj(f(k)).
    # Note: this is approximate.  It doesn't get all the k=0 and k=nside/2 correct.
    # But this is good enough for xi- to be not close to zero.
    ikxp = slice(1,(nside+1)//2)   # kx > 0
    ikxn = slice(-1,nside//2,-1)   # kx < 0
    ikyp = slice(1,(nside+1)//2)   # ky > 0
    ikyn = slice(-1,nside//2,-1)   # ky < 0
    f[ikyp,ikxn] = np.conj(f[ikyn,ikxp])
    f[ikyn,ikxn] = np.conj(f[ikyp,ikxp])

    # Multiply by the power spectrum to get a realization of a field with this P(k)
    f *= Pk

    # Inverse fft gives the real-space field.
    kappa = nside * np.fft.ifft2(f)

    # Multiply by exp(2iphi) to get gamma field, rather than kappa.
    ksq[0,0] = 1.  # Avoid division by zero
    exp2iphi = k**2 / ksq
    f *= exp2iphi
    gamma = nside * np.fft.ifft2(f)

    # Generate x,y values for the real-space field
    x,y = np.meshgrid(np.linspace(0.,1000.,nside), np.linspace(0.,1000.,nside))

    x = x.ravel()
    y = y.ravel()
    gamma = gamma.ravel()
    kappa = np.real(kappa.ravel())

    return x, y, np.real(gamma), np.imag(gamma), kappa

@timer
def test_gg_jk():
    # Test the variance estimate for GG correlation with jackknife (and other) error estimate.

    if __name__ == '__main__':
        # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
        nside = 1000
        npatch = 64
        tol_factor = 1
    else:
        # Use ~1/10 of the objects when running unit tests
        nside = 200
        npatch = 16
        tol_factor = 8

    # The full simulation needs to run a lot of times to get a good estimate of the variance,
    # but this takes a long time.  So we store the results in the repo.
    # To redo the simulation, just delete the file data/test_gg_jk_*.npz
    file_name = 'data/test_gg_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ggs = []
        for run in range(nruns):
            x, y, g1, g2, _ = generate_shear_field(nside)
            print(run,': ',np.mean(g1),np.std(g1),np.min(g1),np.max(g1))
            cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            gg = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
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

    rng = np.random.RandomState(1234)
    # First run with the normal variance estimate, which is too small.
    x, y, g1, g2, _ = generate_shear_field(nside, rng)

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    gg1 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    gg1.process(cat)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    # Quick gratuitous coverage test:
    assert '_ok' not in gg1.__dict__
    assert 'lazy_property' in str(treecorr.GGCorrelation._ok)
    gg1._ok
    assert '_ok' in gg1.__dict__

    print('weight = ',gg1.weight)
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
    np.testing.assert_allclose(gg1.varxip, mean_varxip, rtol=0.03 * tol_factor)
    np.testing.assert_allclose(gg1.varxim, mean_varxim, rtol=0.03 * tol_factor)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxip, var_xip)
    np.testing.assert_array_less(mean_varxim, var_xim)
    np.testing.assert_array_less(gg1.varxip, var_xip)
    np.testing.assert_array_less(gg1.varxim, var_xim)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch)
    gg2 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='shot',
                                 rng=rng)
    t0 = time.time()
    gg2.process(cat)
    t1 = time.time()
    print('Time for shot processing = ',t1-t0)
    print('weight = ',gg2.weight)
    print('xip = ',gg2.xip)
    print('xim = ',gg2.xim)
    print('varxip = ',gg2.varxip)
    print('varxim = ',gg2.varxim)
    np.testing.assert_allclose(gg2.weight, gg1.weight, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.xip, gg1.xip, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.xim, gg1.xim, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(gg2.varxip, gg1.varxip, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.varxim, gg1.varxim, rtol=1.e-2*tol_factor)

    # Can get this as a (diagonal) covariance matrix using estimate_cov
    np.testing.assert_allclose(gg2.estimate_cov('shot'),
                               np.diag(np.concatenate([gg2.varxip, gg2.varxim])))

    with assert_raises(ValueError):
        gg2.build_cov_design_matrix('shot')
    with assert_raises(ValueError):
        gg2.build_cov_design_matrix('invalid')

    # Now run with jackknife variance estimate.  Should be much better.
    gg3 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife',
                                 rng=rng)
    t0 = time.time()
    gg3.process(cat)
    t1 = time.time()
    print('Time for jackknife processing = ',t1-t0)
    print('xip = ',gg3.xip)
    print('xim = ',gg3.xim)
    print('varxip = ',gg3.varxip)
    print('ratio = ',gg3.varxip / var_xip)
    print('varxim = ',gg3.varxim)
    print('ratio = ',gg3.varxim / var_xim)
    np.testing.assert_allclose(gg3.weight, gg2.weight)
    np.testing.assert_allclose(gg3.xip, gg2.xip)
    np.testing.assert_allclose(gg3.xim, gg2.xim)
    # Not perfect, but within about 30%.
    np.testing.assert_allclose(np.log(gg3.varxip), np.log(var_xip), atol=0.4*tol_factor)
    np.testing.assert_allclose(np.log(gg3.varxim), np.log(var_xim), atol=0.3*tol_factor)

    # Can get the covariance matrix using estimate_cov, which is also stored as cov attribute
    t0 = time.time()
    np.testing.assert_allclose(gg3.estimate_cov('jackknife'), gg3.cov)
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)

    # Can also get the shot covariance matrix using estimate_cov
    np.testing.assert_allclose(gg3.estimate_cov('shot'),
                               np.diag(np.concatenate([gg2.varxip, gg2.varxim])))

    # And can even get the jackknife covariance from a run that used var_method='shot'
    np.testing.assert_allclose(gg2.estimate_cov('jackknife'), gg3.cov)

    # Test design matrix
    A, w = gg2.build_cov_design_matrix('jackknife')
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, gg3.cov)

    # Check that cross-covariance between xip and xim is significant.
    n = gg3.nbins
    print('cross covariance = ',gg3.cov[:n,n:],np.sum(gg3.cov[n:,n:]**2))
    # Make cross correlation matrix
    c = gg3.cov[:n,n:] / (np.sqrt(gg3.varxip)[:,np.newaxis] * np.sqrt(gg3.varxim)[np.newaxis,:])
    print('cross correlation = ',c)
    assert np.sum(c**2) > 1.e-2    # Should be significantly non-zero
    assert np.all(np.abs(c) < 1.)  # And all are between -1 and -1.

    # If gg2 and gg3 were two different calculations, can use
    # estimate_multi_cov to get combined covariance
    t0 = time.time()
    cov23 = treecorr.estimate_multi_cov([gg2,gg3], 'jackknife')
    t1 = time.time()
    print('Time for jackknife cross-covariance = ',t1-t0)
    np.testing.assert_allclose(cov23[:2*n,:2*n], gg3.cov)
    np.testing.assert_allclose(cov23[2*n:,2*n:], gg3.cov)
    # In this case, they aren't different, so they are perfectly correlated.
    np.testing.assert_allclose(cov23[:2*n,2*n:], gg3.cov)
    np.testing.assert_allclose(cov23[2*n:,:2*n], gg3.cov)

    # Test design matrix
    A, w = treecorr.build_multi_cov_design_matrix([gg2,gg3], 'jackknife')
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov23)

    # Repeat with bootstraps, who used to have a bug related to this
    for method in ['sample', 'marked_bootstrap', 'bootstrap']:
        t0 = time.time()
        cov23 = treecorr.estimate_multi_cov([gg2,gg3], method)
        t1 = time.time()
        print('Time for %s cross-covariance = '%method,t1-t0)
        np.testing.assert_allclose(cov23[:2*n,:2*n], gg3.cov, rtol=1.e-2, atol=1.e-5)
        np.testing.assert_allclose(cov23[2*n:,2*n:], gg3.cov, rtol=1.e-2, atol=1.e-5)
        np.testing.assert_allclose(cov23[:2*n,2*n:], gg3.cov, rtol=1.e-2, atol=1.e-5)
        np.testing.assert_allclose(cov23[2*n:,:2*n], gg3.cov, rtol=1.e-2, atol=1.e-5)

    # Check advertised example to reorder the data vectors with func argument
    func = lambda corrs: np.concatenate([c.xip for c in corrs] + [c.xim for c in corrs])
    cov23_alt = treecorr.estimate_multi_cov([gg2,gg3], 'jackknife', func=func)
    np.testing.assert_allclose(cov23_alt[:n,:n], gg3.cov[:n,:n])
    np.testing.assert_allclose(cov23_alt[n:2*n,n:2*n], gg3.cov[:n,:n])
    np.testing.assert_allclose(cov23_alt[2*n:3*n,2*n:3*n], gg3.cov[n:,n:])
    np.testing.assert_allclose(cov23_alt[3*n:4*n,3*n:4*n], gg3.cov[n:,n:])

    # Test design matrix
    A, w = treecorr.build_multi_cov_design_matrix([gg2,gg3], 'jackknife', func=func)
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov23_alt)

    # Check a func that changes the stat length
    func = lambda corrs: corrs[0].xip + corrs[1].xip
    cov23_alt = treecorr.estimate_multi_cov([gg2,gg3], 'jackknife', func=func)
    np.testing.assert_allclose(cov23_alt, 4*gg3.cov[:n,:n])

    # Check func with estimate_cov
    covxip = gg3.estimate_cov('jackknife', func=lambda gg: gg.xip)
    np.testing.assert_allclose(covxip, gg3.cov[:n,:n])

    # Check sample covariance estimate
    treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='sample')
    t0 = time.time()
    cov_sample = gg3.estimate_cov('sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxip = ',cov_sample.diagonal()[:n])
    print('ratio = ',cov_sample.diagonal()[:n] / var_xip)
    print('varxim = ',cov_sample.diagonal()[n:])
    print('ratio = ',cov_sample.diagonal()[n:] / var_xim)
    # It's not too bad ast small scales, but at larger scales the variance in the number of pairs
    # among the different samples gets bigger (since some are near the edge, and others not).
    # So this is only good to a little worse than a factor of 2.
    np.testing.assert_allclose(cov_sample.diagonal()[:n], var_xip, rtol=0.5*tol_factor)
    np.testing.assert_allclose(cov_sample.diagonal()[n:], var_xim, rtol=0.5*tol_factor)

    # Test design matrix
    A, w = gg2.build_cov_design_matrix('sample')
    w /= np.sum(w)
    A -= np.mean(A, axis=0)
    C = 1./(npatch-1) * (w*A.conj().T).dot(A)
    np.testing.assert_allclose(C, cov_sample)

    # Check marked-point bootstrap covariance estimate
    treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='marked_bootstrap')
    t0 = time.time()
    rng_state = gg3.rng.get_state()
    cov_boot = gg3.estimate_cov('marked_bootstrap')
    t1 = time.time()
    print('Time to calculate marked_bootstrap covariance = ',t1-t0)
    print('varxip = ',cov_boot.diagonal()[:n])
    print('ratio = ',cov_boot.diagonal()[:n] / var_xip)
    print('varxim = ',cov_boot.diagonal()[n:])
    print('ratio = ',cov_boot.diagonal()[n:] / var_xim)
    # Not really much better than sample.
    np.testing.assert_allclose(cov_boot.diagonal()[:n], var_xip, rtol=0.5*tol_factor)
    np.testing.assert_allclose(cov_boot.diagonal()[n:], var_xim, rtol=0.5*tol_factor)

    # Test design matrix
    gg2.rng.set_state(rng_state)
    A, w = gg2.build_cov_design_matrix('marked_bootstrap')
    nboot = A.shape[0]
    A -= np.mean(A, axis=0)
    C = 1./(nboot-1) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov_boot)

    # Check bootstrap covariance estimate.
    treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='bootstrap')
    t0 = time.time()
    rng_state = gg3.rng.get_state()
    cov_boot = gg3.estimate_cov('bootstrap')
    t1 = time.time()
    print('Time to calculate bootstrap covariance = ',t1-t0)
    print('varxip = ',cov_boot.diagonal()[:n])
    print('ratio = ',cov_boot.diagonal()[:n] / var_xip)
    print('varxim = ',cov_boot.diagonal()[n:])
    print('ratio = ',cov_boot.diagonal()[n:] / var_xim)
    np.testing.assert_allclose(cov_boot.diagonal()[:n], var_xip, rtol=0.3*tol_factor)
    np.testing.assert_allclose(cov_boot.diagonal()[n:], var_xim, rtol=0.5*tol_factor)

    # Test design matrix
    gg2.rng.set_state(rng_state)
    A, w = gg2.build_cov_design_matrix('bootstrap')
    nboot = A.shape[0]
    A -= np.mean(A, axis=0)
    C = 1./(nboot-1) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov_boot)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_gg.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(gg3, f)
    with open(file_name, 'rb') as f:
        gg4 = pickle.load(f)
    cov4 = gg4.estimate_cov('jackknife')
    np.testing.assert_allclose(cov4, gg3.cov)
    covxip4 = gg4.estimate_cov('jackknife', func=lambda gg: gg.xip)
    np.testing.assert_allclose(covxip4, covxip)

    # And again using the normal write command.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_gg.fits')
        gg3.write(file_name, write_patch_results=True)
        gg4 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        gg4.read(file_name)
        cov4 = gg4.estimate_cov('jackknife')
        np.testing.assert_allclose(cov4, gg3.cov)
        covxip4 = gg4.estimate_cov('jackknife', func=lambda gg: gg.xip)
        np.testing.assert_allclose(covxip4, covxip)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_gg.dat')
    gg3.write(file_name, write_patch_results=True)
    gg5 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    gg5.read(file_name)
    cov5 = gg5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, gg3.cov)
    covxip5 = gg5.estimate_cov('jackknife', func=lambda gg: gg.xip)
    np.testing.assert_allclose(covxip5, covxip)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_gg.hdf5')
        gg3.write(file_name, write_patch_results=True)
        gg6 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        gg6.read(file_name)
        cov6 = gg6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, gg3.cov)
        covxip6 = gg6.estimate_cov('jackknife', func=lambda gg: gg.xip)
        np.testing.assert_allclose(covxip6, covxip)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        gg2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        gg1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        gg1.estimate_cov('sample')
    with assert_raises(ValueError):
        gg1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        gg1.estimate_cov('bootstrap')
    # All of them need to use patches
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],'jackknife')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],'jackknife')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],'sample')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],'sample')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],'marked_bootstrap')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],'marked_bootstrap')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],'bootstrap')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],'bootstrap')
    # All need to use the same patches
    cat3 = treecorr.Catalog(x=x[:100], y=y[:100], g1=g1[:100], g2=g2[:100], npatch=7)
    gg3 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    gg3.process(cat3)
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg3, gg2],'jackknife')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg2, gg3],'jackknife')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg3, gg2],'sample')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg2, gg3],'sample')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg3, gg2],'marked_bootstrap')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg2, gg3],'marked_bootstrap')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg3, gg2],'bootstrap')
    with assert_raises(RuntimeError):
        treecorr.estimate_multi_cov([gg2, gg3],'bootstrap')

@timer
def test_ng_jk():
    # Test the variance estimate for NG correlation with jackknife error estimate.

    if __name__ == '__main__':
        # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
        nside = 1000
        nlens = 50000
        npatch = 64
        tol_factor = 1
    else:
        # If much smaller, then there can be no lenses in some patches, so only 1/4 the galaxies
        # and use more than half the number of patches
        nside = 200
        nlens = 3000
        npatch = 8
        tol_factor = 4

    file_name = 'data/test_ng_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ngs = []
        for run in range(nruns):
            x, y, g1, g2, k = generate_shear_field(nside)
            thresh = np.partition(k.flatten(), -nlens)[-nlens]
            w = np.zeros_like(k)
            w[k>=thresh] = 1.
            print(run,': ',np.mean(g1),np.std(g1),np.min(g1),np.max(g1),thresh)
            cat1 = treecorr.Catalog(x=x, y=y, w=w)
            cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            ng = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
            ng.process(cat1, cat2)
            all_ngs.append(ng)

        mean_xi = np.mean([ng.xi for ng in all_ngs], axis=0)
        var_xi = np.var([ng.xi for ng in all_ngs], axis=0)
        mean_varxi = np.mean([ng.varxi for ng in all_ngs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    var_xi = data['var_xi']
    mean_varxi = data['mean_varxi']

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)

    rng = np.random.RandomState(1234)
    # First run with the normal variance estimate, which is too small.
    x, y, g1, g2, k = generate_shear_field(nside, rng)
    thresh = np.partition(k.flatten(), -nlens)[-nlens]
    w = np.zeros_like(k)
    w[k>=thresh] = 1.
    cat1 = treecorr.Catalog(x=x, y=y, w=w)
    cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    ng1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    ng1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',ng1.weight)
    print('xi = ',ng1.xi)
    print('varxi = ',ng1.varxi)
    print('pullsq for xi = ',(ng1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((ng1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((ng1.xi - mean_xi)**2/var_xi, 25) # within 5 sigma
    np.testing.assert_allclose(ng1.varxi, mean_varxi, rtol=0.03 * tol_factor)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxi, var_xi)
    np.testing.assert_array_less(ng1.varxi, var_xi)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    # Note: This turns out to work significantly better if cat1 is used to make the patches.
    # Otherwise the number of lenses per patch varies a lot, which affects the variance estimate.
    # But that means we need to keep the w=0 object in the catalog, so all objects get a patch.
    cat1p = treecorr.Catalog(x=x, y=y, w=w, npatch=npatch, keep_zero_weight=True)
    cat2p = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, patch=cat1p.patch)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(cat2p.w[cat2p.patch==i])))
    ng2 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='shot')
    t0 = time.time()
    ng2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for shot processing = ',t1-t0)
    print('weight = ',ng2.weight)
    print('xi = ',ng2.xi)
    print('varxi = ',ng2.varxi)
    np.testing.assert_allclose(ng2.weight, ng1.weight, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng2.varxi, ng1.varxi, rtol=1.e-2*tol_factor)

    # Can get this as a (diagonal) covariance matrix using estimate_cov
    np.testing.assert_allclose(ng2.estimate_cov('shot'), np.diag(ng2.varxi))
    np.testing.assert_allclose(ng1.estimate_cov('shot'), np.diag(ng1.varxi))

    # Now run with jackknife variance estimate.  Should be much better.
    ng3 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife',
                                 rng=rng)
    t0 = time.time()
    ng3.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for jackknife processing = ',t1-t0)
    print('xi = ',ng3.xi)
    print('varxi = ',ng3.varxi)
    print('ratio = ',ng3.varxi / var_xi)
    np.testing.assert_allclose(ng3.weight, ng2.weight)
    np.testing.assert_allclose(ng3.xi, ng2.xi)
    np.testing.assert_allclose(np.log(ng3.varxi), np.log(var_xi), atol=0.4*tol_factor)

    # Check using estimate_cov
    t0 = time.time()
    np.testing.assert_allclose(ng2.estimate_cov('jackknife'), ng3.cov)
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)

    # Check only using patches for one of the two catalogs.
    # Not as good as using patches for both, but not much worse.
    ng4 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife',
                                 rng=rng)
    t0 = time.time()
    ng4.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('weight = ',ng4.weight)
    print('xi = ',ng4.xi)
    print('varxi = ',ng4.varxi)
    np.testing.assert_allclose(ng4.weight, ng1.weight, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng4.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(np.log(ng4.varxi), np.log(var_xi), atol=0.6*tol_factor)

    ng5 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife',
                                 rng=rng)
    t0 = time.time()
    ng5.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('weight = ',ng5.weight)
    print('xi = ',ng5.xi)
    print('varxi = ',ng5.varxi)
    np.testing.assert_allclose(ng5.weight, ng1.weight, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng5.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(np.log(ng5.varxi), np.log(var_xi), atol=0.5*tol_factor)

    # Check sample covariance estimate
    t0 = time.time()
    cov_sample = ng3.estimate_cov('sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxi = ',cov_sample.diagonal())
    print('ratio = ',cov_sample.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_sample.diagonal()), np.log(var_xi), atol=0.7*tol_factor)

    cov_sample = ng4.estimate_cov('sample')
    print('varxi = ',cov_sample.diagonal())
    print('ratio = ',cov_sample.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_sample.diagonal()), np.log(var_xi), atol=0.7*tol_factor)

    cov_sample = ng5.estimate_cov('sample')
    print('varxi = ',cov_sample.diagonal())
    print('ratio = ',cov_sample.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_sample.diagonal()), np.log(var_xi), atol=0.7*tol_factor)

    # Check marked_bootstrap covariance estimate
    t0 = time.time()
    cov_boot = ng3.estimate_cov('marked_bootstrap')
    t1 = time.time()
    print('Time to calculate marked_bootstrap covariance = ',t1-t0)
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.6*tol_factor)
    cov_boot = ng4.estimate_cov('marked_bootstrap')
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.8*tol_factor)
    cov_boot = ng5.estimate_cov('marked_bootstrap')
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.6*tol_factor)

    # Check bootstrap covariance estimate.
    t0 = time.time()
    cov_boot = ng3.estimate_cov('bootstrap')
    t1 = time.time()
    print('Time to calculate bootstrap covariance = ',t1-t0)
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.2*tol_factor)
    cov_boot = ng4.estimate_cov('bootstrap')
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.6*tol_factor)
    cov_boot = ng5.estimate_cov('bootstrap')
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(np.log(cov_boot.diagonal()), np.log(var_xi), atol=0.5*tol_factor)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_ng.fits')
        ng3.write(file_name, write_patch_results=True)
        ng4 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        ng4.read(file_name)
        print('ng4.xi = ',ng4.xi)
        print('results = ',ng4.results)
        print('results = ',ng4.results.keys())
        cov4 = ng4.estimate_cov('jackknife')
        print('cov4 = ',cov4)
        np.testing.assert_allclose(cov4, ng3.cov)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_ng.dat')
    ng3.write(file_name, write_patch_results=True)
    ng5 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    ng5.read(file_name)
    cov5 = ng5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, ng3.cov)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_ng.hdf5')
        ng3.write(file_name, write_patch_results=True)
        ng6 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        ng6.read(file_name)
        cov6 = ng6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, ng3.cov)

    # Use a random catalog
    # In this case the locations of the source catalog are fine to use as our random catalog,
    # since they fill the region where the lenses are allowed to be.
    rg4 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    rg4.process(cat2p, cat2p)
    t1 = time.time()
    print('Time for processing RG = ',t1-t0)

    ng4 = ng3.copy()
    ng4.calculateXi(rg=rg4)
    print('xi = ',ng4.xi)
    print('varxi = ',ng4.varxi)
    print('ratio = ',ng4.varxi / var_xi)
    np.testing.assert_allclose(ng4.weight, ng3.weight, rtol=0.02*tol_factor)
    np.testing.assert_allclose(ng4.xi, ng3.xi, rtol=0.02*tol_factor)
    np.testing.assert_allclose(np.log(ng4.varxi), np.log(var_xi), atol=0.3*tol_factor)

    # Check using estimate_cov
    t0 = time.time()
    cov = ng4.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    # The covariance has more terms that differ. 3x5 is the largest difference, needing rtol=0.4.
    # I think this is correct -- mostly this is testing that I didn't totally mess up the
    # weight normalization when applying the RG to the patches.
    np.testing.assert_allclose(cov, ng3.cov, rtol=0.4*tol_factor, atol=3.e-6*tol_factor)

    # Use a random catalog without patches.
    rg5 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    rg5.process(cat2, cat2p)
    t1 = time.time()
    print('Time for processing RG = ',t1-t0)

    ng5 = ng3.copy()
    ng5.calculateXi(rg=rg5)
    print('xi = ',ng5.xi)
    print('varxi = ',ng5.varxi)
    print('ratio = ',ng5.varxi / var_xi)
    np.testing.assert_allclose(ng5.weight, ng3.weight, rtol=0.02*tol_factor)
    np.testing.assert_allclose(ng5.xi, ng3.xi, rtol=0.02*tol_factor)
    # This does only slightly worse.
    np.testing.assert_allclose(np.log(ng5.varxi), np.log(var_xi), atol=0.4*tol_factor)

    # Check using estimate_cov
    t0 = time.time()
    cov = ng5.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    np.testing.assert_allclose(cov, ng3.cov, rtol=0.4*tol_factor, atol=3.e-6*tol_factor)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        ng2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        ng1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        ng1.estimate_cov('sample')
    with assert_raises(ValueError):
        ng1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        ng1.estimate_cov('bootstrap')
    # rg also needs patches (at least for the g part).
    with assert_raises(RuntimeError):
        ng3.calculateXi(rg=ng1)

    cat1a = treecorr.Catalog(x=x[:100], y=y[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x[:100], y=y[:100], g1=g1[:100], g2=g2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x[:100], y=y[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x[:100], y=y[:100], g1=g1[:100], g2=g2[:100], npatch=2)
    ng6 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    ng7 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        ng6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        ng7.process(cat1b,cat2a)

@timer
def test_nn_jk():
    # Test the variance estimate for NN correlation with jackknife error estimate.

    if __name__ == '__main__':
        nside = 1000
        nlens = 50000
        npatch = 32
        rand_factor = 20
        tol_factor = 1
    else:
        nside = 500
        nlens = 1000
        npatch = 8
        rand_factor = 20
        tol_factor = 4

    # Make random catalog with 10x number of sources, randomly distributed.
    rng = np.random.RandomState(1234)
    rx = rng.uniform(0,1000, rand_factor*nlens)
    ry = rng.uniform(0,1000, rand_factor*nlens)
    rand_cat = treecorr.Catalog(x=rx, y=ry)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    rr.process(rand_cat)
    t1 = time.time()
    print('Time to process rand cat = ',t1-t0)

    file_name = 'data/test_nn_jk_{}.npz'.format(nside)
    print(file_name)
    dx = 1000/nside
    if not os.path.isfile(file_name):
        nruns = 1000
        all_xia = []
        all_xib = []
        rng1 = np.random.RandomState()
        for run in range(nruns):
            x, y, _, _, k = generate_shear_field(nside, rng1)
            x += rng1.uniform(-dx/2,dx/2,len(x))
            y += rng1.uniform(-dx/2,dx/2,len(x))
            thresh = np.partition(k.flatten(), -nlens)[-nlens]
            w = np.zeros_like(k)
            w[k>=thresh] = 1.
            print(run,': ',np.mean(k),np.std(k),np.min(k),np.max(k),thresh)
            cat = treecorr.Catalog(x=x, y=y, w=w)
            nn = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
            nr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
            nn.process(cat)
            nr.process(cat, rand_cat)
            xia, varxi = nn.calculateXi(rr=rr)
            xib, varxi = nn.calculateXi(rr=rr, dr=nr)
            all_xia.append(xia)
            all_xib.append(xib)

        mean_xia = np.mean(all_xia, axis=0)
        mean_xib = np.mean(all_xib, axis=0)
        var_xia = np.var(all_xia, axis=0)
        var_xib = np.var(all_xib, axis=0)

        np.savez(file_name,
                 mean_xia=mean_xia, var_xia=var_xia,
                 mean_xib=mean_xib, var_xib=var_xib)

    data = np.load(file_name)
    mean_xia = data['mean_xia']
    var_xia = data['var_xia']
    mean_xib = data['mean_xib']
    var_xib = data['var_xib']

    print('mean_xia = ',mean_xia)
    print('var_xia = ',var_xia)
    print('mean_xib = ',mean_xib)
    print('var_xib = ',var_xib)

    # First run with the normal variance estimate, which is too small.
    x, y, _, _, k = generate_shear_field(nside, rng)
    x += rng.uniform(-dx/2,dx/2,len(x))
    y += rng.uniform(-dx/2,dx/2,len(x))
    thresh = np.partition(k.flatten(), -nlens)[-nlens]
    w = np.zeros_like(k)
    w[k>=thresh] = 1.
    cat = treecorr.Catalog(x=x, y=y, w=w)
    nn1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    nn1.process(cat)
    t1 = time.time()
    nr1.process(cat, rand_cat)
    t2 = time.time()
    xia1, varxia1 = nn1.calculateXi(rr=rr)
    t3 = time.time()
    xib1, varxib1 = nn1.calculateXi(rr=rr, dr=nr1)
    t4 = time.time()
    print('Time for non-patch processing = ',t1-t0, t2-t1, t3-t2, t4-t3)

    print('nn1.weight = ',nn1.weight)
    print('nr1.weight = ',nr1.weight)
    print('xia1 = ',xia1)
    print('varxia1 = ',varxia1)
    print('pullsq for xia = ',(xia1-mean_xia)**2/var_xia)
    print('max pull for xia = ',np.sqrt(np.max((xia1-mean_xia)**2/var_xia)))
    np.testing.assert_array_less((xia1 - mean_xia)**2/var_xia, 25) # within 5 sigma
    print('xib1 = ',xib1)
    print('varxib1 = ',varxib1)
    print('pullsq for xi = ',(xib1-mean_xib)**2/var_xib)
    print('max pull for xi = ',np.sqrt(np.max((xib1-mean_xib)**2/var_xib)))
    np.testing.assert_array_less((xib1 - mean_xib)**2/var_xib, 25) # within 5 sigma

    # The naive error estimates only includes shot noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(varxia1, var_xia)
    np.testing.assert_array_less(varxib1, var_xib)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    # The jackknife estimate (later) works better if the patches are based on the full catalog
    # rather than the weighted catalog, since it covers the area more smoothly.
    full_catp = treecorr.Catalog(x=x, y=y, npatch=npatch)
    catp = treecorr.Catalog(x=x, y=y, w=w, patch_centers=full_catp.patch_centers)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(catp.w[catp.patch==i])))
    nn2 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., var_method='shot')
    nr2 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., var_method='shot')
    t0 = time.time()
    nn2.process(catp)
    t1 = time.time()
    nr2.process(catp, rand_cat)
    t2 = time.time()
    xia2, varxia2 = nn2.calculateXi(rr=rr)
    t3 = time.time()
    xib2, varxib2 = nn2.calculateXi(rr=rr, dr=nr2)
    t4 = time.time()
    print('Time for shot processing = ',t1-t0, t2-t1, t3-t2, t4-t3)
    print('nn2.weight = ',nn2.weight)
    print('ratio = ',nn2.weight / nn1.weight)
    print('nr2.weight = ',nr2.weight)
    print('ratio = ',nr2.weight / nr1.weight)
    print('xia = ',xia2)
    print('varxia = ',varxia2)
    print('xib = ',xib2)
    print('varxib = ',varxib2)
    np.testing.assert_allclose(nn2.weight, nn1.weight, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(xia2, xia1, rtol=2.e-2*tol_factor)
    np.testing.assert_allclose(varxia2, varxia1, rtol=2.e-2*tol_factor)
    np.testing.assert_allclose(xib2, xib1, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(varxib2, varxib1, rtol=2.e-2*tol_factor)

    # Can get this as a (diagonal) covariance matrix using estimate_cov
    np.testing.assert_allclose(nn2.estimate_cov('shot'), np.diag(varxib2))
    np.testing.assert_allclose(nn1.estimate_cov('shot'), np.diag(varxib1))

    # Now run with jackknife variance estimate.  Should be much better.
    nn3 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., var_method='jackknife')
    nr3 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    nn3.process(catp)
    t1 = time.time()
    nr3.process(catp, rand_cat)
    t2 = time.time()
    xia3, varxia3 = nn3.calculateXi(rr=rr)
    t3 = time.time()
    xib3, varxib3 = nn3.calculateXi(rr=rr, dr=nr3)
    t4 = time.time()
    print('Time for jackknife processing = ',t1-t0, t2-t1, t3-t2, t4-t3)
    print('xia = ',xia3)
    print('varxia = ',varxia3)
    print('ratio = ',varxia3 / var_xia)
    np.testing.assert_allclose(nn3.weight, nn2.weight)
    np.testing.assert_allclose(xia3, xia2)
    np.testing.assert_allclose(np.log(varxia3), np.log(var_xia), atol=0.4*tol_factor)
    print('xib = ',xib3)
    print('varxib = ',varxib3)
    print('ratio = ',varxib3 / var_xib)
    np.testing.assert_allclose(xib3, xib2)
    np.testing.assert_allclose(np.log(varxib3), np.log(var_xib), atol=0.4*tol_factor)

    # Check using estimate_cov
    t0 = time.time()
    cov3 = nn2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',cov3.diagonal())
    np.testing.assert_allclose(cov3, nn3.cov)

    # Check sample covariance estimate
    t0 = time.time()
    cov3b = nn3.estimate_cov('sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxi = ',cov3b.diagonal())
    print('ratio = ',cov3b.diagonal() / var_xib)
    np.testing.assert_allclose(cov3b.diagonal(), var_xib, rtol=0.4*tol_factor)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_dd.fits')
        rr_file_name = os.path.join('output','test_write_results_rr.fits')
        dr_file_name = os.path.join('output','test_write_results_dr.fits')
        nn3.write(file_name, write_patch_results=True)
        rr.write(rr_file_name, write_patch_results=True)
        nr3.write(dr_file_name, write_patch_results=True)
        nn4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        rr4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        nr4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        nn4.read(file_name)
        rr4.read(rr_file_name)
        nr4.read(dr_file_name)
        nn4.calculateXi(rr=rr4, dr=nr4)
        cov4 = nn4.estimate_cov('jackknife')
        np.testing.assert_allclose(cov4, nn3.cov)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_dd.dat')
    rr_file_name = os.path.join('output','test_write_results_rr.dat')
    dr_file_name = os.path.join('output','test_write_results_dr.dat')
    nn3.write(file_name, write_patch_results=True, precision=15)
    rr.write(rr_file_name, write_patch_results=True, precision=15)
    nr3.write(dr_file_name, write_patch_results=True, precision=15)
    nn5 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    rr5 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    dr5 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nn5.read(file_name)
    rr5.read(rr_file_name)
    dr5.read(dr_file_name)
    nn5.calculateXi(rr=rr5, dr=dr5)
    cov5 = nn5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, nn3.cov)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_dd.hdf5')
        rr_file_name = os.path.join('output','test_write_results_rr.hdf5')
        dr_file_name = os.path.join('output','test_write_results_dr.hdf5')
        nn3.write(file_name, write_patch_results=True)
        rr.write(rr_file_name, write_patch_results=True)
        nr3.write(dr_file_name, write_patch_results=True)
        nn6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        rr6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        dr6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        nn6.read(file_name)
        rr6.read(rr_file_name)
        dr6.read(dr_file_name)
        nn6.calculateXi(rr=rr6, dr=dr6)
        cov6 = nn6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, nn3.cov)

    # Check NN cross-correlation and other combinations of dr, rd.
    rn3 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    nn3.process(catp, catp)
    t1 = time.time()
    print('Time for cross processing = ',t1-t0)
    np.testing.assert_allclose(nn3.weight, 2*nn2.weight)
    rn3.process(rand_cat, catp)
    xic3, varxic3 = nn3.calculateXi(rr=rr, rd=rn3)
    print('xic = ',xic3)
    print('varxic = ',varxic3)
    print('ratio = ',varxic3 / var_xib)
    print('ratio = ',varxic3 / varxib3)
    np.testing.assert_allclose(xic3, xib3)
    np.testing.assert_allclose(varxic3, varxib3)
    xid3, varxid3 = nn3.calculateXi(rr=rr, dr=nr3, rd=rn3)
    print('xid = ',xid3)
    print('varxid = ',varxid3)
    print('ratio = ',varxid3 / var_xib)
    print('ratio = ',varxid3 / varxib3)
    np.testing.assert_allclose(xid3, xib2)
    np.testing.assert_allclose(varxid3, varxib3)

    # Check serialization with the zero_copies (this didn't work in 4.2.0)
    do_pickle(nn3)

    # Compare to using a random catalog with patches
    rand_catp = treecorr.Catalog(x=rx, y=ry, patch_centers=full_catp.patch_centers)
    rr4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    rn4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nr4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nn4 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., var_method='jackknife')
    t0 = time.time()
    nn4.process(catp)
    rr4.process(rand_catp)
    rn4.process(rand_catp, catp)
    nr4.process(catp, rand_catp)
    t1 = time.time()
    print('Time for cross processing with patches = ',t1-t0)
    np.testing.assert_allclose(nn4.weight, nn2.weight)
    # Save the initial results dict so we test feature of adding additional result keys in dr or rd.
    res = nn4.results.copy()
    xia4, varxia4 = nn4.calculateXi(rr=rr4)
    nn4.results = res.copy()
    xib4, varxib4 = nn4.calculateXi(rr=rr4, dr=nr4)
    nn4.results = res.copy()
    xic4, varxic4 = nn4.calculateXi(rr=rr4, rd=rn4)
    nn4.results = res.copy()
    xid4, varxid4 = nn4.calculateXi(rr=rr4, dr=nr4, rd=rn4)
    print('xia = ',xia4)
    print('xib = ',xib4)
    print('xic = ',xic4)
    print('xid = ',xic4)
    np.testing.assert_allclose(xia4, xia3, rtol=0.03)
    np.testing.assert_allclose(xib4, xib3, rtol=0.03)
    np.testing.assert_allclose(xic4, xic3, rtol=0.03)
    np.testing.assert_allclose(xid4, xid3, rtol=0.03)
    print('varxia = ',varxia4)
    print('ratio = ',varxic4 / var_xia)
    print('varxib = ',varxib4)
    print('ratio = ',varxib4 / var_xib)
    # Using patches for the randoms is not as good.  Only good to rtol=0.6, rather than 0.4 above.
    np.testing.assert_allclose(np.log(varxia4), np.log(var_xia), atol=0.6*tol_factor)
    np.testing.assert_allclose(np.log(varxib4), np.log(var_xib), atol=0.6*tol_factor)
    np.testing.assert_allclose(varxic4, varxib4)
    np.testing.assert_allclose(varxid4, varxib4)

    # Check some invalid parameters
    # randoms need patches, at least for d part.
    with assert_raises(RuntimeError):
        nn3.calculateXi(rr=rr, dr=nr1)
    with assert_raises(RuntimeError):
        nn3.calculateXi(rr=rr, dr=rn3)
    with assert_raises(RuntimeError):
        nn3.calculateXi(rr=rr, rd=nr3)
    with assert_raises(RuntimeError):
        nn3.calculateXi(rr=rr, dr=nr3, rd=nr3)
    with assert_raises(RuntimeError):
        nn3.calculateXi(rr=rr, dr=rn3, rd=rn3)
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        nn1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        nn1.estimate_cov('sample')
    # Need to run calculateXi to get patch-based covariance
    nn5 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nn5.process(catp)
    with assert_raises(RuntimeError):
        nn5.estimate_cov('jackknife')

    # Randoms need to use the same number of patches as data
    catp7 = treecorr.Catalog(x=x[:100], y=y[:100], npatch=7)
    rand_catp7 = treecorr.Catalog(x=rx[:100], y=ry[:100], npatch=7)
    nn6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    rr6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    rn6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nr6 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nn6.process(catp7)
    rr6.process(rand_catp7)
    rn6.process(rand_catp7, catp7)
    nr6.process(catp7, rand_catp7)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr4)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr6, dr=nr4)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr6, rd=rn4)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr6, dr=nr4, rd=rn6)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr6, dr=nr6, rd=rn4)
    with assert_raises(RuntimeError):
        nn6.calculateXi(rr=rr4, dr=nr6, rd=rn6)

@timer
def test_kappa_jk():
    # Test NK, KK, and KG with jackknife.
    # There's not really anything new to test here.  So just checking the interface works.

    if __name__ == '__main__':
        nside = 1000
        nlens = 50000
        npatch = 64
        tol_factor = 1
    else:
        nside = 200
        nlens = 2000
        npatch = 8
        tol_factor = 4

    file_name = 'data/test_kappa_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_nks = []
        all_kks = []
        all_kgs = []
        for run in range(nruns):
            x, y, g1, g2, k = generate_shear_field(nside)
            thresh = np.partition(k.flatten(), -nlens)[-nlens]
            w = np.zeros_like(k)
            w[k>=thresh] = 1.
            print(run,': ',np.mean(k),np.std(k),np.min(k),np.max(k),thresh)
            cat1 = treecorr.Catalog(x=x, y=y, k=k, w=w)
            cat2 = treecorr.Catalog(x=x, y=y, k=k, g1=g1, g2=g2)
            nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
            kk = treecorr.KKCorrelation(bin_size=0.3, min_sep=6., max_sep=30.)
            kg = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
            nk.process(cat1, cat2)
            kk.process(cat2)
            kg.process(cat2, cat2)
            all_nks.append(nk)
            all_kks.append(kk)
            all_kgs.append(kg)

        mean_nk_xi = np.mean([nk.xi for nk in all_nks], axis=0)
        var_nk_xi = np.var([nk.xi for nk in all_nks], axis=0)
        mean_kk_xi = np.mean([kk.xi for kk in all_kks], axis=0)
        var_kk_xi = np.var([kk.xi for kk in all_kks], axis=0)
        mean_kg_xi = np.mean([kg.xi for kg in all_kgs], axis=0)
        var_kg_xi = np.var([kg.xi for kg in all_kgs], axis=0)

        np.savez(file_name,
                 mean_nk_xi=mean_nk_xi, var_nk_xi=var_nk_xi,
                 mean_kk_xi=mean_kk_xi, var_kk_xi=var_kk_xi,
                 mean_kg_xi=mean_kg_xi, var_kg_xi=var_kg_xi)

    data = np.load(file_name)
    mean_nk_xi = data['mean_nk_xi']
    var_nk_xi = data['var_nk_xi']
    mean_kk_xi = data['mean_kk_xi']
    var_kk_xi = data['var_kk_xi']
    mean_kg_xi = data['mean_kg_xi']
    var_kg_xi = data['var_kg_xi']

    print('mean_nk_xi = ',mean_nk_xi)
    print('var_nk_xi = ',var_nk_xi)
    print('mean_kk_xi = ',mean_kk_xi)
    print('var_kk_xi = ',var_kk_xi)
    print('mean_kg_xi = ',mean_kg_xi)
    print('var_kg_xi = ',var_kg_xi)

    rng = np.random.RandomState(1234)
    x, y, g1, g2, k = generate_shear_field(nside, rng)
    thresh = np.partition(k.flatten(), -nlens)[-nlens]
    w = np.zeros_like(k)
    w[k>=thresh] = 1.
    cat1 = treecorr.Catalog(x=x, y=y, k=k, w=w)
    cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k)
    cat1p = treecorr.Catalog(x=x, y=y, k=k, w=w, keep_zero_weight=True, npatch=npatch)
    cat2p = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, patch=cat1p.patch)

    # NK
    # This one is a bit touchy.  It only works well for a small range of scales.
    # At smaller scales, there just aren't enough sources "behind" the lenses.
    # And at larger scales, the power drops off too quickly (more quickly than shear),
    # since convergence is a more local effect.  So for this choice of ngal, nlens,
    # and power spectrum, this is where the covariance estimate works out reasonably well.
    nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10, max_sep=30., var_method='jackknife')
    t0 = time.time()
    nk.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for NK jackknife processing = ',t1-t0)
    print('xi = ',nk.xi)
    print('varxi = ',nk.varxi)
    print('ratio = ',nk.varxi / var_nk_xi)
    np.testing.assert_array_less((nk.xi - mean_nk_xi)**2/var_nk_xi, 25) # within 5 sigma
    np.testing.assert_allclose(np.log(nk.varxi), np.log(var_nk_xi), atol=0.7*tol_factor)

    # Check sample covariance estimate
    cov_xi = nk.estimate_cov('sample')
    print('NK sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_nk_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_nk_xi, rtol=0.6*tol_factor)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_nk.fits')
        nk.write(file_name, write_patch_results=True)
        nk4 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        nk4.read(file_name)
        cov4 = nk4.estimate_cov('jackknife')
        np.testing.assert_allclose(cov4, nk.cov)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_nk.dat')
    nk.write(file_name, write_patch_results=True)
    nk5 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    nk5.read(file_name)
    cov5 = nk5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, nk.cov)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_nk.hdf5')
        nk.write(file_name, write_patch_results=True)
        nk6 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
        nk6.read(file_name)
        cov6 = nk6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, nk.cov)

    # Use a random catalog
    # In this case the locations of the source catalog are fine to use as our random catalog,
    # since they fill the region where the lenses are allowed to be.
    rk2 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    rk2.process(cat2p, cat2p)
    t1 = time.time()
    print('Time for processing RK = ',t1-t0)

    nk2 = nk.copy()
    nk2.calculateXi(rk=rk2)
    print('xi = ',nk2.xi)
    print('varxi = ',nk2.varxi)
    print('ratio = ',nk2.varxi / var_nk_xi)
    np.testing.assert_allclose(nk2.weight, nk.weight, rtol=0.02*tol_factor)
    np.testing.assert_allclose(nk2.xi, nk.xi, rtol=0.02*tol_factor)
    np.testing.assert_allclose(np.log(nk2.varxi), np.log(var_nk_xi), atol=0.4*tol_factor)

    # Use a random catalog without patches
    rk3 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    t0 = time.time()
    rk3.process(cat2, cat2p)
    t1 = time.time()
    print('Time for processing RK = ',t1-t0)

    nk3 = nk.copy()
    nk3.calculateXi(rk=rk3)
    print('xi = ',nk3.xi)
    print('varxi = ',nk3.varxi)
    print('ratio = ',nk3.varxi / var_nk_xi)
    np.testing.assert_allclose(nk3.weight, nk.weight, rtol=0.02*tol_factor)
    np.testing.assert_allclose(nk3.xi, nk.xi, rtol=0.02*tol_factor)
    np.testing.assert_allclose(np.log(nk3.varxi), np.log(var_nk_xi), atol=0.4*tol_factor)

    # KK
    # Smaller scales to capture the more local kappa correlations.
    kk = treecorr.KKCorrelation(bin_size=0.3, min_sep=6, max_sep=30., var_method='jackknife')
    t0 = time.time()
    kk.process(cat2p)
    t1 = time.time()
    print('Time for KK jackknife processing = ',t1-t0)
    print('xi = ',kk.xi)
    print('varxi = ',kk.varxi)
    print('ratio = ',kk.varxi / var_kk_xi)
    np.testing.assert_allclose(kk.weight, kk.weight)
    np.testing.assert_allclose(kk.xi, kk.xi)
    np.testing.assert_allclose(np.log(kk.varxi), np.log(var_kk_xi), atol=0.3*tol_factor)

    # Check sample covariance estimate
    cov_xi = kk.estimate_cov('sample')
    print('KK sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_kk_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_kk_xi, rtol=0.4*tol_factor)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kk.fits')
        kk.write(file_name, write_patch_results=True)
        kk4 = treecorr.KKCorrelation(bin_size=0.3, min_sep=6., max_sep=30.)
        kk4.read(file_name)
        cov4 = kk4.estimate_cov('jackknife')
        np.testing.assert_allclose(cov4, kk.cov)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_kk.dat')
    kk.write(file_name, write_patch_results=True)
    kk5 = treecorr.KKCorrelation(bin_size=0.3, min_sep=6., max_sep=30.)
    kk5.read(file_name)
    cov5 = kk5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, kk.cov)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_kk.hdf5')
        kk.write(file_name, write_patch_results=True)
        kk6 = treecorr.KKCorrelation(bin_size=0.3, min_sep=6., max_sep=30.)
        kk6.read(file_name)
        cov6 = kk6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, kk.cov)

    # KG
    # Same scales as we used for NG, which works fine with kappa as the "lens" too.
    kg = treecorr.KGCorrelation(bin_size=0.3, min_sep=10, max_sep=50., var_method='jackknife')
    t0 = time.time()
    kg.process(cat2p, cat2p)
    t1 = time.time()
    print('Time for KG jackknife processing = ',t1-t0)
    print('xi = ',kg.xi)
    print('varxi = ',kg.varxi)
    print('ratio = ',kg.varxi / var_kg_xi)
    np.testing.assert_allclose(kg.weight, kg.weight)
    np.testing.assert_allclose(kg.xi, kg.xi)
    np.testing.assert_allclose(np.log(kg.varxi), np.log(var_kg_xi), atol=0.3*tol_factor)

    # Check sample covariance estimate
    cov_xi = kg.estimate_cov('sample')
    print('KG sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_kg_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_kg_xi, rtol=0.4*tol_factor)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kg.fits')
        kg.write(file_name, write_patch_results=True)
        kg4 = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        kg4.read(file_name)
        cov4 = kg4.estimate_cov('jackknife')
        np.testing.assert_allclose(cov4, kg.cov)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_kg.dat')
    kg.write(file_name, write_patch_results=True)
    kg5 = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    kg5.read(file_name)
    cov5 = kg5.estimate_cov('jackknife')
    np.testing.assert_allclose(cov5, kg.cov)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_kg.hdf5')
        kg.write(file_name, write_patch_results=True)
        kg6 = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
        kg6.read(file_name)
        cov6 = kg6.estimate_cov('jackknife')
        np.testing.assert_allclose(cov6, kg.cov)

    # Do a real multi-statistic covariance.
    t0 = time.time()
    cov = treecorr.estimate_multi_cov([nk,kk,kg], 'jackknife')
    t1 = time.time()
    print('Time for jackknife cross-covariance = ',t1-t0)
    n1 = nk.nbins
    n2 = nk.nbins + kk.nbins
    np.testing.assert_allclose(cov[:n1,:n1], nk.cov)
    np.testing.assert_allclose(cov[n1:n2,n1:n2], kk.cov)
    np.testing.assert_allclose(cov[n2:,n2:], kg.cov)

    # Turn into a correlation matrix
    cor = cov / np.sqrt(cov.diagonal())[:,np.newaxis] / np.sqrt(cov.diagonal())[np.newaxis,:]

    print('nk-kk cross correlation = ',cor[:n1,n1:n2])
    print('nk-kg cross correlation = ',cor[:n1,n2:])
    print('kk-kg cross correlation = ',cor[n1:n2,n2:])
    # These should be rather large.  Most entries are > 0.1, so sum is much > 1.
    assert np.sum(np.abs(cor[:n1,n1:n2])) > 1
    assert np.sum(np.abs(cor[:n1,n2:])) > 1
    assert np.sum(np.abs(cor[n1:n2,n2:])) > 1

    # Reorder the data vector
    func = lambda corrs: np.concatenate([corrs[2].xi[:4], corrs[0].xi, corrs[1].xi[2:]])
    cov_alt = treecorr.estimate_multi_cov([nk,kk,kg], 'jackknife', func=func)
    np.testing.assert_allclose(cov_alt[:4,:4], cov[n2:n2+4,n2:n2+4])
    np.testing.assert_allclose(cov_alt[4:4+n1,4:4+n1], cov[:n1,:n1])
    np.testing.assert_allclose(cov_alt[4+n1:,4+n1:], cov[n1+2:n2,n1+2:n2])

    rk2 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
    rk2.process(cat2, cat2)
    with assert_raises(RuntimeError):
        nk2.calculateXi(rk=rk2)
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([nk,kk,kg], 'shot', func=func)

@timer
def test_save_patches():
    # Test the option to write the patches to disk
    try:
        import fitsio
    except ImportError:
        print('Skip test_save_patches, since fitsio not installed.')
        return

    if __name__ == '__main__':
        ngal = 10000
        npatch = 128
    else:
        ngal = 1000
        npatch = 8
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) ) + 100  # Put everything at large y, so smallish angle on sky
    z = rng.normal(0,s, (ngal,) )
    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)
    ra *= 180./np.pi
    dec *= 180./np.pi

    file_name = os.path.join('output','test_save_patches.fits')
    cat0 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
    cat0.write(file_name)


    # When catalog has explicit ra, dec, etc., then file names are patch000.fits, ...
    clear_save('patch%03d.fits', npatch)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', npatch=npatch,
                            save_patch_dir='output')
    assert len(cat1.patches) == npatch
    for i in range(npatch):
        patch_file_name = os.path.join('output','patch%03d.fits'%i)
        assert os.path.exists(patch_file_name)
        cat_i = treecorr.Catalog(patch_file_name, ra_col='ra', dec_col='dec',
                                 ra_units='rad', dec_units='rad', patch=i)
        assert not cat_i.loaded
        assert cat1.patches[i].loaded
        assert cat_i == cat1.patches[i]
        assert cat_i.loaded

    # When catalog is a file, then base name off of given file_name.
    clear_save('test_save_patches_%03d.fits', npatch)
    cat2 = treecorr.Catalog(file_name, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                            npatch=npatch, save_patch_dir='output')
    assert not cat2.loaded
    cat2.get_patches(low_mem=True)
    assert len(cat2.patches) == npatch
    assert cat2.loaded  # Making patches triggers load.  Also when write happens.
    for i in range(npatch):
        patch_file_name = os.path.join('output','test_save_patches_%03d.fits'%i)
        assert os.path.exists(patch_file_name)
        cat_i = treecorr.Catalog(patch_file_name, ra_col='ra', dec_col='dec',
                                 ra_units='rad', dec_units='rad', patch=i)
        assert not cat_i.loaded
        assert not cat2.patches[i].loaded
        assert cat_i == cat2.patches[i]
        assert cat_i.loaded
        assert cat2.patches[i].loaded

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_save_patches.hdf5')
        cat0.write(file_name)
        # And also try to match the type if HDF
        clear_save('test_save_patches_%03d.hdf5', npatch)
        cat2 = treecorr.Catalog(file_name, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                                npatch=npatch, save_patch_dir='output')
        assert not cat2.loaded
        cat2.get_patches(low_mem=True)
        assert len(cat2.patches) == npatch
        assert cat2.loaded  # Making patches triggers load.  Also when write happens.
        for i in range(npatch):
            patch_file_name = os.path.join('output','test_save_patches_%03d.hdf5'%i)
            assert os.path.exists(patch_file_name)
            cat_i = treecorr.Catalog(patch_file_name, ra_col='ra', dec_col='dec',
                                     ra_units='rad', dec_units='rad', patch=i)
            assert not cat_i.loaded
            assert not cat2.patches[i].loaded
            assert cat_i == cat2.patches[i]
            assert cat_i.loaded
            assert cat2.patches[i].loaded


    # Check x,y,z, as well as other possible columns
    w = rng.uniform(1,2, (ngal,) )
    g1 = rng.uniform(-0.5,0.5, (ngal,) )
    g2 = rng.uniform(-0.5,0.5, (ngal,) )
    k = rng.uniform(-1.2,1.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x, y=y, z=z, w=w, g1=g1, g2=g2, k=k, npatch=npatch)
    file_name2 = os.path.join('output','test_save_patches2.dat')
    cat3.write(file_name2)

    clear_save('test_save_patches2_%03d.fits', npatch)
    cat4 = treecorr.Catalog(file_name2,
                            x_col=1, y_col=2, z_col=3, w_col=4,
                            g1_col=5, g2_col=6, k_col=7, patch_col=8,
                            save_patch_dir='output')
    assert not cat4.loaded
    cat4.get_patches(low_mem=True)
    assert len(cat4.patches) == npatch
    assert cat4.loaded  # Making patches triggers load.
    for i in range(npatch):
        patch_file_name = os.path.join('output','test_save_patches2_%03d.fits'%i)
        assert os.path.exists(patch_file_name)
        cat_i = treecorr.Catalog(patch_file_name, patch=i,
                                 x_col='x', y_col='y', z_col='z', w_col='w',
                                 g1_col='g1', g2_col='g2', k_col='k')
        assert not cat_i.loaded
        assert not cat4.patches[i].loaded
        assert cat_i == cat4.patches[i]
        assert cat_i.loaded
        assert cat4.patches[i].loaded
    # Make sure making patch_centers doesn't screw things up.  (It used to.)
    cat4.patch_centers
    p4 = cat4.patches
    cat4._patches = None
    assert cat4.get_patches(low_mem=True) == p4
    cat4._patches = None
    assert cat4.get_patches(low_mem=False) == p4

    # If patches are made with patch_centers, then making patches doesn't trigger full load.
    # Note: for coverage reasons, only do x,y this time, and use wpos.
    clear_save('test_save_patches2_%03d.fits', npatch)
    cat5 = treecorr.Catalog(file_name2,
                            x_col=1, y_col=2, wpos_col=4, k_col=7,
                            patch_centers=cat4.patch_centers[:,:2],
                            save_patch_dir='output')
    assert not cat5.loaded
    cat5.get_patches(low_mem=True)
    assert len(cat5.patches) == npatch
    assert not cat5.loaded
    for i in range(npatch):
        patch_file_name = cat5.patches[i].file_name
        assert patch_file_name == os.path.join('output','test_save_patches2_%03d.fits'%i)
        assert os.path.exists(patch_file_name)
        cat_i = treecorr.Catalog(patch_file_name, patch=i,
                                 x_col='x', y_col='y', wpos_col='wpos', k_col='k')
        assert not cat_i.loaded
        assert not cat5.patches[i].loaded
        assert cat_i == cat5.patches[i]
        assert cat_i.loaded
        assert cat5.patches[i].loaded
    assert not cat5.loaded

    # Finally, test read/write commands explicitly.
    with CaptureLog() as cl:
        cat5.logger = cl.logger
        cat5.write_patches('output')
    print(cl.output)
    assert "Writing patch 3 to output/test_save_patches2_003.fits" in cl.output
    cat6 = treecorr.Catalog(file_name2,
                            x_col=1, y_col=2, wpos_col=4, k_col=7,
                            patch_centers=cat4.patch_centers[:,:2])
    with CaptureLog() as cl:
        cat6.logger = cl.logger
        cat6.read_patches('output')
    print(cl.output)
    assert "Patches created from files output/test_save_patches2_000.fits .. " in cl.output
    assert cat5.patches == cat6.patches

    # cat6 doesn't have save_patches set, so argument here is required.
    with assert_raises(ValueError):
        cat6.write_patches()
    with assert_raises(ValueError):
        cat6.read_patches()

@timer
def test_clusters():
    # The original version of J/K variance assumed that both catalogs had some items
    # in every patch.  But clusters can be very low density, so it can be very plausible
    # that some patches won't have any clusters in them.  This should be allowed.
    # (Thanks to Ariel Amsellem and Judit Prat for pointing out this bug in the
    # original implementation.)

    if __name__ == '__main__':
        npatch = 128
        nlens = 400   # Average of 3.13 clusters per patch.  So ~4% should have zero clusters.
        nsource = 50000
        size = 1000
        tol_factor = 1
    else:
        npatch = 32
        nlens = 60
        nsource = 1000
        size = 200
        tol_factor = 4
    rng = np.random.RandomState(1234)

    def make_gals():
        lens_x = rng.uniform(0,size,nlens)
        lens_y = rng.uniform(0,size,nlens)
        source_x = rng.uniform(0,size,nsource)
        source_y = rng.uniform(0,size,nsource)
        m = rng.uniform(0.05,0.2,nlens)

        # SIS model: g = g0/r
        dx = source_x - lens_x[:,np.newaxis]
        dy = source_y - lens_y[:,np.newaxis]
        rsq = dx**2 + dy**2
        g = dx + 1j * dy
        g *= g
        g /= rsq
        g /= np.sqrt(rsq)
        g *= -m[:,np.newaxis]
        g = np.sum(g,axis=0)
        source_g1 = g.real
        source_g2 = g.imag
        source_g1 += rng.normal(0,3.e-3)
        source_g2 += rng.normal(0,3.e-3)
        return lens_x, lens_y, source_x, source_y, source_g1, source_g2

    file_name = 'data/test_clusters_{}.npz'.format(nlens)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ngs = []
        for run in range(nruns):
            print(run)
            lens_x, lens_y, source_x, source_y, source_g1, source_g2 = make_gals()
            cat1 = treecorr.Catalog(x=lens_x, y=lens_y)
            cat2 = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2)
            ng = treecorr.NGCorrelation(bin_size=0.4, min_sep=1., max_sep=20.)
            ng.process(cat1, cat2)
            all_ngs.append(ng)

        mean_xi = np.mean([ng.xi for ng in all_ngs], axis=0)
        var_xi = np.var([ng.xi for ng in all_ngs], axis=0)
        mean_varxi = np.mean([ng.varxi for ng in all_ngs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    var_xi = data['var_xi']
    mean_varxi = data['mean_varxi']

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)

    # First run with the normal variance estimate, which is too small.
    lens_x, lens_y, source_x, source_y, source_g1, source_g2 = make_gals()
    cat1 = treecorr.Catalog(x=lens_x, y=lens_y)
    cat2 = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2)
    ng1 = treecorr.NGCorrelation(bin_size=0.4, min_sep=1., max_sep=20.)
    t0 = time.time()
    ng1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',ng1.weight)
    print('xi = ',ng1.xi)
    print('varxi = ',ng1.varxi)
    print('mean_varxi = ',mean_varxi)
    print('npair = ',ng1.npairs)
    print('pullsq for xi = ',(ng1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((ng1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((ng1.xi - mean_xi)**2/var_xi, 25) # within 5 sigma
    np.testing.assert_allclose(np.log(ng1.varxi), np.log(mean_varxi), atol=0.5*tol_factor)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxi, var_xi)
    np.testing.assert_array_less(ng1.varxi, var_xi)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    # Note: This turns out to work significantly better if cat1 is used to make the patches.
    # Otherwise the number of lenses per patch varies a lot, which affects the variance estimate.
    # But that means we need to keep the w=0 object in the catalog, so all objects get a patch.
    cat2p = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2, npatch=npatch)
    cat1p = treecorr.Catalog(x=lens_x, y=lens_y, patch_centers=cat2p.patch_centers)
    print('tot n = ',nlens)
    print('Patch\tNlens')
    nwith0 = 0
    for i in range(npatch):
        n = np.sum(cat1p.w[cat1p.patch==i])
        #print('%d\t%d'%(i,n))
        if n == 0: nwith0 += 1
    print('Found %s patches with no lenses'%nwith0)
    assert nwith0 > 0  # This is the point of this test!
    ng2 = treecorr.NGCorrelation(bin_size=0.4, min_sep=1., max_sep=20., var_method='shot')
    t0 = time.time()
    ng2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for shot processing = ',t1-t0)
    print('weight = ',ng2.weight)
    print('xi = ',ng2.xi)
    print('varxi = ',ng2.varxi)
    np.testing.assert_allclose(ng2.weight, ng1.weight, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng2.varxi, ng1.varxi, rtol=3.e-2*tol_factor)

    # Now run with jackknife variance estimate.  Should be much better.
    ng3 = treecorr.NGCorrelation(bin_size=0.4, min_sep=1., max_sep=20., var_method='jackknife',
                                 rng=rng)
    t0 = time.time()
    ng3.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for jackknife processing = ',t1-t0)
    print('xi = ',ng3.xi)
    print('varxi = ',ng3.varxi)
    print('ratio = ',ng3.varxi / var_xi)
    np.testing.assert_allclose(ng3.weight, ng2.weight)
    np.testing.assert_allclose(ng3.xi, ng2.xi)
    np.testing.assert_allclose(np.log(ng3.varxi), np.log(var_xi), atol=0.3*tol_factor)

    # Check sample covariance estimate
    t0 = time.time()
    with assert_raises(RuntimeError):
        ng3.estimate_cov('sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)

    # Check marked_bootstrap covariance estimate
    t0 = time.time()
    cov_boot = ng3.estimate_cov('marked_bootstrap')
    t1 = time.time()
    print('Time to calculate marked_bootstrap covariance = ',t1-t0)
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(cov_boot.diagonal(), var_xi, rtol=0.3*tol_factor)

    # Check bootstrap covariance estimate.
    t0 = time.time()
    cov_boot = ng3.estimate_cov('bootstrap')
    t1 = time.time()
    print('Time to calculate bootstrap covariance = ',t1-t0)
    print('varxi = ',cov_boot.diagonal())
    print('ratio = ',cov_boot.diagonal() / var_xi)
    np.testing.assert_allclose(cov_boot.diagonal(), var_xi, rtol=0.5*tol_factor)

    # Use a random catalog
    # In this case the locations of the source catalog are fine to use as our random catalog,
    # since they fill the region where the lenses are allowed to be.
    rg3 = treecorr.NGCorrelation(bin_size=0.4, min_sep=1., max_sep=20.)
    t0 = time.time()
    rg3.process(cat2p, cat2p)
    t1 = time.time()
    print('Time for processing RG = ',t1-t0)

    ng3b = ng3.copy()
    ng3b.calculateXi(rg=rg3)
    print('xi = ',ng3b.xi)
    print('varxi = ',ng3b.varxi)
    print('ratio = ',ng3b.varxi / var_xi)
    np.testing.assert_allclose(ng3b.weight, ng3.weight, rtol=0.02*tol_factor)
    np.testing.assert_allclose(ng3b.xi, ng3.xi, rtol=0.02*tol_factor)
    np.testing.assert_allclose(np.log(ng3b.varxi), np.log(var_xi), atol=0.3*tol_factor)

@timer
def test_brute_jk():
    # With bin_slop = 0, the jackknife calculation from patches should match a
    # brute force calcaulation where we literally remove one patch at a time to make
    # the vectors.
    if __name__ == '__main__':
        nside = 100
        nlens = 100
        nsource = 5000
        npatch = 32
        rand_factor = 5
        tol_factor = 1
    else:
        nside = 100
        nlens = 30
        nsource = 500
        npatch = 16
        rand_factor = 5
        tol_factor = 3

    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(nside, rng)
    indx = rng.choice(range(len(x)),nsource,replace=False)
    source_cat = treecorr.Catalog(x=x[indx], y=y[indx],
                                  g1=g1[indx], g2=g2[indx], k=k[indx],
                                  npatch=npatch)
    print('source_cat patches = ',np.unique(source_cat.patch))
    print('len = ',source_cat.nobj, source_cat.ntot)
    assert source_cat.nobj == nsource
    indx = rng.choice(np.where(k>0)[0],nlens,replace=False)
    print('indx = ',indx)
    lens_cat = treecorr.Catalog(x=x[indx], y=y[indx], k=k[indx],
                                g1=g1[indx], g2=g2[indx],
                                patch_centers=source_cat.patch_centers)
    print('lens_cat patches = ',np.unique(lens_cat.patch))
    print('len = ',lens_cat.nobj, lens_cat.ntot)
    assert lens_cat.nobj == nlens

    rand_source_cat = treecorr.Catalog(x=rng.uniform(0,1000,nsource*rand_factor),
                                       y=rng.uniform(0,1000,nsource*rand_factor),
                                       patch_centers=source_cat.patch_centers)
    print('rand_source_cat patches = ',np.unique(rand_source_cat.patch))
    print('len = ',rand_source_cat.nobj, rand_source_cat.ntot)
    rand_lens_cat = treecorr.Catalog(x=rng.uniform(0,1000,nlens*rand_factor),
                                     y=rng.uniform(0,1000,nlens*rand_factor),
                                     patch_centers=source_cat.patch_centers)
    print('rand_lens_cat patches = ',np.unique(rand_lens_cat.patch))
    print('len = ',rand_lens_cat.nobj, rand_lens_cat.ntot)

    # Start with NK, since relatively simple.
    nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True,
                                var_method='jackknife')
    nk.process(lens_cat, source_cat)
    print('TreeCorr jackknife:')
    print('nk = ',nk.xi)
    print('var = ',nk.varxi)

    # Now do this using brute force calculation.
    print('Direct jackknife:')
    xi_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i])
        nk1 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        nk1.process(lens_cat1, source_cat1)
        xi_list.append(nk1.xi)
    xi_list = np.array(xi_list)
    xi = np.mean(xi_list, axis=0)
    print('mean xi = ',xi)
    C = np.cov(xi_list.T, bias=True) * (len(xi_list)-1)
    varxi = np.diagonal(C)
    print('varxi = ',varxi)
    # xi isn't exact because of the variation in denominators, which doesn't commute with the mean.
    # nk.xi is more accurate for the overall estimate of the correlation function.
    # The difference gets less as npatch increases.
    np.testing.assert_allclose(nk.xi, xi, rtol=0.01 * tol_factor)
    np.testing.assert_allclose(nk.varxi, varxi)

    # Repeat with randoms.
    rk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
    rk.process(rand_lens_cat, source_cat)
    nk.calculateXi(rk=rk)
    print('With randoms:')
    print('nk = ',nk.xi)
    print('var = ',nk.varxi)

    print('Direct jackknife:')
    xi_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        rand_lens_cat1 = treecorr.Catalog(x=rand_lens_cat.x[rand_lens_cat.patch != i],
                                          y=rand_lens_cat.y[rand_lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i])
        nk1 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        nk1.process(lens_cat1, source_cat1)
        rk1 = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        rk1.process(rand_lens_cat1, source_cat1)
        nk1.calculateXi(rk=rk1)
        xi_list.append(nk1.xi)
    xi_list = np.array(xi_list)
    C = np.cov(xi_list.T, bias=True) * (len(xi_list)-1)
    varxi = np.diagonal(C)
    print('var = ',varxi)
    np.testing.assert_allclose(nk.varxi, varxi)

    # Repeat for NG, GG, KK, KG
    ng = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True,
                                var_method='jackknife')
    ng.process(lens_cat, source_cat)
    gg = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True,
                                var_method='jackknife')
    gg.process(source_cat)
    kk = treecorr.KKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True,
                                var_method='jackknife')
    kk.process(source_cat)
    kg = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True,
                                var_method='jackknife')
    kg.process(lens_cat, source_cat)

    ng_xi_list = []
    gg_xip_list = []
    gg_xim_list = []
    kk_xi_list = []
    kg_xi_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i],
                                     k=lens_cat.k[lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        ng1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        ng1.process(lens_cat1, source_cat1)
        ng_xi_list.append(ng1.xi)
        gg1 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        gg1.process(source_cat1)
        gg_xip_list.append(gg1.xip)
        gg_xim_list.append(gg1.xim)
        kk1 = treecorr.KKCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        kk1.process(source_cat1)
        kk_xi_list.append(kk1.xi)
        kg1 = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        kg1.process(lens_cat1, source_cat1)
        kg_xi_list.append(kg1.xi)
    ng_xi_list = np.array(ng_xi_list)
    varxi = np.diagonal(np.cov(ng_xi_list.T, bias=True)) * (len(ng_xi_list)-1)
    print('NG: treecorr jackknife varxi = ',ng.varxi)
    print('NG: direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(ng.varxi, varxi)
    gg_xip_list = np.array(gg_xip_list)
    varxi = np.diagonal(np.cov(gg_xip_list.T, bias=True)) * (len(gg_xip_list)-1)
    print('GG: treecorr jackknife varxip = ',gg.varxip)
    print('GG: direct jackknife varxip = ',varxi)
    np.testing.assert_allclose(gg.varxip, varxi)
    gg_xim_list = np.array(gg_xim_list)
    varxi = np.diagonal(np.cov(gg_xim_list.T, bias=True)) * (len(gg_xim_list)-1)
    print('GG: treecorr jackknife varxim = ',gg.varxim)
    print('GG: direct jackknife varxim = ',varxi)
    np.testing.assert_allclose(gg.varxim, varxi)
    kk_xi_list = np.array(kk_xi_list)
    varxi = np.diagonal(np.cov(kk_xi_list.T, bias=True)) * (len(kk_xi_list)-1)
    print('KK: treecorr jackknife varxi = ',kk.varxi)
    print('KK: direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(kk.varxi, varxi)
    kg_xi_list = np.array(kg_xi_list)
    varxi = np.diagonal(np.cov(kg_xi_list.T, bias=True)) * (len(kg_xi_list)-1)
    print('KG: treecorr jackknife varxi = ',kg.varxi)
    print('KG: direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(kg.varxi, varxi)

    # Repeat NG with randoms.
    rg = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
    rg.process(rand_lens_cat, source_cat)
    ng.calculateXi(rg=rg)

    xi_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        rand_lens_cat1 = treecorr.Catalog(x=rand_lens_cat.x[rand_lens_cat.patch != i],
                                          y=rand_lens_cat.y[rand_lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        ng1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        ng1.process(lens_cat1, source_cat1)
        rg1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=30., brute=True)
        rg1.process(rand_lens_cat1, source_cat1)
        ng1.calculateXi(rg=rg1)
        xi_list.append(ng1.xi)
    xi_list = np.array(xi_list)
    C = np.cov(xi_list.T, bias=True) * (len(xi_list)-1)
    varxi = np.diagonal(C)
    print('NG with randoms:')
    print('treecorr jackknife varxi = ',ng.varxi)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(ng.varxi, varxi)

    # Finally, test NN, which is complicated, since several different combinations of randoms.
    # 1. (DD-RR)/RR
    # 2. (DD-2DR+RR)/RR
    # 3. (DD-2RD+RR)/RR
    # 4. (DD-DR-RD+RR)/RR
    dd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dd.process(lens_cat, source_cat)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rr.process(rand_lens_cat, rand_source_cat)
    rd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rd.process(rand_lens_cat, source_cat)
    dr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dr.process(lens_cat, rand_source_cat)

    # Now do this using brute force calculation.
    xi1_list = []
    xi2_list = []
    xi3_list = []
    xi4_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i])
        rand_lens_cat1 = treecorr.Catalog(x=rand_lens_cat.x[rand_lens_cat.patch != i],
                                          y=rand_lens_cat.y[rand_lens_cat.patch != i])
        rand_source_cat1 = treecorr.Catalog(x=rand_source_cat.x[rand_source_cat.patch != i],
                                            y=rand_source_cat.y[rand_source_cat.patch != i])
        dd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dd1.process(lens_cat1, source_cat1)
        rr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rr1.process(rand_lens_cat1, rand_source_cat1)
        rd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rd1.process(rand_lens_cat1, source_cat1)
        dr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dr1.process(lens_cat1, rand_source_cat1)
        xi1_list.append(dd1.calculateXi(rr=rr1)[0])
        xi2_list.append(dd1.calculateXi(rr=rr1, dr=dr1)[0])
        xi3_list.append(dd1.calculateXi(rr=rr1, rd=rd1)[0])
        xi4_list.append(dd1.calculateXi(rr=rr1, dr=dr1, rd=rd1)[0])

    print('(DD-RR)/RR')
    xi1_list = np.array(xi1_list)
    xi1, varxi1 = dd.calculateXi(rr=rr)
    varxi = np.diagonal(np.cov(xi1_list.T, bias=True)) * (len(xi1_list)-1)
    print('treecorr jackknife varxi = ',varxi1)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2DR+RR)/RR')
    xi2_list = np.array(xi2_list)
    xi2, varxi2 = dd.calculateXi(rr=rr, dr=dr)
    varxi = np.diagonal(np.cov(xi2_list.T, bias=True)) * (len(xi2_list)-1)
    print('treecorr jackknife varxi = ',varxi2)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2RD+RR)/RR')
    xi3_list = np.array(xi3_list)
    xi3, varxi3 = dd.calculateXi(rr=rr, rd=rd)
    varxi = np.diagonal(np.cov(xi3_list.T, bias=True)) * (len(xi3_list)-1)
    print('treecorr jackknife varxi = ',varxi3)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-DR-RD+RR)/RR')
    xi4_list = np.array(xi4_list)
    xi4, varxi4 = dd.calculateXi(rr=rr, rd=rd, dr=dr)
    varxi = np.diagonal(np.cov(xi4_list.T, bias=True)) * (len(xi4_list)-1)
    print('treecorr jackknife varxi = ',varxi4)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

@timer
def test_lowmem():
    # Test using patches to keep the memory usage lower.
    try:
        import fitsio
    except ImportError:
        print('Skip test_lowmem, since fitsio not installed.')
        return

    if __name__ == '__main__':
        ngal = 2000000
        npatch = 64
        himem = 1.e8   # These are empirical of course.  The point is himem >> lomem.
        lomem = 4.e6
    else:
        ngal = 100000
        npatch = 16
        himem = 5.e6
        lomem = 4.e5
    rng = np.random.RandomState(8675309)
    x = rng.uniform(-20,20, (ngal,) )
    y = rng.uniform(80,120, (ngal,) )  # Put everything at large y, so smallish angle on sky
    z = rng.uniform(-20,20, (ngal,) )
    ra, dec, r = coord.CelestialCoord.xyz_to_radec(x,y,z, return_r=True)
    ra *= 180./np.pi  # -> deg
    dec *= 180./np.pi

    file_name = os.path.join('output','test_lowmem.fits')
    orig_cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
    orig_cat.write(file_name)
    del orig_cat

    try:
        import guppy
        hp = guppy.hpy()
        hp.setrelheap()
    except Exception:
        hp = None

    partial_cat = treecorr.Catalog(file_name, every_nth=100,
                                   ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                                   npatch=npatch)

    patch_centers = partial_cat.patch_centers
    del partial_cat

    full_cat = treecorr.Catalog(file_name,
                                ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                                patch_centers=patch_centers)

    dd = treecorr.NNCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    dd.process(full_cat)
    t1 = time.time()
    s1 = hp.heap().size if hp else 2*himem
    print('regular: ',s1, t1-t0, s1-s0)
    assert s1-s0 > himem  # This version uses a lot of memory.

    npairs1 = dd.npairs

    full_cat.unload()
    dd.clear()

    # Remake with save_patch_dir.
    clear_save('test_lowmem_%03d.fits', npatch)
    save_cat = treecorr.Catalog(file_name,
                                ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                                patch_centers=patch_centers, save_patch_dir='output')

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    dd.process(save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem  # This version uses a lot less memory
    npairs2 = dd.npairs
    np.testing.assert_array_equal(npairs1, npairs2)

    # Check running as a cross-correlation
    save_cat.unload()
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    dd.process(save_cat, save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem x: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem
    npairs3 = dd.npairs
    np.testing.assert_array_equal(npairs3, npairs2)

    # Check other combinations
    # Use a smaller catalog to run faster.
    # And along the way we'll check other aspects of the patch usage.
    g1 = rng.uniform(-0.1,0.1, (ngal//100,) )
    g2 = rng.uniform(-0.1,0.1, (ngal//100,) )
    k = rng.uniform(-0.1,0.1, (ngal//100,) )
    gk_cat0 = treecorr.Catalog(ra=ra[:ngal//100], dec=dec[:ngal//100], r=r[:ngal//100],
                               ra_units='deg', dec_units='deg',
                               g1=g1, g2=g2, k=k,
                               npatch=4)
    patch_centers = gk_cat0.patch_centers
    file_name = os.path.join('output','test_lowmem_gk.fits')
    gk_cat0.write(file_name)
    del gk_cat0
    if hp:
        if __name__ == "__main__":
            hp.setrelheap()
        else:
            # For nosetests, turn off the rest of the guppy stuff, since they are slow,
            # and we don't actually bother doing any asserts with them aftet here.
            hp = None

    # First GG with normal ra,dec from a file
    clear_save('test_lowmem_gk_%03d.fits', npatch)
    gk_cat1 = treecorr.Catalog(file_name,
                               ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                               g1_col='g1', g2_col='g2', k_col='k',
                               patch_centers=patch_centers)
    gk_cat2 = treecorr.Catalog(file_name,
                               ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                               g1_col='g1', g2_col='g2', k_col='k',
                               patch_centers=patch_centers, save_patch_dir='output')

    gg1 = treecorr.GGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    gg1.process(gk_cat1)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('GG1: ',s1, t1-t0, s1-s0)
    gk_cat1.unload()
    gg2 = treecorr.GGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    gg2.process(gk_cat2, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('GG2: ',s1, t1-t0, s1-s0)
    gk_cat2.unload()
    np.testing.assert_allclose(gg1.xip, gg2.xip)
    np.testing.assert_allclose(gg1.xim, gg2.xim)
    np.testing.assert_allclose(gg1.weight, gg2.weight)

    # NG, still with the same file, but cross correlation.
    ng1 = treecorr.NGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    ng1.process(gk_cat1, gk_cat1)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('NG1: ',s1, t1-t0, s1-s0)
    gk_cat1.unload()
    ng2 = treecorr.NGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    ng2.process(gk_cat2, gk_cat2, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('NG2: ',s1, t1-t0, s1-s0)
    gk_cat2.unload()
    np.testing.assert_allclose(ng1.xi, ng2.xi)
    np.testing.assert_allclose(ng1.weight, ng2.weight)

    # KK with r_col now to test that that works properly.
    clear_save('test_lowmem_gk_%03d.fits', npatch)
    gk_cat1 = treecorr.Catalog(file_name,
                               ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                               r_col='r', g1_col='g1', g2_col='g2', k_col='k',
                               patch_centers=patch_centers)
    gk_cat2 = treecorr.Catalog(file_name,
                               ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg',
                               r_col='r', g1_col='g1', g2_col='g2', k_col='k',
                               patch_centers=patch_centers, save_patch_dir='output')

    kk1 = treecorr.KKCorrelation(bin_size=0.5, min_sep=1., max_sep=20.)
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kk1.process(gk_cat1)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('KK1: ',s1, t1-t0, s1-s0)
    gk_cat1.unload()
    kk2 = treecorr.KKCorrelation(bin_size=0.5, min_sep=1., max_sep=20.)
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kk2.process(gk_cat2, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('KK2: ',s1, t1-t0, s1-s0)
    gk_cat2.unload()
    np.testing.assert_allclose(kk1.xi, kk2.xi)
    np.testing.assert_allclose(kk1.weight, kk2.weight)

    # NK with r_col still, but not from a file.
    clear_save('patch%03d.fits', npatch)
    gk_cat1 = treecorr.Catalog(ra=ra[:ngal//100], dec=dec[:ngal//100], r=r[:ngal//100],
                               g1=g1[:ngal//100], g2=g2[:ngal//100], k=k[:ngal//100],
                               ra_units='deg', dec_units='deg',
                               patch_centers=patch_centers)
    gk_cat2 = treecorr.Catalog(ra=ra[:ngal//100], dec=dec[:ngal//100], r=r[:ngal//100],
                               g1=g1[:ngal//100], g2=g2[:ngal//100], k=k[:ngal//100],
                               ra_units='deg', dec_units='deg',
                               patch_centers=patch_centers, save_patch_dir='output')

    nk1 = treecorr.NKCorrelation(bin_size=0.5, min_sep=1., max_sep=20.)
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    nk1.process(gk_cat1, gk_cat1)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('NK1: ',s1, t1-t0, s1-s0)
    gk_cat1.unload()
    nk2 = treecorr.NKCorrelation(bin_size=0.5, min_sep=1., max_sep=20.)
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    nk2.process(gk_cat2, gk_cat2, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('NK2: ',s1, t1-t0, s1-s0)
    gk_cat2.unload()
    np.testing.assert_allclose(nk1.xi, nk2.xi)
    np.testing.assert_allclose(nk1.weight, nk2.weight)

    # KG without r_col, and not from a file.
    clear_save('patch%03d.fits', npatch)
    gk_cat1 = treecorr.Catalog(ra=ra[:ngal//100], dec=dec[:ngal//100],
                               g1=g1[:ngal//100], g2=g2[:ngal//100], k=k[:ngal//100],
                               ra_units='deg', dec_units='deg',
                               patch_centers=patch_centers)
    gk_cat2 = treecorr.Catalog(ra=ra[:ngal//100], dec=dec[:ngal//100],
                               g1=g1[:ngal//100], g2=g2[:ngal//100], k=k[:ngal//100],
                               ra_units='deg', dec_units='deg',
                               patch_centers=patch_centers, save_patch_dir='output')

    kg1 = treecorr.KGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kg1.process(gk_cat1, gk_cat1)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('KG1: ',s1, t1-t0, s1-s0)
    gk_cat1.unload()
    kg2 = treecorr.KGCorrelation(bin_size=0.5, min_sep=1., max_sep=30., sep_units='arcmin')
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kg2.process(gk_cat2, gk_cat2, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('KG2: ',s1, t1-t0, s1-s0)
    gk_cat2.unload()
    np.testing.assert_allclose(kg1.xi, kg2.xi)
    np.testing.assert_allclose(kg1.weight, kg2.weight)

@timer
def test_config():
    """Test using npatch and var_method in config file
    """
    try:
        import fitsio
    except ImportError:
        print('Skip test_config, since fitsio not installed.')
        return

    get_from_wiki('Aardvark.fit')
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.yaml')

    config['gg_file_name'] = 'output/Aardvark.boot.fits'
    config['var_method'] = 'bootstrap'
    config['num_bootstrap'] = 1000
    if __name__ == '__main__':
        logger = treecorr.config.setup_logger(2)
        config['npatch'] = 64
    else:
        logger = treecorr.config.setup_logger(0)
        config['npatch'] = 8
        config['verbose'] = 0
        config['last_row'] = 10000

    cat = treecorr.Catalog(file_name, config, logger=logger)
    gg = treecorr.GGCorrelation(config, logger=logger)
    gg.process(cat)

    # Check that we get the same result using the corr2 function
    treecorr.corr2(config, logger=logger)
    gg2 = treecorr.GGCorrelation(config)
    gg2.read(config['gg_file_name'])
    print('gg.varxip = ',gg.varxip)
    print('gg2.varxip = ',gg2.varxip)

    # Bootstrap has intrinisic randomness, so this doesn't get all that close actually.
    np.testing.assert_allclose(np.log(gg.varxip), np.log(gg2.varxip), atol=0.6)

    # Jackknife should be exactly equal (so long as npatch is 2^n), since deterministic.
    varxi_jk = gg.estimate_cov('jackknife').diagonal()
    config['var_method'] = 'jackknife'
    treecorr.corr2(config, logger=logger)
    gg2.read(config['gg_file_name'])
    print('gg.varxi = ',varxi_jk)
    print('gg2.varxip = ',gg2.varxip)
    print('gg2.varxim = ',gg2.varxim)
    np.testing.assert_allclose(varxi_jk, np.concatenate([gg2.varxip, gg2.varxim]), rtol=1.e-10)

@timer
def test_finalize_false():
    # Test the finalize=false option to do a full calculation in stages.

    nside = 50
    nlens = 2000
    npatch = 16
    ngal = nside**2

    rng = np.random.RandomState(1234)
    # Make three independent data sets
    x_1, y_1, g1_1, g2_1, k_1 = generate_shear_field(nside, rng)
    x_2, y_2, g1_2, g2_2, k_2 = generate_shear_field(nside, rng)
    x_3, y_3, g1_3, g2_3, k_3 = generate_shear_field(nside, rng)

    # Make a single catalog with all three together
    cat = treecorr.Catalog(x=np.concatenate([x_1, x_2, x_3]),
                           y=np.concatenate([y_1, y_2, y_3]),
                           g1=np.concatenate([g1_1, g1_2, g1_3]),
                           g2=np.concatenate([g2_1, g2_2, g2_3]),
                           k=np.concatenate([k_1, k_2, k_3]),
                           npatch=npatch)

    # Now the three separately, using the same patch centers
    cat1 = treecorr.Catalog(x=x_1, y=y_1, g1=g1_1, g2=g2_1, k=k_1, patch_centers=cat.patch_centers)
    cat2 = treecorr.Catalog(x=x_2, y=y_2, g1=g1_2, g2=g2_2, k=k_2, patch_centers=cat.patch_centers)
    cat3 = treecorr.Catalog(x=x_3, y=y_3, g1=g1_3, g2=g2_3, k=k_3, patch_centers=cat.patch_centers)

    np.testing.assert_array_equal(cat1.patch, cat.patch[0:ngal])
    np.testing.assert_array_equal(cat2.patch, cat.patch[ngal:2*ngal])
    np.testing.assert_array_equal(cat3.patch, cat.patch[2*ngal:3*ngal])

    # NK
    nk1 = treecorr.NKCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    nk1.process(cat, cat)

    nk2 = treecorr.NKCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    nk2.process(cat1, cat1, initialize=True, finalize=False)
    nk2.process(cat1, cat2, initialize=False, finalize=False)
    nk2.process(cat1, cat3, initialize=False, finalize=False)
    nk2.process(cat2, cat1, initialize=False, finalize=False)
    nk2.process(cat2, cat2, initialize=False, finalize=False)
    nk2.process(cat2, cat3, initialize=False, finalize=False)
    nk2.process(cat3, cat1, initialize=False, finalize=False)
    nk2.process(cat3, cat2, initialize=False, finalize=False)
    nk2.process(cat3, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(nk1.npairs, nk2.npairs)
    np.testing.assert_allclose(nk1.weight, nk2.weight)
    np.testing.assert_allclose(nk1.meanr, nk2.meanr)
    np.testing.assert_allclose(nk1.meanlogr, nk2.meanlogr)
    np.testing.assert_allclose(nk1.xi, nk2.xi)
    np.testing.assert_allclose(nk1.varxi, nk2.varxi)
    np.testing.assert_allclose(nk1.cov, nk2.cov)

    # KK
    kk1 = treecorr.KKCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    kk1.process(cat)

    kk2 = treecorr.KKCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    kk2.process(cat1, initialize=True, finalize=False)
    kk2.process(cat2, initialize=False, finalize=False)
    kk2.process(cat3, initialize=False, finalize=False)
    kk2.process(cat1, cat2, initialize=False, finalize=False)
    kk2.process(cat1, cat3, initialize=False, finalize=False)
    kk2.process(cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kk1.npairs, kk2.npairs)
    np.testing.assert_allclose(kk1.weight, kk2.weight)
    np.testing.assert_allclose(kk1.meanr, kk2.meanr)
    np.testing.assert_allclose(kk1.meanlogr, kk2.meanlogr)
    np.testing.assert_allclose(kk1.xi, kk2.xi)
    np.testing.assert_allclose(kk1.varxi, kk2.varxi)
    np.testing.assert_allclose(kk1.cov, kk2.cov)

    # NG
    ng1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    ng1.process(cat, cat)

    ng2 = treecorr.NGCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife')
    ng2.process(cat1, cat1, initialize=True, finalize=False)
    ng2.process(cat1, cat2, initialize=False, finalize=False)
    ng2.process(cat1, cat3, initialize=False, finalize=False)
    ng2.process(cat2, cat1, initialize=False, finalize=False)
    ng2.process(cat2, cat2, initialize=False, finalize=False)
    ng2.process(cat2, cat3, initialize=False, finalize=False)
    ng2.process(cat3, cat1, initialize=False, finalize=False)
    ng2.process(cat3, cat2, initialize=False, finalize=False)
    ng2.process(cat3, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ng1.npairs, ng2.npairs)
    np.testing.assert_allclose(ng1.weight, ng2.weight)
    np.testing.assert_allclose(ng1.meanr, ng2.meanr)
    np.testing.assert_allclose(ng1.meanlogr, ng2.meanlogr)
    np.testing.assert_allclose(ng1.xi, ng2.xi, atol=1.e-10)
    np.testing.assert_allclose(ng1.varxi, ng2.varxi)
    np.testing.assert_allclose(ng1.cov, ng2.cov, atol=1.e-12)

    # GG
    gg1 = treecorr.GGCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife',
                                 bin_slop=0)
    gg1.process(cat)

    gg2 = treecorr.GGCorrelation(bin_size=0.3, min_sep=20, max_sep=100., var_method='jackknife',
                                 bin_slop=0)
    gg2.process(cat1, initialize=True, finalize=False)
    gg2.process(cat2, initialize=False, finalize=False)
    gg2.process(cat3, initialize=False, finalize=False)
    gg2.process(cat1, cat2, initialize=False, finalize=False)
    gg2.process(cat1, cat3, initialize=False, finalize=False)
    gg2.process(cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(gg1.npairs, gg2.npairs)
    np.testing.assert_allclose(gg1.weight, gg2.weight)
    np.testing.assert_allclose(gg1.meanr, gg2.meanr)
    np.testing.assert_allclose(gg1.meanlogr, gg2.meanlogr)
    np.testing.assert_allclose(gg1.xip, gg2.xip)
    np.testing.assert_allclose(gg1.xim, gg2.xim)
    np.testing.assert_allclose(gg1.varxip, gg2.varxip)
    np.testing.assert_allclose(gg1.varxim, gg2.varxim)
    np.testing.assert_allclose(gg1.cov, gg2.cov)

    # KG
    kg1 = treecorr.KGCorrelation(bin_size=0.1, min_sep=20, max_sep=100., var_method='sample',
                                 bin_slop=0)
    kg1.process(cat, cat)

    kg2 = treecorr.KGCorrelation(bin_size=0.1, min_sep=20, max_sep=100., var_method='sample',
                                 bin_slop=0)
    kg2.process(cat1, cat1, initialize=True, finalize=False)
    kg2.process(cat1, cat2, initialize=False, finalize=False)
    kg2.process(cat1, cat3, initialize=False, finalize=False)
    kg2.process(cat2, cat1, initialize=False, finalize=False)
    kg2.process(cat2, cat2, initialize=False, finalize=False)
    kg2.process(cat2, cat3, initialize=False, finalize=False)
    kg2.process(cat3, cat1, initialize=False, finalize=False)
    kg2.process(cat3, cat2, initialize=False, finalize=False)
    kg2.process(cat3, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kg1.npairs, kg2.npairs)
    np.testing.assert_allclose(kg1.weight, kg2.weight)
    np.testing.assert_allclose(kg1.meanr, kg2.meanr)
    np.testing.assert_allclose(kg1.meanlogr, kg2.meanlogr)
    np.testing.assert_allclose(kg1.xi, kg2.xi)
    np.testing.assert_allclose(kg1.varxi, kg2.varxi)
    np.testing.assert_allclose(kg1.cov, kg2.cov)

    # NN
    nn1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=20, max_sep=100.)
    nn1.process(cat)

    nn2 = treecorr.NNCorrelation(bin_size=0.3, min_sep=20, max_sep=100.)
    nn2.process(cat1, initialize=True, finalize=False)
    nn2.process(cat2, initialize=False, finalize=False)
    nn2.process(cat3, initialize=False, finalize=False)
    nn2.process(cat1, cat2, initialize=False, finalize=False)
    nn2.process(cat1, cat3, initialize=False, finalize=False)
    nn2.process(cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(nn1.npairs, nn2.npairs)
    np.testing.assert_allclose(nn1.weight, nn2.weight)
    np.testing.assert_allclose(nn1.meanr, nn2.meanr)
    np.testing.assert_allclose(nn1.meanlogr, nn2.meanlogr)

@timer
def test_empty_patches():
    # This is an even more extreme test than the test_clusters test.
    # In that test about 4% of the patches had no objects.
    # In this test, most of the patches have no objects.
    # This used to cause a run-time error when np.sum ended up with empty arrays,
    # which it didn't like and raises an exception.
    # Probbly most of the time, this will be user error that the patches aren't set up properly,
    # but it is at least conceivable that you might want to do something real that would cause
    # this, so this test makes sure it works correctly.

    # Thanks to Joe Zuntz for pointing out this bug in issue #123.

    npatch = 25
    nlens = 10
    nsource = 5000
    rng = np.random.RandomState(1234)

    # Put all the lenses in a tight patch, but the sources in a much larger patch.
    # This seems like the least implausible scenario where this could happen.
    lens_x = rng.uniform(-1,1,nlens)
    lens_y = rng.uniform(-1,1,nlens)
    source_x = rng.uniform(-100,100,nsource)
    source_y = rng.uniform(-100,100,nsource)

    # I don't actually care about the signal here being realistic, but might as well...
    # SIS model: g = g0/r
    lens_mass = rng.uniform(0.05,0.2,nlens)
    lens_e1 = rng.uniform(-0.1,0.1,nlens)
    lens_e2 = rng.uniform(-0.1,0.1,nlens)

    dx = source_x - lens_x[:,np.newaxis]
    dy = source_y - lens_y[:,np.newaxis]
    rsq = dx**2 + dy**2
    r = np.sqrt(rsq)
    g = -(dx + 1j * dy)**2 / r**3
    g *= lens_mass[:,np.newaxis]
    g = np.sum(g,axis=0)
    source_g1 = g.real
    source_g2 = g.imag
    source_g1 += rng.normal(0,3.e-3)
    source_g2 += rng.normal(0,3.e-3)
    k = lens_mass[:,np.newaxis] / r
    source_k = np.sum(k,axis=0)

    cat1 = treecorr.Catalog(x=lens_x, y=lens_y, g1=lens_e1, g2=lens_e2, k=lens_mass)
    cat2 = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2, k=source_k)
    # Note: use cat2 to make the patches, since that's the one with a larger area.
    cat2p = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2, k=source_k,
                             npatch=npatch, rng=rng)
    cat1p = treecorr.Catalog(x=lens_x, y=lens_y, g1=lens_e1, g2=lens_e2, k=lens_mass,
                             patch_centers=cat2p.patch_centers)
    nwith0 = 0
    for i in range(npatch):
        n1 = np.sum(cat1p.w[cat1p.patch==i])
        n2 = np.sum(cat2p.w[cat2p.patch==i])
        print('%d\t%d\t%d\t%s'%(i,n1,n2,cat2p.patch_centers[i]))
        if n1 == 0: nwith0 += 1
    print('Found %s patches with no lenses'%nwith0)

    ng1 = treecorr.NGCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.)
    ng2 = treecorr.NGCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.)
    ng3 = treecorr.NGCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.)
    ng1.process(cat1, cat2)
    ng2.process(cat1, cat2p)
    ng3.process(cat1p, cat2p)
    np.testing.assert_allclose(ng2.weight, ng1.weight, rtol=0.05)
    np.testing.assert_allclose(ng3.weight, ng1.weight, rtol=0.05)
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=0.05, atol=1.e-3)
    np.testing.assert_allclose(ng3.xi, ng1.xi, rtol=0.05, atol=1.e-3)

    # No asserts here, but make sure covariance estimate doesn't crash.
    cov2j = ng2.estimate_cov('jackknife')
    cov3j = ng3.estimate_cov('jackknife')

    # Check warning emitted.
    with CaptureLog() as cl:
        ng3l = treecorr.NGCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.,
                                      logger=cl.logger)
        ng3l.process(cat1p, cat2p)
        ng3l.estimate_cov('jackknife')
    #print(cl.output)
    assert 'WARNING: A xi for calculating the jackknife covariance has no patch pairs.' in cl.output

    # Sample is even worse, and there is no way to compute the covariance.
    cov2s = ng2.estimate_cov('sample')
    with assert_raises(RuntimeError):
        cov3 = ng3.estimate_cov('sample')

    # Finally, if all the patch pairs have no counts, then this used to fail in a different way.
    # The easiest way to achieve this is to have the separation range be wrong.
    ng4 = treecorr.NGCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=1000., max_sep=5000.)
    ng4.process(cat1p, cat2p)
    ng4.estimate_cov('jackknife')
    np.testing.assert_array_equal(ng4.xi, 0.)
    np.testing.assert_array_equal(ng4.varxi, 0.)
    np.testing.assert_array_equal(ng4.cov, 0.)

    with assert_raises(RuntimeError):
        ng4.estimate_cov('sample')

    # With NN and NNN the check happens in a different place.
    nn = treecorr.NNCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.)
    nn.process(cat1p)
    cov = nn.estimate_cov('jackknife', func=lambda c: c.weight)
    np.testing.assert_array_equal(cov, 0.)
    with assert_raises(RuntimeError):
        cov = nn.estimate_cov('sample', func=lambda c: c.weight)

    nnn = treecorr.NNNCorrelation(bin_size=0.1, bin_slop=0.1, min_sep=10., max_sep=100.)
    nnn.process(cat1p)
    cov = nnn.estimate_cov('jackknife', func=lambda c: c.weight.ravel())
    np.testing.assert_array_equal(cov, 0.)
    with assert_raises(RuntimeError):
        cov = nnn.estimate_cov('sample', func=lambda c: c.weight.ravel())

    # With even more patches, there was also a different problem with the 1-2p correlation,
    # because some (0,k) pairs don't have any data, so don't end up in results dict.
    npatch = 100
    cat2p = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2, k=source_k,
                             npatch=npatch, rng=rng)
    ng2.process(cat1, cat2p)
    np.testing.assert_allclose(ng2.weight, ng1.weight, rtol=0.05)
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=0.05, atol=1.e-3)
    # The test here is really that the following all compute something.
    # They used to all fail with an error about some (0,k) pair not being in results dict.
    cov2j = ng2.estimate_cov('jackknife')
    with assert_raises(RuntimeError):
        # This one still gives a runtime error, but not the KeyError it used to raise.
        cov2s = ng2.estimate_cov('sample')
    cov2m = ng2.estimate_cov('marked_bootstrap')
    cov2b = ng2.estimate_cov('bootstrap')

@timer
def test_huge_npatch():
    # Test with npatch = 1000, which used to require >56GB of memory (~56B * npatch^3) when
    # making the list of patch pairs.  Especially bad for NN, where all patch combos are needed
    # to get ntot right.
    # Now this should only require ~56MB for that step, which is much more reasonable.

    if __name__ == '__main__':
        ngal = 200000
        npatch = 1000
    else:
        ngal = 2000
        # 50 isn't actually enough to trigger the old memory problem, but no need for
        # that on a regular test run.
        npatch = 50

    rng = np.random.RandomState(1234)

    # First without patches:
    x = rng.uniform(0,100, ngal)
    y = rng.uniform(0,100, ngal)
    k = rng.normal(1,0.1, (ngal,) )
    cat = treecorr.Catalog(x=x, y=y, k=k)
    kk1 = treecorr.KKCorrelation(bin_size=0.3, min_sep=1., max_sep=100., bin_slop=0.1)
    t0 = time.time()
    kk1.process(cat)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    # Now run with patches:
    catp = treecorr.Catalog(x=x, y=y, k=k, npatch=npatch)
    print('Patch\tNlens')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum([catp.patch==i])))
    kk2 = treecorr.KKCorrelation(bin_size=0.3, min_sep=1., max_sep=100., bin_slop=0.1)
    t0 = time.time()
    # I'm getting a KMP runtime error when using threads here.  Hence num_threads=1.
    # I suspect it's because there is so little to do, locks are cycling too fast for the OS
    # to correctly manage them.
    kk2.process(catp, num_threads=1)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('kk2.weight = ',kk2.weight)
    print('ratio = ',kk2.weight / kk1.weight)
    print('kk2.xi = ',kk2.xi)
    print('ratio = ',kk2.xi / kk1.xi)
    np.testing.assert_allclose(kk2.weight, kk1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kk2.xi, kk1.xi, rtol=2.e-2)

    # Check patch-based covariance calculations:
    t0 = time.time()
    cov = kk2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',cov.diagonal())

    t0 = time.time()
    cov = kk2.estimate_cov('sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxi = ',cov.diagonal())

    t0 = time.time()
    cov = kk2.estimate_cov('bootstrap')
    t1 = time.time()
    print('Time to calculate bootstrap covariance = ',t1-t0)
    print('varxi = ',cov.diagonal())

    t0 = time.time()
    cov = kk2.estimate_cov('marked_bootstrap')
    t1 = time.time()
    print('Time to calculate marked_bootstrap covariance = ',t1-t0)
    print('varxi = ',cov.diagonal())


if __name__ == '__main__':
    test_cat_patches()
    test_cat_centers()
    test_gg_jk()
    test_ng_jk()
    test_nn_jk()
    test_kappa_jk()
    test_save_patches()
    test_clusters()
    test_brute_jk()
    test_lowmem()
    test_config()
    test_finalize_false()
    test_empty_patches()
