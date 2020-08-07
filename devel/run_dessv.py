
# Follow a realistic workflow for a very large input catalog.
# Not that DES SV is actually all that large, but do the same procedure as you
# would do on say DES Y5 where not everything fits in memory at once.

import treecorr
import fitsio
import numpy as np
import os
import time

from test_helper import get_from_wiki, profile

def download():
    # The download can be a bit slow, and the files need to be merged to get something
    # that includes both ra/dec and e1,e2.  So we did this block once and uploaded
    # the results to the wiki.  Normal test running can just get the result from the wiki.

    # Download the public DES SV files (if not already downloaded)
    host = 'http://desdr-server.ncsa.illinois.edu/despublic/sva1_files/'
    lens_file = 'redmagic_sva1_public_v6.3_faint.fits.gz'
    get_from_wiki(lens_file, host=host)
    lens_file = os.path.join('data',lens_file)

    ngmix_file = 'sva1_gold_r1.0_ngmix.fits.gz'
    get_from_wiki(ngmix_file, host=host)
    ngmix_file = os.path.join('data',ngmix_file)

    info_file = 'sva1_gold_r1.0_wlinfo.fits.gz'
    get_from_wiki(info_file, host=host)
    info_file = os.path.join('data',info_file)

    source_file = os.path.join('data','sva1_gold_r1.0_merged.fits')
    if not os.path.exists(source_file):
        print('Reading ngmix_data')
        ngmix_data = fitsio.read(ngmix_file)
        print('Reading info_data')
        info_data = fitsio.read(info_file)

        col_names = ['RA', 'DEC']
        cols = [info_data[n] for n in col_names]  # These come from wlinfo
        col_names += ['E_1', 'E_2', 'W']
        cols += [ngmix_data[n] for n in col_names[2:]]  # These are in ngmix

        # combine the two sensitivity estimates
        col_names += ['SENS']
        cols += [(ngmix_data['SENS_1'] + ngmix_data['SENS_2'])/2.]

        # Save time by cutting to only flag != 0 objects here.
        use = info_data['NGMIX_FLAG'] == 0
        print('total number of galaxies = ',len(use))
        print('number to use = ',np.sum(use))
        cols = [col[use] for col in cols]
        print('writing merged file: ',source_file)
        treecorr.util.gen_write(source_file, col_names, cols, file_type='FITS')

    return source_file, lens_file

def run_dessv(source_file, lens_file, use_patches):
    if use_patches:
        # First determine patch centers using 1/10 of the total source catalog.
        # Only need positions for this.
        # This isn't strictly necessary.  It's trying to showcase how to do this when the
        # whole catalog doesn't fit in memory.  If it all fits, then fine to use the full
        # source catalog to run KMeans.
        print('Read 1/10 of source catalog for kmeans patches')
        npatch = 128
        small_cat = treecorr.Catalog(source_file, ra_col='RA', dec_col='DEC', file_type='FITS',
                                     ra_units='deg', dec_units='deg', every_nth=10, npatch=npatch,
                                     verbose=2)

        # Write the patch centers
        patch_file = os.path.join('output','test_dessv_patches.fits')
        small_cat.write_patch_centers(patch_file)
        print('wrote patch centers file ',patch_file)
        #print('centers = ',small_cat.patch_centers)

        patch_kwargs = dict(patch_centers=patch_file, save_patch_dir='output')
    else:
        patch_kwargs = {}

    # Now load the full catalog using these patch centers.
    # Note: they need to use the same patch_file!
    print('make source catalog')
    sources = treecorr.Catalog(source_file, ra_col='RA', dec_col='DEC', file_type='FITS',
                               ra_units='deg', dec_units='deg',
                               g1_col='E_1', g2_col='E_2', w_col='W', k_col='SENS',
                               **patch_kwargs)

    print('make lens catalog')
    lenses = treecorr.Catalog(lens_file, ra_col='RA', dec_col='DEC', file_type='FITS',
                              ra_units='deg', dec_units='deg',
                              **patch_kwargs)

    # Configuration of correlation functions.
    bin_config = dict(bin_size=0.2, min_sep=10., max_sep=200.,
                      bin_slop=0.1, sep_units='arcmin',
                      verbose=1, output_dots=False)
    if use_patches:
        bin_config['var_method'] = 'jackknife'

    # Run the various 2pt correlations.  I'll skip NN here, to avoid dealing with randoms,
    # but that could be included as well.
    gg = treecorr.GGCorrelation(bin_config)
    ng = treecorr.NGCorrelation(bin_config)

    print('Process gg')
    gg.process(sources)
    print('Process ng')
    ng.process(lenses, sources)

    print('gg.xip = ',gg.xip)
    print('gg.xim = ',gg.xim)
    print('ng.xi = ',ng.xi)
    nbins = len(ng.xi)

    method = 'jackknife' if use_patches else 'shot'
    cov = treecorr.estimate_multi_cov([ng,gg], method)
    print('cov = ',cov)
    print('sigma = ',np.sqrt(cov.diagonal()))
    print('S/N = ',np.concatenate([gg.xip,gg.xim,ng.xi]) / np.sqrt(cov.diagonal()))

    assert len(gg.xip) == nbins
    assert len(gg.xim) == nbins
    assert cov.shape == (3*nbins, 3*nbins)

    # Apply sensitivities.
    print('Process kk')
    kk = treecorr.KKCorrelation(bin_config)
    print('Process nk')
    nk = treecorr.NKCorrelation(bin_config)

    kk.process(sources)
    nk.process(lenses, sources)

    ng.xi /= nk.xi
    gg.xip /= kk.xi
    gg.xim /= kk.xi

    # This makes the assumption that the power spectrum of the sensitivity is effectively uniform
    # across the survey.  So don't bother propagating covariance of sens.
    cov /= np.outer(np.concatenate([nk.xi,kk.xi,kk.xi]),
                    np.concatenate([nk.xi,kk.xi,kk.xi]))

    print('gg.xip => ',gg.xip)
    print('gg.xim => ',gg.xim)
    print('ng.xi => ',ng.xi)
    print('cov => ',cov)


if __name__ == '__main__':
    source_file, lens_file = download()
    t0 = time.time()
    print('Run without patches:')
    run_dessv(source_file, lens_file, use_patches=False)
    t1 = time.time()
    print('Run with patches:')
    with profile():
        run_dessv(source_file, lens_file, use_patches=True)
    t2 = time.time()
    print('Time for normal non-patch run = ',t1-t0)
    print('Time for patch run with jackknife covariance = ',t2-t1)
