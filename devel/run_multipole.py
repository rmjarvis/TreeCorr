import treecorr
import numpy as np

# http://cuillin.roe.ac.uk/~jharno/SLICS/MockProducts/KiDS450/GalCatalog_LOS1.fits
file_name = 'GalCatalog_LOS1.fits'

#bs = 0
#ext = '_bs0'
#bs = 1
#ext = ''
bs = 0.5
ext = '_bs05'
npatch = 1

cat = treecorr.Catalog(file_name, x_col='x_arcmin', y_col='y_arcmin',
                       g1_col='shear1', g2_col='shear2', npatch=npatch)

if 1:
    gggm = treecorr.GGGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200, max_n=30,
                                   verbose=2, bin_slop=bs, bin_type='LogMultipole')
    gggm.process(cat, low_mem=True)
    out_file_name = 'LOS1_ggg' + ext + '.fits'
    gggm.write(out_file_name)

    ggg = gggm.toSAS(phi_bin_size=0.05)
    map3 = ggg.calculateMap3()

    print('map3 = ',map3[0])

    out_file_name = 'LOS1_map3' + ext + '.npz'
    np.savez(out_file_name, map3)

else:
    nnnm = treecorr.NNNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200, max_n=30,
                                   verbose=2, bin_slop=bs, bin_type='LogMultipole')
    nnnm.process(cat)
    out_file_name = 'LOS1_nnn' + ext + '.fits'
    nnnm.write(out_file_name)
