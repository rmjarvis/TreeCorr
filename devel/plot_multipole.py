import treecorr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm


out_file_name0 = 'LOS1_ggg_bs0.fits'
gggm0 = treecorr.GGGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200, max_n=30,
                                bin_type='LogMultipole')
gggm0.read(out_file_name0)
y0 = [gggm0.gam0, gggm0.gam1, gggm0.gam2, gggm0.gam3]
w0 = gggm0.weight
theta2 = gggm0.meand2
theta3 = gggm0.meand3
norm0 = (theta2 * theta3)**1.5

# Note: index 30 in the last dimension is n=0.

out_file_name1 = 'LOS1_ggg.fits'
gggm1 = treecorr.GGGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200, max_n=30,
                                bin_type='LogMultipole')
gggm1.read(out_file_name1)
y1 = [gggm1.gam0, gggm1.gam1, gggm1.gam2, gggm1.gam3]
w1 = gggm1.weight
theta2 = gggm1.meand2
theta3 = gggm1.meand3
norm1 = (theta2 * theta3)**1.5

if 0:
    nnn_out_file_name1 = 'LOS1_nnn.fits'
    nnnm1 = treecorr.NNNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200, max_n=30,
                                    bin_type='LogMultipole')
    nnnm1.read(nnn_out_file_name1)
    w1 = nnnm1.weight

nx = w0.shape[0]
ny = w0.shape[1]
x = np.exp(np.linspace(np.log(0.5), np.log(200), nx+1))
y = np.exp(np.linspace(np.log(0.5), np.log(200), ny+1))

for n in [0, 1, 3, 10, 30]:
    print('n = ',n)

    fig, ax = plt.subplots(5,5, figsize=(18,16))

    for mu in range(4):
        y0norm = y0[mu][:,:,30+n] / norm0[:,:,30]
        y1norm = y1[mu][:,:,30+n] / norm1[:,:,30]

        c = ax[mu,0].pcolormesh(x, y, np.log10(np.abs(y0norm)))
        fig.colorbar(c, ax=ax[mu,0])
        c = ax[mu,1].pcolormesh(x, y, np.log10(np.abs(y1norm)))
        fig.colorbar(c, ax=ax[mu,1])
        c = ax[mu,2].pcolormesh(x, y, np.log10(np.abs(y1norm/y0norm)), norm=CenteredNorm())
        fig.colorbar(c, ax=ax[mu,2])
        c = ax[mu,3].pcolormesh(x, y, np.real(y1norm - y0norm)/np.abs(np.max(y0norm)), norm=CenteredNorm())
        fig.colorbar(c, ax=ax[mu,3])
        c = ax[mu,4].pcolormesh(x, y, np.imag(y1norm - y0norm)/np.abs(np.max(y0norm)), norm=CenteredNorm())
        fig.colorbar(c, ax=ax[mu,4])

        ax[mu,0].set_title('bs=0, log10(Y%d_%d)'%(mu,n))
        ax[mu,1].set_title('bs=1, log10(Y%d_%d)'%(mu,n))
        ax[mu,2].set_title('log10(ratio bs=1/bs=0)')
        ax[mu,3].set_title('diff/max (real)')
        ax[mu,4].set_title('diff/max (imag)')

    w0norm = w0[:,:,30+n] / norm0[:,:,30]
    w1norm = w1[:,:,30+n] / norm1[:,:,30]
    print('bs=0 diagonals:')
    print(w0norm.diagonal())
    print(w0norm.diagonal(1))
    print('bs=1 diagonals:')
    print(w1norm.diagonal())
    print(w1norm.diagonal(1))
    print('diffs')
    print((w1norm-w0norm).diagonal())
    print((w1norm-w0norm).diagonal(1))
    print('sum:')
    print(w0norm.diagonal()[:-1] + w0norm.diagonal(1) + w0norm.diagonal(-1))
    print(w1norm.diagonal()[:-1] + w1norm.diagonal(1) + w1norm.diagonal(-1))

    c = ax[4,0].pcolormesh(x, y, np.log10(np.abs(w0norm)))
    fig.colorbar(c, ax=ax[4,0])
    c = ax[4,1].pcolormesh(x, y, np.log10(np.abs(w1norm)))
    fig.colorbar(c, ax=ax[4,1])
    c = ax[4,2].pcolormesh(x, y, np.log10(np.abs(w1norm/w0norm)), norm=CenteredNorm())
    fig.colorbar(c, ax=ax[4,2])
    c = ax[4,3].pcolormesh(x, y, np.real(w1norm-w0norm)/np.abs(np.max(w0norm)), norm=CenteredNorm())
    fig.colorbar(c, ax=ax[4,3])
    c = ax[4,4].pcolormesh(x, y, np.imag(w1norm-w0norm)/np.abs(np.max(w0norm)), norm=CenteredNorm())
    fig.colorbar(c, ax=ax[4,4])

    ax[4,0].set_title('bs=0, log10(W_%d)'%n)
    ax[4,1].set_title('bs=1, log10(W_%d)'%n)
    ax[4,2].set_title('log10(ratio bs=1/bs=0)')
    ax[4,3].set_title('diff/max (real)')
    ax[4,4].set_title('diff/max (imag)')

    for i in range(5):
        for j in range(5):
            ax[i,j].set_xlabel('Theta1')
            ax[i,j].set_ylabel('Theta2')
            ax[i,j].set_xscale('log')
            ax[i,j].set_yscale('log')
            ax[i,j].set_box_aspect(1)

    fig.subplots_adjust(top=0.8)
    fig.suptitle("Comparison of Y and W for n=%d at bin_slop={0,1}"%n, fontsize=16, y=0.99)

    fig.tight_layout()
    fig.savefig('raw_multipole_n%s.png'%n)
