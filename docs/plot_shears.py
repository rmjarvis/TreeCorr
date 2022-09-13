import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import AxesZero

for xlabel, ylabel, filename in [('X', 'Y', 'xy_shear.png'), ('W', 'N', 'nw_shear.png')]:

    fig = plt.figure(figsize=(20,8))

    g1s = [0.8, -0.8, 0., 0.]
    g2s = [0., 0., 0.8, -0.8]

    # Make the x,y axes on each plot:
    # cf. https://matplotlib.org/stable/gallery/axisartist/demo_axisline_style.html
    for i in range(4):

        ax = fig.add_subplot(1,4,i+1, axes_class=AxesZero)
        g1 = g1s[i]
        g2 = g2s[i]

        for direction in ["xzero", "yzero"]:
            # adds arrows at the ends of each axis
            ax.axis[direction].set_axisline_style("-|>")

            # adds X and Y-axis from the origin
            ax.axis[direction].set_visible(True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)

        for direction in ["left", "right", "bottom", "top"]:
            # hides borders
            ax.axis[direction].set_visible(False)

        ax.text(1.9, -0.3, xlabel, fontsize=15)
        ax.text(-0.3, 1.9, ylabel, fontsize=15)
        ax.set_aspect('equal')

        x = np.linspace(-2, 2, 500)
        y = np.linspace(-2, 2, 500)
        x,y = np.meshgrid(x,y)
        r2 = ((1-g1)*x**2 + 2*g2*x*y + (1+g1)*y**2)
        z = np.exp(-r2*5)

        ax.imshow(z, cmap='Greys', extent=(-2,2,-2,2))

        if i == 0:
            ax.set_title(r'$g_1 > 0$, $g_2 = 0$', fontsize=25, pad=20)
        if i == 1:
            ax.set_title(r'$g_1 < 0$, $g_2 = 0$', fontsize=25, pad=20)
        if i == 2:
            ax.set_title(r'$g_1 = 0$, $g_2 > 0$', fontsize=25, pad=20)
        if i == 3:
            ax.set_title(r'$g_1 = 0$, $g_2 < 0$', fontsize=25, pad=20)

    plt.savefig(filename, bbox_inches='tight')
