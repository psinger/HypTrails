from __future__ import division

__author__ = 'psinger'

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#ax = fig.add_subplot(111, projection='3d')

def plot_bar(ax, dz):

    xpos = [1,1,1, 2,2,2,3,3,3]
    ypos = [1,2,3,1,2,3,1,2,3]
    zpos = [0,0,0,0,0,0,0,0,0]

    dx = [0.8]*9
    dy = [0.8]*9
    #dz = [1,1,1,1,1,1,1,1,1]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

    """
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
    """

    ax.set_xlabel('si')
    ax.set_ylabel('sj')
    ax.set_zlabel('nr. of chips')

    ax.invert_xaxis()


    ax.set_xticks(np.arange(min(xpos), max(xpos)+1, 1.0))
    ax.set_yticks(np.arange(min(ypos), max(ypos)+1, 1.0))
    ax.set_zticks(np.arange(min(dz), max(dz)+1, 1.0))


def plot_table(ax, table_vals):
    col_labels = ['$s_1$', '$s_2$', '$s_3$']
    row_labels = ['$s_1$', '$s_2$', '$s_3$']
    #table_vals = [[1.0, 0.1, 0.0], [0.1, 1.0, 0.8], [0.0, 0.8, 1.0]]

    vals = []
    for x in table_vals:
        vals.append(["%.2f" % a for a in x])

    ax.axis("off")

    the_table = ax.table(cellText=vals,
                      colWidths=[0.1] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    the_table.scale(4, 2)
#table.auto_set_font_size(False)

a = np.ndarray((3,3))
a[0,0] = 1.
a[0,1] = 0.1
a[0,2] = 0.
a[1,0] = 0.1
a[1,1] = 1.
a[1,2] = 0.9
a[2,0] = .0
a[2,1] = .9
a[2,2] = 1.

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8,8))

gs = gridspec.GridSpec(6, 3)

ax1 = fig.add_subplot(gs[0:2,0:1])

plot_table(ax1,[a[0,:], a[1,:], a[2,:]])

ax2 = fig.add_subplot(gs[0:2,1:3], projection='3d')

plot_bar(ax2, [1,1,1,1,1,1,1,1,1])

ax3 = fig.add_subplot(gs[2:4,0:1])

a = a / a.sum()

a = np.around(a.astype(np.double),2)

plot_table(ax3,[a[0,:], a[1,:], a[2,:]])

ax4 = fig.add_subplot(gs[2:4,2:3])

a = a * 9.

plot_table(ax4,[a[0,:], a[1,:], a[2,:]])

ax5 = fig.add_subplot(gs[4:6,0:2], projection='3d')

plot_bar(ax5, [1,1,1,1,1,1,1,1,1])

fig.set_tight_layout(True)

plt.show()

#plt.savefig("test.pdf")
