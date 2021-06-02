import sys, os
sys.path.append(os.getcwd())

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sys import argv

from figsize import set_size
from matplotlib.pyplot import figure
fig_size = set_size('amsart')
# plt.style.use('tex')

# saving to file
combined_file = 'output/timestep_data.h5'

# save location
loc = argv[-2]
tag = argv[-1]


#######################################################################
pair = ['time', 'particle_mean_CurrPos']
png_file = loc+'/2d-collision-'+tag+'-CurrPos.png'

p = ['A', 'B']

figure(figsize=fig_size)
with h5py.File(combined_file, 'r') as f:
    print(f.keys())

    tt = np.array(f[pair[0]]) * 1e6
    for i in range(2):
        plt.plot(
                tt,
                1e3 * np.array(f[pair[1]])[:,i,1],
                label=p[i]
                )

    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('y-position of centroid (mm)')

    plt.xlim(min(tt), max(tt))

    plt.gca().legend()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

#######################################################################
pair = ['time', 'particle_mean_vel']
png_file = loc+'/2d-collision-'+tag+'-vel.png'

p = ['A', 'B']

figure(figsize=fig_size)
with h5py.File(combined_file, 'r') as f:
    print(f.keys())

    tt = np.array(f[pair[0]]) * 1e6
    for i in range(2):
        plt.plot(
                tt,
                np.array(f[pair[1]])[:,i,1],
                label=p[i]
                )

    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('y-velocity of centroid (m/s)')

    plt.xlim(min(tt), max(tt))

    plt.gca().legend()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

#######################################################################
# pair = ['time', 'particle_mean_vel']
# png_file = loc+'/2d-collision-'+tag+'-kinetic-energy.png'

# p = ['A', 'B']

# figure(figsize=set_size(latex_linewidth))
# with h5py.File(combined_file, 'r') as f:
    # print(f.keys())

    # tt = np.array(f[pair[0]]) * 1e6
    # plt.plot(
            # tt,
            # 0.5*np.array(f[pair[1]])[:,0,1]**2 +
            # 0.5*np.array(f[pair[1]])[:,1,1]**2,
            # # label=p[0]
            # )

    # plt.xlabel(r'Time ($\mu$s)')
    # plt.ylabel('Total kinetic energy')

    # plt.xlim(min(tt), max(tt))

    # # plt.gca().legend()
    # plt.savefig(png_file, bbox_inches='tight')
    # plt.close()
