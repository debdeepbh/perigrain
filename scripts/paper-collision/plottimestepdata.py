import sys, os
sys.path.append(os.getcwd())

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sys import argv

# saving to file
combined_file = 'output/timestep_data.h5'

# save location
loc = argv[-1]

#######################################################################
# pair = ['time', 'particle_mean_vel']
pair = ['time', 'particle_mean_CurrPos']

png_file = loc+'/2d-collision-elastic-CurrPos.png'

with h5py.File(combined_file, 'r') as f:
    print(f.keys())

    plt.plot(
            np.array(f[pair[0]]),
            np.array(f[pair[1]])[:,0,1],
            label='A'
            )
    plt.plot(
            np.array(f[pair[0]]),
            np.array(f[pair[1]])[:,1,1],
            label='B'
            )

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    # plt.gca().legend()
    # plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

