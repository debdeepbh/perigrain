import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# saving to file
combined_file = 'output/timestep_data.h5'

#######################################################################
# pair = ['volume_fraction', 'bulk_damage']
pair = ['time', 'particle_mean_vel']
# pair = ['time', 'particle_mean_CurrPos']

png_file = 'output/test_plot.png'

with h5py.File(combined_file, 'r') as f:
    print(f.keys())

    # print(np.array(f[pair[1]])[:,0])
    # norm = np.sum(np.array(f[pair[1]])[:,0]**2, axis=1)

    plt.plot(
            np.array(f[pair[0]]),
            # np.array(f[pair[1]]),
            np.array(f[pair[1]])[:,0],
            # norm,
            # label='label'
            )

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    # plt.gca().legend()
    plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

