import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# saving to file
combined_file = 'output/timestep_data.h5'

#######################################################################
pair = ['volume_fraction', 'bulk_damage']
png_file = 'output/timestep_plot.png'

with h5py.File(combined_file, 'r') as f:
    print(f.keys())
    plt.plot(
            np.array(f[pair[0]]),
            np.array(f[pair[1]]),
            # label='label'
            )

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    # plt.gca().legend()
    plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

