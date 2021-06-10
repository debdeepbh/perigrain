import sys, os
sys.path.append(os.getcwd())

import numpy as np
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import time
import h5py
from sys import argv

import load_setup
from read_sim_output import read_plotinfo
from read_sim_output import extract_bulk

# fields to save
fields = [
        'time',
        'wall_loc',
        'wall_force',
        'volume_fraction',
        'bulk_damage',
        'particle_damage',
        'particle_force',
        ] 

loc = argv[-1]
plti = read_plotinfo(loc+'plotinfo.h5')
dim = plti.dim
plti.fc = int(argv[1])
plti.lc = int(argv[2])

# saving to file
combined_file = loc+'timestep_data.h5'
#######################################################################


# load the main experiment setup data
exp_b = load_setup.read_setup(loc+'all.h5')
######################################################################
def compute_all(t):
    return extract_bulk(t, loc, fields, exp_b, plti)

# print(compute_all(1))

start = time.time()
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()
print('')
print('Time taken: ', time.time() - start)


# convert to h5 file and save
print('Saving timestep data to:', combined_file)
with h5py.File(combined_file, 'w') as f:
    for i, field in enumerate(fields):
        col = [v[i] for v in V]
        # print(col)
        # f.create_dataset(field, data=np.array(col))
        f.create_dataset(field, data=col)

