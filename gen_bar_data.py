import numpy as np
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt

from sys import argv

import load_setup
from read_sim_output import read_plotinfo, read_bar_val

# saving to file
npy_file = 'output/V.npy'

#######################################################################
loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')
dim = plti.dim

# override here
if len(argv) == 2:
    plti.fc = int(argv[1])
if len(argv) == 3:
    plti.lc = int(argv[2])

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()
print('Total volume:', vol)
print('Total mass:', mass)
######################################################################

def compute_all(t):
    # return read_bar_val(t, loc, dim, get_f_sum=True, input_exp_b=exp_b)
    return read_bar_val(t, loc, dim, input_exp_b=exp_b)

start = time.time()
# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()

# force = plti.fc * 

# print('Saving to disk: ', npy_file)
# np.save(npy_file, V)

strain = [(v.x_max - (-10e-3))/20e-3 for v in V]
print(strain)


# plt.plot(
